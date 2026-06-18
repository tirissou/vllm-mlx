# SPDX-License-Identifier: Apache-2.0
"""Differential diagnostic for cache-hit vs fresh-prefill drift.

Goal: tell apart three hypotheses for why a cache-hit conversation drifts in
quality over multiple turns while a fresh-prefill conversation does not.

  (A) Single-segment round-trip lossiness
      segment(extract(cache_fresh)) -> assemble(...) vs cache_fresh
      Tests whether one reconstruction loses information at all.
      Expected non-zero for rotating layers (dequant + ring re-rotation).
      Expected zero for full-attention bf16 layers.

  (B) Multi-turn path-merge accumulation
      Build a cache by prefilling each turn delta in sequence, segmenting
      each via slice_kv_to_delta, merging segments per layer, then assembling.
      Diff against a single fresh prefill of the same total tokens.
      Tests whether the slice/concat path agrees with one big prefill.

Per-turn output (one row per (turn_index, class_name)):

  turn  layer_class       n_layers  max_abs   mean_abs  shape_drift
  ----  ----------------  --------  --------  --------  -----------
  1     KVCache                  7  0.00e+00  0.00e+00  0
  1     RotatingKVCache         28  3.91e-03  1.20e-04  0
  ...

Interpretation:
  - A diffs > 0 for a layer class -> reconstruction is lossy for that class.
  - B diffs >> A diffs           -> accumulation amplifies the error.
  - B diffs > 0 while A diffs = 0 -> slice/concat boundary is the bug.

Usage:
  python scripts/diag_cache_drift.py \
      --model mlx-community/gemma-4-e4b-it-4bit \
      --turns 6

This script does not test the model's output text — it only diffs cache state.
That is on purpose: drift in cache state is a leading indicator of quality drift
and is far less noisy than sampling-level comparison.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from vllm_mlx.cache_translator import assemble, segment, slice_kv_to_delta
from vllm_mlx.cache_types import KVQuantPolicy
from vllm_mlx.kv_cache import QuantizedArray, extract_layer_state


# A 6-turn chat that exercises the after-tool-call pattern the user described.
DEFAULT_CHAT = [
    {"role": "system", "content": "You are a helpful assistant. Keep replies short."},
    {"role": "user", "content": "List the inner planets, one per line."},
    {"role": "assistant", "content": "Mercury\nVenus\nEarth\nMars"},
    {"role": "user", "content": "Which of those has the most moons?"},
    {"role": "assistant", "content": "Mars, with two moons (Phobos and Deimos)."},
    {"role": "user", "content": "What are their names?"},
    {"role": "assistant", "content": "Phobos and Deimos."},
    {"role": "user", "content": "Which is bigger?"},
    {"role": "assistant", "content": "Phobos is the larger of the two."},
    {"role": "user", "content": "By how much?"},
    {"role": "assistant", "content": "Phobos is roughly twice the diameter of Deimos."},
    {"role": "user", "content": "Summarise this conversation in one sentence."},
]


@dataclass
class LayerDiff:
    class_name: str
    layer_index: int
    # Whole-cache diff (positions 0..ref_offset)
    max_abs_k: float
    max_abs_v: float
    mean_abs_k: float
    mean_abs_v: float
    shape_drift: bool
    # Per-region diff (None if not applicable, e.g. round-trip A or turn 1)
    prefix_max_k: float | None = None
    prefix_max_v: float | None = None
    prefix_mean_k: float | None = None
    prefix_mean_v: float | None = None
    new_max_k: float | None = None
    new_max_v: float | None = None
    new_mean_k: float | None = None
    new_mean_v: float | None = None


def _install_unfused_sdpa() -> None:
    """Replace mx.fast.scaled_dot_product_attention with an explicit Python impl.

    Mirrors the pattern in vllm_mlx/patches/mlx_lm_quantized_sdpa.py:
        scores = Q @ K^T * scale
        scores += additive mask (lower-right causal)
        scores = softmax(precise=True)
        out = scores @ V

    Lower-right alignment matches `mask="causal"`: for shapes (T_q, T_kv) with
    T_q <= T_kv, query i (in 0..T_q-1) attends to keys [0..(T_kv - T_q + i)].

    Also walks sys.modules and overwrites mlx_lm.models.*'s already-bound
    `scaled_dot_product_attention` so models that did
    `from .base import scaled_dot_product_attention` pick up the new impl.
    """
    import sys
    import mlx_lm.models.base as _base

    def _unfused(queries, keys, values, *, scale, mask=None, sinks=None):
        # GQA: keys / values may have fewer heads than queries.
        n_q = queries.shape[1]
        n_kv = keys.shape[1]
        if n_q != n_kv:
            repeats = n_q // n_kv
            keys = mx.repeat(keys, repeats, axis=1)
            values = mx.repeat(values, repeats, axis=1)

        scores = mx.matmul(queries * scale, keys.swapaxes(-1, -2))
        qL, kL = scores.shape[-2], scores.shape[-1]

        if mask is None or (isinstance(mask, str) and mask == "causal"):
            q_idx = mx.arange(kL - qL, kL)
            k_idx = mx.arange(kL)
            causal = q_idx[:, None] >= k_idx[None]
            scores = mx.where(causal, scores, mx.finfo(scores.dtype).min)
        elif isinstance(mask, mx.array):
            if mask.dtype == mx.bool_:
                scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
            else:
                scores = scores + mask
        # Else: unrecognised mask type — leave scores alone (will likely error in softmax).

        if sinks is not None:
            raise NotImplementedError("unfused SDPA does not implement attention sinks")

        scores = mx.softmax(scores, axis=-1, precise=True)
        return mx.matmul(scores, values)

    mx.fast.scaled_dot_product_attention = _unfused

    # mlx-lm wraps mx.fast.scaled_dot_product_attention in its own dispatcher
    # function and many model modules did `from .base import scaled_dot_product_attention`
    # at import time. Patch the base wrapper too, and any already-imported model
    # module that has a rebound name.
    def _patched_wrapper(queries, keys, values, cache, scale, mask, sinks=None):
        return _unfused(queries, keys, values, scale=scale, mask=mask, sinks=sinks)

    _base.scaled_dot_product_attention = _patched_wrapper
    for mod_name, module in list(sys.modules.items()):
        if (
            mod_name.startswith("mlx_lm.models.")
            and mod_name != "mlx_lm.models.base"
            and hasattr(module, "scaled_dot_product_attention")
        ):
            module.scaled_dot_product_attention = _patched_wrapper


def _flatten_arrays(x):
    """Yield every mx.array reachable through tuples/lists/QuantizedArray."""
    if isinstance(x, mx.array):
        yield x
    elif isinstance(x, QuantizedArray):
        yield x.packed; yield x.scales; yield x.biases
    elif isinstance(x, (tuple, list)):
        for v in x:
            yield from _flatten_arrays(v)


def _to_float_array(x) -> mx.array:
    """Dequantize a QuantizedArray to bf16, or return the array unchanged."""
    if isinstance(x, QuantizedArray):
        return mx.dequantize(
            x.packed, x.scales, x.biases,
            group_size=64, bits=8,  # group_size/bits read from segment if needed
        )
    return x


def _diff(a, b) -> tuple[float, float, bool]:
    """Return (max_abs, mean_abs, shape_drift) between two arrays/QuantizedArrays."""
    a_f = _to_float_array(a)
    b_f = _to_float_array(b)
    if a_f.shape != b_f.shape:
        return float("inf"), float("inf"), True
    d = mx.abs(a_f.astype(mx.float32) - b_f.astype(mx.float32))
    mx.eval(d)
    return float(d.max().item()), float(d.mean().item()), False


def _diff_caches(
    label: str,
    cache_ref: list,
    cache_test: list,
    split_at: int | None = None,
) -> list[LayerDiff]:
    """Per-layer K/V diff between two live mlx-lm cache lists.

    If split_at is provided, also reports a positional split into
    "prefix" ([0:split_at], the tokens that existed before the new chunk)
    vs "new"  ([split_at:ref_offset], the tokens added by the new chunk).

    For RotatingKVCache the buffer is a ring rather than a chronological
    sequence, so the prefix/new split is approximate; we still report it
    but note that the positional axis is ring-position, not absolute.
    """
    diffs: list[LayerDiff] = []
    assert len(cache_ref) == len(cache_test), (
        f"[{label}] layer count mismatch: ref={len(cache_ref)} test={len(cache_test)}"
    )
    for i, (lr, lt) in enumerate(zip(cache_ref, cache_test)):
        cname = type(lr).__name__
        raw_off = getattr(lr, "offset", 0)
        ref_offset = int(raw_off.item()) if hasattr(raw_off, "shape") else int(raw_off)
        kr = lr.keys[..., :ref_offset, :] if hasattr(lr, "keys") and lr.keys is not None else None
        vr = lr.values[..., :ref_offset, :] if hasattr(lr, "values") and lr.values is not None else None
        kt = lt.keys[..., :ref_offset, :] if hasattr(lt, "keys") and lt.keys is not None else None
        vt = lt.values[..., :ref_offset, :] if hasattr(lt, "values") and lt.values is not None else None
        if kr is None or kt is None:
            diffs.append(LayerDiff(cname, i, 0.0, 0.0, 0.0, 0.0, False))
            continue
        mk, ak, sd_k = _diff(kr, kt)
        mv, av, sd_v = _diff(vr, vt)
        d = LayerDiff(cname, i, mk, mv, ak, av, sd_k or sd_v)
        if split_at is not None and 0 < split_at < ref_offset:
            pkr = kr[..., :split_at, :]
            pvr = vr[..., :split_at, :]
            pkt = kt[..., :split_at, :]
            pvt = vt[..., :split_at, :]
            nkr = kr[..., split_at:, :]
            nvr = vr[..., split_at:, :]
            nkt = kt[..., split_at:, :]
            nvt = vt[..., split_at:, :]
            d.prefix_max_k, d.prefix_mean_k, _ = _diff(pkr, pkt)
            d.prefix_max_v, d.prefix_mean_v, _ = _diff(pvr, pvt)
            d.new_max_k, d.new_mean_k, _ = _diff(nkr, nkt)
            d.new_max_v, d.new_mean_v, _ = _diff(nvr, nvt)
        diffs.append(d)
    return diffs


def _print_diff_table(label: str, turn_idx: int, diffs: list[LayerDiff]) -> None:
    """Aggregate per-class then print a row per class, with prefix/new split if present."""
    by_class: dict[str, list[LayerDiff]] = {}
    for d in diffs:
        by_class.setdefault(d.class_name, []).append(d)

    has_split = any(d.prefix_max_k is not None for d in diffs)
    print(f"\n[{label}] turn={turn_idx}")
    if has_split:
        print(f"  {'layer_class':<22} {'n':>3}  "
              f"{'prefix_maxK':>11} {'prefix_maxV':>11} "
              f"{'new_maxK':>10} {'new_maxV':>10}  "
              f"{'prefix_meanV':>12} {'new_meanV':>10}")
        for cname, items in by_class.items():
            def _maxof(attr):
                vs = [getattr(d, attr) for d in items if getattr(d, attr) is not None]
                return max(vs) if vs else 0.0
            def _meanof(attr):
                vs = [getattr(d, attr) for d in items if getattr(d, attr) is not None]
                return (sum(vs) / len(vs)) if vs else 0.0
            print(f"  {cname:<22} {len(items):>3}  "
                  f"{_maxof('prefix_max_k'):>11.3e} {_maxof('prefix_max_v'):>11.3e} "
                  f"{_maxof('new_max_k'):>10.3e} {_maxof('new_max_v'):>10.3e}  "
                  f"{_meanof('prefix_mean_v'):>12.3e} {_meanof('new_mean_v'):>10.3e}")
    else:
        print(f"  {'layer_class':<24} {'n':>4}  {'max_abs_K':>10} {'max_abs_V':>10}  "
              f"{'mean_abs_K':>10} {'mean_abs_V':>10}  shape_drift")
        for cname, items in by_class.items():
            max_k = max(d.max_abs_k for d in items)
            max_v = max(d.max_abs_v for d in items)
            mean_k = sum(d.mean_abs_k for d in items) / len(items)
            mean_v = sum(d.mean_abs_v for d in items) / len(items)
            sd = sum(1 for d in items if d.shape_drift)
            print(f"  {cname:<24} {len(items):>4}  {max_k:>10.3e} {max_v:>10.3e}  "
                  f"{mean_k:>10.3e} {mean_v:>10.3e}  {sd}")


def _prefill(model, tokens: mx.array, cache: list) -> None:
    """One forward pass that populates `cache` in place. Eager-evaluates outputs."""
    out = model(tokens[None], cache=cache)
    mx.eval(out)


def _clone_cache_state(cache: list) -> list[dict]:
    """Snapshot a live cache to the dict form `segment()` expects."""
    states = []
    for layer in cache:
        d = extract_layer_state(layer)
        if d is not None:
            states.append(d)
    return states


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/gemma-4-e4b-it-4bit")
    ap.add_argument("--turns", type=int, default=6,
                    help="Number of user/assistant turn boundaries to walk.")
    ap.add_argument("--full-bits", type=int, default=None,
                    help="Override KVQuantPolicy.full_bits (None = bf16, matches "
                         "kv_cache_quantization=False).")
    ap.add_argument("--sliding-bits", type=int, default=None,
                    help="Override KVQuantPolicy.sliding_bits (None = bf16, default).")
    ap.add_argument("--unfused-sdpa", action="store_true",
                    help="Monkey-patch mx.fast.scaled_dot_product_attention with an "
                         "explicit Python implementation (matmul + lower-right causal "
                         "mask + precise fp32 softmax + matmul). Tests whether the "
                         "fused Metal kernel's shape-dependent behaviour is the cause "
                         "of chunked-prefill drift.")
    ap.add_argument("--fp32", action="store_true",
                    help="Cast model parameters to fp32 after load. Tests whether bf16 "
                         "matmul shape-dependence is the cause: if C collapses to ~0 "
                         "under fp32, bf16 precision is the culprit. Use a non-quantized "
                         "model for a clean test (e.g. mlx-community/Qwen3-0.6B).")
    args = ap.parse_args()

    if args.unfused_sdpa:
        _install_unfused_sdpa()
        print("unfused SDPA installed (matmul + explicit causal mask + precise softmax)")

    print(f"loading model: {args.model}")
    model, tokenizer = load(args.model)
    if args.fp32:
        from mlx.utils import tree_map
        _float_dtypes = {mx.float16, mx.bfloat16}
        def _cast(x):
            if not isinstance(x, mx.array):
                return x
            if x.dtype in _float_dtypes:
                return x.astype(mx.float32)
            return x  # leave packed uint32 weights, int indices, etc. alone
        model.update(tree_map(_cast, model.parameters()))
        mx.eval(model.parameters())
        print("model parameters cast to fp32 (scales/biases for quantized layers)")
    policy = KVQuantPolicy(
        sliding_bits=args.sliding_bits,
        full_bits=args.full_bits,
    )
    print(f"policy: {policy.describe()}")

    # Build cumulative token prefixes — one per turn boundary.
    chat = DEFAULT_CHAT[: 1 + 2 * args.turns]
    turn_prompts: list[mx.array] = []
    prev_len = 0
    deltas: list[mx.array] = []
    for end in range(2, len(chat) + 1):
        prompt = tokenizer.apply_chat_template(
            chat[:end], tokenize=False, add_generation_prompt=False,
        )
        tokens = mx.array(tokenizer.encode(prompt))
        if tokens.shape[0] <= prev_len:
            continue  # encoder produced a degenerate result; skip this boundary
        turn_prompts.append(tokens)
        deltas.append(tokens[prev_len:])
        prev_len = tokens.shape[0]

    print(f"running {len(turn_prompts)} turn boundaries, "
          f"token counts: {[int(t.shape[0]) for t in turn_prompts]}")

    # ---- Round-trip A: single-segment round-trip at each turn boundary ----
    for turn_i, full_tokens in enumerate(turn_prompts, start=1):
        cache_fresh = make_prompt_cache(model)
        _prefill(model, full_tokens, cache_fresh)

        states = _clone_cache_state(cache_fresh)
        kv_segs, rec_segs = segment(states, policy=policy)
        cache_round = assemble(kv_segs, rec_segs)

        diffs = _diff_caches(f"A:single-rt turn={turn_i}", cache_fresh, cache_round)
        _print_diff_table("A:single-rt", turn_i, diffs)

    # ---- Round-trip B: per-turn deltas, path-merged, then assembled ----
    # Maintain per-layer-index lists of segments; merge via subclass merge_path.
    cache_running = make_prompt_cache(model)
    per_layer_kv_segs: dict[int, list] = {}
    per_layer_rec_segs: dict[int, list] = {}
    prev_end = 0

    for turn_i, (full_tokens, delta) in enumerate(zip(turn_prompts, deltas), start=1):
        _prefill(model, delta, cache_running)
        states = _clone_cache_state(cache_running)
        sliced = slice_kv_to_delta(states, prev_end)
        kv_segs, rec_segs = segment(sliced, policy=policy)

        for s in kv_segs:
            if s is not None:
                per_layer_kv_segs.setdefault(s.layer_index, []).append(s)
        for r in rec_segs:
            if r is not None:
                li = r.metadata["layer_index"]
                per_layer_rec_segs.setdefault(li, []).append(r)

        # Build a merged cache from everything accumulated so far.
        merged_kv = []
        for li in sorted(per_layer_kv_segs):
            path = per_layer_kv_segs[li]
            merged_kv.append(path[0].merge_path(path))
        merged_rec = [v[-1] for _, v in sorted(per_layer_rec_segs.items())]
        cache_b = assemble(merged_kv, merged_rec)

        # Fresh prefill of the same total prompt to use as the reference.
        cache_fresh_full = make_prompt_cache(model)
        _prefill(model, full_tokens, cache_fresh_full)

        diffs = _diff_caches(
            f"B:path-merge turn={turn_i}", cache_fresh_full, cache_b,
            split_at=prev_end,
        )
        _print_diff_table("B:path-merge", turn_i, diffs)
        prev_end = int(full_tokens.shape[0])

    # ---- Round-trip C: bare chunked prefill, no segment/assemble involved ----
    # Mirrors mllm_batch_generator.py:1062-1069 exactly: chunked model() calls,
    # eval between chunks. The cache_translator is NOT in the loop, so any
    # diff here is purely the model's continuation-prefill behaviour.
    print("\n---- C: bare chunked prefill (no cache_translator in the loop) ----")
    cache_chunked = make_prompt_cache(model)
    prev_end = 0
    for turn_i, (full_tokens, delta) in enumerate(zip(turn_prompts, deltas), start=1):
        out = model(delta[None], cache=cache_chunked)
        eval_args = [out]
        for layer in cache_chunked:
            for arr in _flatten_arrays(getattr(layer, "state", ())):
                eval_args.append(arr)
        mx.eval(*eval_args)

        cache_fresh_full = make_prompt_cache(model)
        _prefill(model, full_tokens, cache_fresh_full)

        diffs = _diff_caches(
            f"C:chunked turn={turn_i}", cache_fresh_full, cache_chunked,
            split_at=prev_end,
        )
        _print_diff_table("C:chunked", turn_i, diffs)
        prev_end = int(full_tokens.shape[0])

    # ---- Round-trip D: does our padding/reconstruction affect continuation? ----
    # For each turn, build two caches that are bit-equal in their active data
    # (A=0 proves that), then continue both with the next turn's delta. If the
    # results differ, our reconstructed buffer interacts differently with
    # mlx-lm's update_and_fetch (likely via the padding/grow boundary).
    print("\n---- D: continuation on reconstructed vs stock cache ----")
    for turn_i in range(1, len(turn_prompts)):
        turn_tokens = turn_prompts[turn_i - 1]
        next_delta = deltas[turn_i]
        ref_offset_after = int(turn_prompts[turn_i].shape[0])

        # Path 1: stock cache after turn_i prefill, then continue
        cache_stock = make_prompt_cache(model)
        _prefill(model, turn_tokens, cache_stock)
        _ = model(next_delta[None], cache=cache_stock)
        eval_args = []
        for layer in cache_stock:
            for arr in _flatten_arrays(getattr(layer, "state", ())):
                eval_args.append(arr)
        mx.eval(*eval_args)

        # Path 2: stock cache after turn_i, round-tripped via assemble (padded),
        # then continue with the same delta
        cache_for_rt = make_prompt_cache(model)
        _prefill(model, turn_tokens, cache_for_rt)
        states = _clone_cache_state(cache_for_rt)
        kv_segs, rec_segs = segment(states, policy=policy)
        cache_recon = assemble(kv_segs, rec_segs)
        _ = model(next_delta[None], cache=cache_recon)
        eval_args = []
        for layer in cache_recon:
            for arr in _flatten_arrays(getattr(layer, "state", ())):
                eval_args.append(arr)
        mx.eval(*eval_args)

        diffs = _diff_caches(
            f"D:padded-vs-stock turn={turn_i}->{turn_i+1}",
            cache_stock, cache_recon,
            split_at=int(turn_tokens.shape[0]),
        )
        _print_diff_table("D:padded-vs-stock", turn_i, diffs)

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
