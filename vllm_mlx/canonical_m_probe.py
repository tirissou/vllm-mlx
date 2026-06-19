"""Pure functions for canonical-M probing.

Used by both the production verify probe in prefix_cache_adapters._scan_chunking
and the standalone CLI scripts/find_canonical_m.py.
"""

from __future__ import annotations

import mlx.core as mx


def _schedules_for(n_tokens: int) -> list[tuple[str, int, int]]:
    """Build chunking schedules (name, divisor, chunk_size) for N, skipping <8."""
    out: list[tuple[str, int, int]] = []
    for divisor in (1, 2, 4, 8):
        if n_tokens % divisor != 0:
            continue
        chunk = n_tokens // divisor
        if chunk < 8:
            continue
        out.append((f"{divisor}x{chunk}", divisor, chunk))
    return out


def run_chunking_probe(
    model,
    n_tokens: int,
    batch_size: int,
    layer_indices: list[int] | None = None,
    *,
    vocab_size: int | None = None,
) -> dict:
    """Run chunking schedules over the same physical token range and diff K,V.

    Returns dict with keys:
      - schedules: list[(name, chunk_size)]
      - diffs: dict[(layer_idx, "K"|"V"), dict[(name_i, name_j), float]]
              symmetric, excluding self-diff
      - layer_types: dict[layer_idx, class_name_str]
    """
    from mlx_lm.models.cache import make_prompt_cache

    schedules = _schedules_for(n_tokens)
    if not schedules:
        raise ValueError(f"No usable schedules for N={n_tokens}")

    if vocab_size is None:
        # Try a few common attrs; fall back to 32000.
        for attr in ("vocab_size",):
            vocab_size = getattr(getattr(model, "args", model), attr, None)
            if vocab_size is not None:
                break
        vocab_size = vocab_size or 32000

    tokens = mx.random.randint(0, vocab_size, (batch_size, n_tokens))

    sample_cache = make_prompt_cache(model)
    n_layers = len(sample_cache)
    if layer_indices is None:
        layer_indices = sorted({0, 1, max(1, n_layers // 2), n_layers - 1})
    layer_types = {L: type(sample_cache[L]).__name__ for L in layer_indices if L < n_layers}
    del sample_cache
    mx.clear_cache()

    captures: list[tuple[str, dict[int, tuple[mx.array, mx.array]]]] = []
    for name, _div, chunk_size in schedules:
        cache = make_prompt_cache(model)
        cursor = 0
        while cursor < n_tokens:
            chunk = tokens[:, cursor : cursor + chunk_size]
            model(chunk, cache=cache)
            cursor += chunk_size
        dump: dict[int, tuple[mx.array, mx.array]] = {}
        for L in layer_indices:
            if L >= len(cache):
                continue
            layer = cache[L]
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if isinstance(keys, mx.array) and isinstance(values, mx.array):
                if keys.shape[-2] < n_tokens or values.shape[-2] < n_tokens:
                    continue
                k = mx.contiguous(keys[..., :n_tokens, :].astype(mx.float32))
                v = mx.contiguous(values[..., :n_tokens, :].astype(mx.float32))
                mx.eval(k, v)
                dump[L] = (k, v)
        captures.append((name, dump))
        del cache
        mx.clear_cache()

    diffs: dict[tuple[int, str], dict[tuple[str, str], float]] = {}
    for L in layer_indices:
        for kind_idx, kind in enumerate(("K", "V")):
            cell: dict[tuple[str, str], float] = {}
            for i, (ni, di) in enumerate(captures):
                for j, (nj, dj) in enumerate(captures):
                    if i == j or L not in di or L not in dj:
                        continue
                    ai = di[L][kind_idx]
                    aj = dj[L][kind_idx]
                    cell[(ni, nj)] = float(mx.abs(ai - aj).max().item())
            if cell:
                diffs[(L, kind)] = cell

    return {
        "schedules": [(name, chunk) for name, _div, chunk in schedules],
        "diffs": diffs,
        "layer_types": layer_types,
    }


def compute_canonical_band(probe_result: dict) -> list[int]:
    """Return the largest contiguous set of chunk sizes M whose pairwise diffs
    are exactly zero across every probed (layer, K|V) pair."""
    name_to_M = {name: M for name, M in probe_result["schedules"]}
    all_M = sorted(set(name_to_M.values()))

    # For each chunk size M, gather names that produce it.
    M_to_names: dict[int, list[str]] = {}
    for name, M in probe_result["schedules"]:
        M_to_names.setdefault(M, []).append(name)

    def is_pair_zero(name_i: str, name_j: str) -> bool:
        for cell in probe_result["diffs"].values():
            d = cell.get((name_i, name_j))
            if d is None:
                d = cell.get((name_j, name_i))
            if d is None:
                continue
            if d != 0.0:
                return False
        return True

    def M_pair_zero(Mi: int, Mj: int) -> bool:
        ni = M_to_names[Mi][0]
        nj = M_to_names[Mj][0]
        return is_pair_zero(ni, nj)

    # Find all contiguous bands where every pair has zero diff
    bands: list[list[int]] = []
    for start_idx in range(len(all_M)):
        for end_idx in range(start_idx, len(all_M)):
            # Check if all_M[start_idx:end_idx+1] form a zero-diff band
            band = all_M[start_idx:end_idx+1]
            is_valid = True
            for i in range(len(band)):
                for j in range(i+1, len(band)):
                    if not M_pair_zero(band[i], band[j]):
                        is_valid = False
                        break
                if not is_valid:
                    break
            if is_valid and len(band) > 0:
                bands.append(band)

    if not bands:
        return []
    return max(bands, key=len)


def intersect_per_batch_bands(per_batch: dict[int, list[int]]) -> list[int]:
    """Return sorted intersection of M values across batch sizes."""
    if not per_batch:
        return []
    sets = [set(band) for band in per_batch.values()]
    return sorted(set.intersection(*sets))
