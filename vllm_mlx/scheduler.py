# SPDX-License-Identifier: Apache-2.0
"""
Scheduler for vllm-mlx continuous batching.

This module provides a Scheduler class that manages request scheduling
using mlx-lm's BatchGenerator for efficient continuous batching.

The scheduler follows vLLM's design with:
- Waiting queue for pending requests
- Running set for active requests
- Continuous batching via BatchGenerator
"""

from functools import lru_cache
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

from vllm_mlx.turn_prefix_cache import Segment

from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .kv_cache import (
    RequestCacheState,
    _BATCH_KV_TYPES,
)
from .utils.mamba_cache import ensure_mamba_support
from .mllm_batch_generator import _eval_prompt_cache
from .patches.mlx_lm_quantized_sdpa import patch_quantized_sdpa
from .patches.mlx_lm_prefill_flash_sdpa import apply as _apply_prefill_flash_sdpa
from .patches.gemma4_llm import patch_gemma4_attention_for_batching as _patch_gemma4_llm

patch_quantized_sdpa()
_apply_prefill_flash_sdpa()
_patch_gemma4_llm()


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Enable MambaCache batching support for models like Nemotron
ensure_mamba_support()

# Error patterns that indicate cache corruption
CACHE_CORRUPTION_PATTERNS = [
    "'NoneType' object is not subscriptable",
    "cache",
    "BatchKVCache",
]


class SchedulingPolicy(Enum):
    """Scheduling policy for request ordering."""

    FCFS = "fcfs"  # First-Come-First-Served
    PRIORITY = "priority"  # Priority-based


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    # Maximum number of concurrent requests in the batch
    max_num_seqs: int = 256
    # Maximum tokens to process per step (for prefill chunking)
    max_num_batched_tokens: int = 8192
    # Scheduling policy
    policy: SchedulingPolicy = SchedulingPolicy.FCFS
    # BatchGenerator settings
    prefill_batch_size: int = 8
    completion_batch_size: int = 32
    prefill_step_size: int = 2048
    # Optional override for MLLM prefill guard (None = use MLLM default).
    mllm_prefill_step_size: Optional[int] = None

    # Prefix cache settings
    enable_prefix_cache: bool = True
    prefix_cache_size: int = 100  # Max cached entries (legacy, ignored if memory-aware)

    # Memory-aware cache settings (recommended for large models)
    use_memory_aware_cache: bool = True  # Use memory-based eviction
    cache_memory_mb: Optional[int] = None  # None = auto-detect (20% of available RAM)
    cache_memory_percent: float = 0.20  # Fraction of available RAM if auto-detecting

    # KV cache quantization (reduces prefix cache memory).
    # Per-layer-type bits: sliding-window (RotatingKVCache) vs full-attention (KVCache).
    # See KVQuantPolicy and _build_kv_quant_policy() below.
    kv_cache_quantization: bool = False
    kv_cache_bits_sliding: int | None = None
    kv_cache_bits_full: int | None = 8
    # Provenance — True iff the value came from a user CLI override.
    # Used by _build_kv_quant_policy to emit the right advisory warnings.
    kv_cache_bits_sliding_override: bool = False
    kv_cache_bits_full_override: bool = False
    kv_cache_quantization_group_size: int = 64
    kv_cache_min_quantize_tokens: int = 256

    # Paged cache settings (experimental - for memory efficiency)
    use_paged_cache: bool = (
        False  # Use BlockAwarePrefixCache instead of PrefixCacheManager
    )
    paged_cache_block_size: int = 64  # Tokens per block
    max_cache_blocks: int = 1000  # Maximum number of cache blocks

    # TurnPrefixCache settings
    use_turn_cache: bool = False
    turn_cache_stride: int = 512
    turn_cache_ssd_gb: float = 50.0
    turn_cache_memory_gb: float = 20.0

    # Chunked prefill: max tokens to prefill per scheduler step (0 = disabled)
    # When enabled, large prompts are split into chunks so that active
    # generation requests are not starved during long prefills.
    chunked_prefill_tokens: int = 0

    # Mid-prefill cache saving: save intermediate KV cache every N tokens
    # during chunked prefill. If the client disconnects mid-prefill, the
    # saved cache is reused for the next request with the same prefix.
    # 0 = disabled. Only effective when chunked_prefill_tokens > 0.
    mid_prefill_save_interval: int = 8192

    # SSD cache tiering
    ssd_cache_dir: Optional[str] = None  # None = disabled
    ssd_cache_max_gb: float = 10.0

    # Maximum KV cache size per sequence (0 = unbounded; >0 enables RotatingKVCache)
    max_kv_size: int = 0

    # Debug: write every cache GET/PUT key (as JSON lines) to this file.
    # Each line: {"ts": float, "op": "get"|"put", "request_id": str,
    #             "n_tokens": int, "tokens": [int, ...]}.
    # None = disabled.
    cache_key_log_path: Optional[str] = None

    # MTP (Multi-Token Prediction) settings
    # Uses the model's built-in MTP head to predict multiple tokens per step
    enable_mtp: bool = False
    mtp_num_draft_tokens: int = 1  # Number of draft tokens from MTP head
    mtp_optimistic: bool = False  # Skip acceptance check for max speed

    def __post_init__(self) -> None:
        if self.mllm_prefill_step_size is not None and self.mllm_prefill_step_size <= 0:
            raise ValueError("mllm_prefill_step_size must be > 0 when provided")


@dataclass
class SchedulerOutput:
    """
    Output from a scheduling step.

    Contains information about what was scheduled and results.
    """

    # Requests scheduled in this step
    scheduled_request_ids: List[str] = field(default_factory=list)
    # Total tokens scheduled
    num_scheduled_tokens: int = 0
    # Requests that finished in this step
    finished_request_ids: Set[str] = field(default_factory=set)
    # Request outputs (tokens generated)
    outputs: List[RequestOutput] = field(default_factory=list)
    # Whether any work was done
    has_work: bool = False


class _InstrumentedBatchGenerator(BatchGenerator):
    """BatchGenerator subclass that fires a mid-prefill callback after each chunk.

    After every _next() call, for each sequence still in the prompt batch
    (end_of_prompt=False), calls mid_prefill_callback(uid, processed, cache)
    where cache is the per-sequence KV state extracted from the shared batch
    cache.  Throttled by save_interval: only fires when at least save_interval
    new tokens have been processed since the last save for that uid.
    """

    def __init__(
        self,
        *args,
        mid_prefill_callback=None,
        save_interval=0,
        kv_cache_bits=None,
        kv_cache_group_size=64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._mid_prefill_callback = mid_prefill_callback
        self._save_interval = save_interval
        self._uid_last_saved: dict = {}
        self._kv_cache_bits = kv_cache_bits
        self._kv_cache_group_size = kv_cache_group_size

    def _make_new_cache(self):
        from mlx_lm.models.cache import KVCache, QuantizedKVCache

        caches = super()._make_new_cache()
        if self._kv_cache_bits is None:
            return caches
        return [
            (
                QuantizedKVCache(
                    group_size=self._kv_cache_group_size, bits=self._kv_cache_bits
                )
                if isinstance(c, KVCache)
                else c
            )
            for c in caches
        ]

    def _next(self):
        _probe_pre_active = mx.get_active_memory()
        mx.reset_peak_memory()
        _probe_n_prompt = len(getattr(self._prompt_batch, "uids", []) or [])

        prompt_responses, gen_responses = super()._next()

        _probe_post_active = mx.get_active_memory()
        _probe_peak = mx.get_peak_memory()
        if _probe_n_prompt > 0 or _probe_post_active != _probe_pre_active:
            logger.warning(
                "[memprobe:step] phase=%s n_prompt=%d pre=%.2fGB post=%.2fGB "
                "delta=%.2fGB step_peak=%.2fGB",
                "prefill" if _probe_n_prompt > 0 else "decode",
                _probe_n_prompt,
                _probe_pre_active / 1e9,
                _probe_post_active / 1e9,
                (_probe_post_active - _probe_pre_active) / 1e9,
                _probe_peak / 1e9,
            )

        if self._mid_prefill_callback and prompt_responses:
            uid_to_idx = {uid: i for i, uid in enumerate(self._prompt_batch.uids)}
            for resp in prompt_responses:
                if resp.end_of_prompt or resp.uid not in uid_to_idx:
                    continue
                processed = resp.progress[0]
                last = self._uid_last_saved.get(resp.uid, 0)
                if self._save_interval > 0 and (processed - last) < self._save_interval:
                    continue
                idx = uid_to_idx[resp.uid]

                _cb_pre = mx.get_active_memory()
                per_uid_cache = self._prompt_batch.extract_cache(idx)
                mx.eval(*[c.keys for c in per_uid_cache if hasattr(c, "keys") and c.keys is not None],
                        *[c.values for c in per_uid_cache if hasattr(c, "values") and c.values is not None])
                _cb_post_extract = mx.get_active_memory()

                self._mid_prefill_callback(resp.uid, processed, per_uid_cache)
                _cb_post_callback = mx.get_active_memory()

                del per_uid_cache
                import gc as _gc
                _gc.collect()
                mx.clear_cache()
                _cb_post_gc = mx.get_active_memory()

                logger.warning(
                    "[memprobe:callback] processed=%d pre=%.2fGB "
                    "post_extract=%.2fGB(+%.2f) post_callback=%.2fGB(+%.2f) "
                    "post_gc=%.2fGB(net+%.2f)",
                    processed,
                    _cb_pre / 1e9,
                    _cb_post_extract / 1e9,
                    (_cb_post_extract - _cb_pre) / 1e9,
                    _cb_post_callback / 1e9,
                    (_cb_post_callback - _cb_post_extract) / 1e9,
                    _cb_post_gc / 1e9,
                    (_cb_post_gc - _cb_pre) / 1e9,
                )

                self._uid_last_saved[resp.uid] = processed

        # Remove tracking for sequences that have left the prompt batch
        active = set(self._prompt_batch.uids)
        for uid in list(self._uid_last_saved):
            if uid not in active:
                del self._uid_last_saved[uid]

        return prompt_responses, gen_responses


def _install_mtp(
    batch_gen: "BatchGenerator",
    model: Any,
    num_draft_tokens: int = 1,
    optimistic: bool = False,
) -> None:
    """
    Monkey-patch a BatchGenerator to use MTP (Multi-Token Prediction)
    with always-advance strategy for hybrid MambaCache + KVCache.

    Flow per generation step:
    1. Use skip_state logits/hidden OR run model forward -> sample primary
    2. MTP head drafts one token after primary
    3. Verify [primary, draft] in one model call (always advances cache)
    4. Accept: skip_state from pos 1, defer draft for next step emission
       Reject: trim KVCache by 1, skip_state from pos 0 (no cold start)
    5. Draft is emitted in the NEXT generation step after primary
    """
    # Placeholder; assigned after _generation_batch is available (bottom of function)
    _orig_gen_step = None

    # Greedy sampler for MTP draft tokens
    _draft_sampler = make_sampler(temp=0.0)

    # Skip state: when MTP accepts, the cache already consumed [primary, draft].
    # Next _step call receives primary as input but must NOT re-feed it.
    # Instead, use stored logits from the verify pass.
    # Format: {'logits': (B, V), 'hidden': (B, 1, H)}
    _skip_state = [None]

    # Deferred drafts: draft tokens to emit in the NEXT generation step,
    # keyed by UID for stability across batch changes.
    # Format: {uid: {'token': int, 'logprobs': mx.array}}
    _deferred_drafts = {}

    # MTP stats
    _mtp_stats = {"accepted": 0, "rejected": 0, "errors": 0}

    def _mtp_step(self):
        """Replacement for GenerationBatch._step with MTP always-advance strategy."""
        self._current_tokens = self._next_tokens
        inputs = self._current_tokens
        batch_size = inputs.shape[0]

        if inputs.shape[0] == 0:
            return _orig_gen_step(self)

        skip = _skip_state[0]
        if skip is not None and skip["logits"].shape[0] != batch_size:
            skip = None
            _skip_state[0] = None

        if skip is not None:
            logits = skip["logits"]
            hidden_states = skip["hidden"]
            _skip_state[0] = None
        else:
            model_output = self.model(
                inputs[:, None], cache=self.prompt_cache, return_hidden=True
            )
            if not isinstance(model_output, tuple):
                return _orig_gen_step(self)
            logits, hidden_states = model_output
            logits = logits[:, -1, :]

        if any(self.logits_processors):
            processed = []
            for e in range(batch_size):
                sl = logits[e : e + 1]
                for proc in self.logits_processors[e]:
                    token_ctx = getattr(self, "_token_context", None)
                    token_ctx_e = token_ctx[e] if token_ctx is not None else None
                    sl = proc(token_ctx_e, sl) if token_ctx_e is not None else sl
                processed.append(sl)
            logits = mx.concatenate(processed, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if any(self.samplers):
            samples = [
                (self.samplers[e] or self.fallback_sampler)(logprobs[e : e + 1])
                for e in range(batch_size)
            ]
            primary_tokens = mx.concatenate(samples, axis=0)
        else:
            primary_tokens = self.fallback_sampler(logprobs)

        current_uids = list(self.uids)

        try:
            draft_logits = self.model.mtp_forward(
                hidden_states[:, -1:, :],
                primary_tokens[:, None],
                mtp_cache=None,
            )
            draft_logits = draft_logits[:, -1, :]
            draft_logprobs = draft_logits - mx.logsumexp(
                draft_logits, axis=-1, keepdims=True
            )
            draft_tokens = _draft_sampler(draft_logprobs)

            _rnn_snapshots = {}
            if not optimistic:
                for _ci, _c in enumerate(self.prompt_cache):
                    if not (hasattr(_c, "is_trimmable") and _c.is_trimmable()):
                        if hasattr(_c, "state"):
                            _rnn_snapshots[_ci] = [
                                s.copy() if s is not None else None for s in _c.state
                            ]

            verify_input = mx.concatenate(
                [primary_tokens[:, None], draft_tokens[:, None]], axis=1
            )
            verify_output = self.model(
                verify_input, cache=self.prompt_cache, return_hidden=True
            )
            if isinstance(verify_output, tuple):
                verify_logits, verify_hidden = verify_output
            else:
                verify_logits, verify_hidden = verify_output, None

            if optimistic:
                if verify_hidden is not None:
                    _skip_state[0] = {
                        "logits": verify_logits[:, 1, :],
                        "hidden": verify_hidden[:, -1:, :],
                    }
                    verify_lp = verify_logits[:, 0, :] - mx.logsumexp(
                        verify_logits[:, 0, :], axis=-1, keepdims=True
                    )
                    mx.async_eval(
                        _skip_state[0]["logits"],
                        _skip_state[0]["hidden"],
                        draft_tokens,
                        verify_lp,
                    )
                    for e in range(batch_size):
                        uid = current_uids[e]
                        _deferred_drafts[uid] = {
                            "token_array": draft_tokens[e : e + 1],
                            "logprobs": verify_lp[e],
                        }
                else:
                    _skip_state[0] = None
                _mtp_stats["accepted"] += 1
            else:
                verify_pred = mx.argmax(verify_logits[:, 0, :], axis=-1)
                mx.eval(verify_pred, draft_tokens)
                pred_list = verify_pred.tolist()
                draft_list = draft_tokens.tolist()
                all_accepted = pred_list == draft_list

                if all_accepted and verify_hidden is not None:
                    _skip_state[0] = {
                        "logits": verify_logits[:, 1, :],
                        "hidden": verify_hidden[:, -1:, :],
                    }
                    mx.async_eval(_skip_state[0]["logits"], _skip_state[0]["hidden"])
                    verify_lp = verify_logits[:, 0, :] - mx.logsumexp(
                        verify_logits[:, 0, :], axis=-1, keepdims=True
                    )
                    for e in range(batch_size):
                        _deferred_drafts[current_uids[e]] = {
                            "token": draft_list[e],
                            "logprobs": verify_lp[e],
                        }
                    _mtp_stats["accepted"] += 1
                else:
                    if _rnn_snapshots:
                        for c in self.prompt_cache:
                            if (
                                hasattr(c, "is_trimmable")
                                and c.is_trimmable()
                                and hasattr(c, "trim")
                            ):
                                c.trim(2)
                        for _ci, _snap in _rnn_snapshots.items():
                            self.prompt_cache[_ci].state = _snap
                        rerun = self.model(
                            primary_tokens[:, None],
                            cache=self.prompt_cache,
                            return_hidden=True,
                        )
                        if isinstance(rerun, tuple):
                            _, rerun_hidden = rerun
                            _skip_state[0] = {
                                "logits": verify_logits[:, 0, :],
                                "hidden": rerun_hidden[:, -1:, :],
                            }
                            mx.async_eval(
                                _skip_state[0]["logits"], _skip_state[0]["hidden"]
                            )
                        else:
                            _skip_state[0] = None
                    else:
                        for c in self.prompt_cache:
                            if (
                                hasattr(c, "is_trimmable")
                                and c.is_trimmable()
                                and hasattr(c, "trim")
                            ):
                                c.trim(1)
                        if verify_hidden is not None:
                            _skip_state[0] = {
                                "logits": verify_logits[:, 0, :],
                                "hidden": verify_hidden[:, 0:1, :],
                            }
                            mx.async_eval(
                                _skip_state[0]["logits"], _skip_state[0]["hidden"]
                            )
                        else:
                            _skip_state[0] = None
                    for uid in current_uids:
                        _deferred_drafts.pop(uid, None)
                    _mtp_stats["rejected"] += 1

        except Exception as e:
            logger.debug(f"[MTP] draft/verify failed: {e}")
            _skip_state[0] = None
            _mtp_stats["errors"] += 1

        self._next_tokens = primary_tokens
        self._next_logprobs = list(logprobs)
        mx.async_eval(self._next_tokens, self._next_logprobs)

        mx.eval(inputs, getattr(self, "_current_logprobs", None) or [])
        inputs_list = inputs.tolist()
        for sti, ti in zip(self.tokens, inputs_list):
            sti.append(ti)
        return inputs_list, getattr(self, "_current_logprobs", None) or list(logprobs)

    def _mtp_next(self=batch_gen):
        """Wrapper around _next that emits deferred MTP draft tokens."""
        if not self._generation_batch.uids:
            _skip_state[0] = None
            _deferred_drafts.clear()

        prev_deferred = {}
        if self._generation_batch.uids:
            for uid in self._generation_batch.uids:
                if uid in _deferred_drafts:
                    prev_deferred[uid] = _deferred_drafts.pop(uid)

        prompt_responses, gen_responses = self._inner_next()

        if not prev_deferred or not gen_responses:
            return prompt_responses, gen_responses

        augmented = []
        draft_end_uids = set()
        for r in gen_responses:
            uid = r.uid
            augmented.append(r)

            if r.finish_reason is not None:
                _deferred_drafts.pop(uid, None)
                prev_deferred.pop(uid, None)
                continue

            if uid in prev_deferred:
                draft_info = prev_deferred.pop(uid)
                draft_t = (
                    draft_info["token"]
                    if "token" in draft_info
                    else draft_info["token_array"].item()
                )
                draft_lp = draft_info["logprobs"]

                draft_finish = None
                gb = self._generation_batch
                if gb is not None and uid in gb.uids:
                    e = gb.uids.index(uid)
                    gb._num_tokens[e] = gb._num_tokens[e] + 1
                    if gb._num_tokens[e] >= gb.max_tokens[e]:
                        draft_finish = "length"
                        draft_end_uids.add(uid)

                from dataclasses import replace as _dc_replace

                draft_r = _dc_replace(
                    r,
                    token=draft_t,
                    logprobs=draft_lp,
                    finish_reason=draft_finish,
                    prompt_cache=None,
                )
                augmented.append(draft_r)

        if draft_end_uids and self._generation_batch.uids:
            keep = [
                e
                for e, u in enumerate(self._generation_batch.uids)
                if u not in draft_end_uids
            ]
            self._generation_batch.filter(keep)

        return prompt_responses, augmented

    _orig_gen_step = batch_gen._generation_batch._step
    batch_gen._generation_batch._step = _mtp_step
    batch_gen._inner_next = batch_gen._next
    batch_gen._next = _mtp_next

    if num_draft_tokens != 1:
        logger.warning(
            "[MTP] num_draft_tokens=%d requested, but the current batched MTP "
            "path drafts exactly one token per verify step",
            num_draft_tokens,
        )
    mode_str = "optimistic (no verify)" if optimistic else "always-advance"
    logger.info(
        f"[MTP] installed with num_draft_tokens={num_draft_tokens}, "
        f"effective_draft_tokens=1, {mode_str} mode"
    )


def _build_kv_quant_policy(config: "SchedulerConfig") -> "KVQuantPolicy | None":
    """Build the KVQuantPolicy for this config. Pure — no logging side effects.

    Call `_warn_about_kv_quant_policy(config)` once at the args→config boundary
    to emit advisory warnings; this function may be called any number of times
    downstream without re-emitting them.

    Returns None when kv_cache_quantization is disabled (overrides are ignored).
    Otherwise returns a KVQuantPolicy with provenance flags threaded through.
    """
    from .cache_types import KVQuantPolicy

    if not config.kv_cache_quantization:
        return None

    sliding_bits = (
        config.kv_cache_bits_sliding
        if config.kv_cache_bits_sliding_override
        else None  # smart default: bf16
    )
    full_bits = (
        config.kv_cache_bits_full
        if config.kv_cache_bits_full_override
        else 8  # smart default: q8
    )

    return KVQuantPolicy(
        sliding_bits=sliding_bits,
        full_bits=full_bits,
        sliding_override=config.kv_cache_bits_sliding_override,
        full_override=config.kv_cache_bits_full_override,
    )


def _warn_about_kv_quant_policy(config: "SchedulerConfig") -> None:
    """Emit advisory warnings about a KV quant configuration.

    Call this once at the args→SchedulerConfig boundary (CLI). Downstream code
    paths should use the pure `_build_kv_quant_policy` so warnings don't repeat.
    """
    sliding_set = config.kv_cache_bits_sliding_override
    full_set = config.kv_cache_bits_full_override

    if not config.kv_cache_quantization:
        if sliding_set or full_set:
            which = " and ".join(
                name
                for name, present in (
                    ("--kv-cache-bits-sliding", sliding_set),
                    ("--kv-cache-bits-full", full_set),
                )
                if present
            )
            logger.warning(
                "`%s` is ignored because `--kv-cache-quantization` is disabled.",
                which,
            )
        return

    sliding_bits = config.kv_cache_bits_sliding if sliding_set else None
    full_bits = config.kv_cache_bits_full if full_set else 8

    if full_set and full_bits is None:
        logger.warning(
            "Full-attention layers are the dominant memory consumer; "
            "setting them to bf16 negates the memory benefit of "
            "`--kv-cache-quantization`. Did you mean to leave the default (q8)?"
        )
    if sliding_set and isinstance(sliding_bits, int):
        logger.warning(
            "Sliding-window layers use full RoPE and are sensitive to "
            "quantization error. Recommended default is bf16; quantizing them "
            "may degrade quality. Measure before relying on this setting."
        )
    if isinstance(full_bits, int) and full_bits <= 4:
        logger.info(
            "`--kv-cache-bits-full %d` is supported by the Qwen3.5 partial-RoPE "
            "analogy but unverified for Gemma 4. Measure quality before relying "
            "on this setting.",
            full_bits,
        )


@dataclass
class _PrefixCacheBundle:
    """All prefix-cache objects produced by _build_prefix_cache."""

    adapter: "CacheManager | None" = None
    turn_cache: "TurnPrefixCache | None" = None


def _build_prefix_cache(config: "SchedulerConfig", model: Any) -> _PrefixCacheBundle:
    """Construct the appropriate prefix-cache adapter from SchedulerConfig.

    Only TurnPrefixCache is supported. All other backends have been removed.
    """
    from .prefix_cache_adapters import TurnCacheManager

    bundle = _PrefixCacheBundle()

    if config.use_turn_cache:
        from .turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig

        turn_cache = TurnPrefixCache(
            TurnPrefixCacheConfig(
                checkpoint_stride=config.turn_cache_stride,
                max_memory_gb=config.turn_cache_memory_gb,
                ssd_max_gb=config.turn_cache_ssd_gb,
            )
        )
        bundle.turn_cache = turn_cache
        policy = _build_kv_quant_policy(config)
        bundle.adapter = TurnCacheManager(
            turn_cache,
            policy=policy,
            kv_group_size=config.kv_cache_quantization_group_size,
        )
        logger.info(
            f"TurnPrefixCache enabled: stride={config.turn_cache_stride} "
            f"memory={config.turn_cache_memory_gb}GB"
        )
    else:
        logger.info("Prefix cache disabled (use_turn_cache=False)")

    return bundle


class Scheduler:
    """
    Scheduler for continuous batching using mlx-lm BatchGenerator.

    This scheduler manages the lifecycle of requests:
    1. Requests arrive and are added to the waiting queue
    2. Scheduler moves requests from waiting to running (via BatchGenerator)
    3. BatchGenerator processes all running requests together
    4. Finished requests are removed and outputs returned

    The key insight is that mlx-lm's BatchGenerator already implements
    continuous batching at the token level, so we use it as the backend.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[SchedulerConfig] = None,
    ):
        """
        Initialize the scheduler.

        Args:
            model: The MLX model
            tokenizer: The tokenizer
            config: Scheduler configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SchedulerConfig()

        # Validate TurnPrefixCache configuration
        if self.config.use_turn_cache and self.config.chunked_prefill_tokens == 0:
            raise ValueError(
                "TurnPrefixCache requires --chunked-prefill-tokens to be set. "
                "Set --chunked-prefill-tokens 8192 or higher."
            )

        # Detect if tokenizer is a processor (MLLM) and get the actual tokenizer
        self._actual_tokenizer = self._get_actual_tokenizer(tokenizer)

        # Per-request streaming detokenizers for UTF-8-safe incremental decode
        self._detokenizer_pool: Dict[str, Any] = {}

        # Request management - following vLLM's design
        self.waiting: deque[Request] = deque()  # Waiting queue (FCFS)
        self.running: Dict[str, Request] = {}  # Running requests by ID
        self.requests: Dict[str, Request] = {}  # All requests by ID
        self.finished_req_ids: Set[str] = set()  # Recently finished

        # Mapping between our request IDs and BatchGenerator UIDs
        self.request_id_to_uid: Dict[str, int] = {}
        self.uid_to_request_id: Dict[int, str] = {}

        # BatchGenerator - the actual batching engine
        self.batch_generator: Optional[BatchGenerator] = None
        self._current_sampler_params: Optional[Tuple] = None

        # Optional path for cache-key debug logging.
        self._cache_key_log_path: Optional[str] = self.config.cache_key_log_path

        # Prefix cache for KV state reuse — attributes set by _init_cache_bundle()
        self._prefix_cache = None
        self.turn_cache: Optional[TurnPrefixCache] = None
        self._init_cache_bundle()

        # Thread-safe set for deferred aborts (main thread → executor thread)
        # CPython GIL guarantees set.add() and `x in set` are atomic.
        self._pending_abort_ids: Set[str] = set()

        # Statistics
        self.num_requests_processed = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Memory management: periodic mx.clear_cache() to free Metal command buffers
        # Lower interval = less VRAM spike during generation but slight throughput cost
        self._step_count = 0
        self._clear_cache_interval = 32
        self._memory_log_interval = 256

    def _init_cache_bundle(self) -> None:
        """Initialize cache backend attributes from SchedulerConfig.

        Called from __init__. The Scheduler only knows about CacheManager
        through self._prefix_cache.
        """
        if self.config.enable_prefix_cache:
            _bundle = _build_prefix_cache(self.config, self.model)
            self._prefix_cache = _bundle.adapter
            self.turn_cache = _bundle.turn_cache

    def _get_actual_tokenizer(self, tokenizer: Any) -> Any:
        """
        Get the actual tokenizer from a processor or tokenizer.

        MLLM models use processors (e.g., Qwen3VLProcessor) which wrap
        the tokenizer. This method extracts the actual tokenizer.
        """
        # If it has encode method, it's already a tokenizer
        if hasattr(tokenizer, "encode") and callable(tokenizer.encode):
            return tokenizer
        # If it's a processor, get the wrapped tokenizer
        if hasattr(tokenizer, "tokenizer"):
            return tokenizer.tokenizer
        # Fallback to the original
        return tokenizer

    def _decode_tokens(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text, handling both tokenizers and processors.
        """
        return self._actual_tokenizer.decode(token_ids, skip_special_tokens=False)

    def _log_cache_key(self, op: str, request_id: str, tokens: List[int]) -> None:
        """Append a cache key event to _cache_key_log_path (if set)."""
        path = getattr(self, "_cache_key_log_path", None)
        if not path:
            return
        import json, time as _t

        record = {
            "ts": _t.time(),
            "op": op,
            "request_id": request_id,
            "n_tokens": len(tokens),
            "tokens": tokens,
        }
        try:
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.debug(f"[cache_key_log] write failed: {e}")

    def _get_detokenizer(self, request_id: str) -> Any:
        """Get or create a streaming detokenizer for a request."""
        if request_id not in self._detokenizer_pool:
            detok = NaiveStreamingDetokenizer(self._actual_tokenizer)
            self._detokenizer_pool[request_id] = detok
        return self._detokenizer_pool[request_id]

    def _cleanup_detokenizer(self, request_id: str) -> None:
        """Remove the streaming detokenizer for a finished request."""
        self._detokenizer_pool.pop(request_id, None)

    def _get_stop_tokens(self) -> Set[int]:
        """Get stop token IDs from tokenizer or processor."""
        stop_tokens = set()
        # Check both the processor/tokenizer and the actual tokenizer
        for tok in [self.tokenizer, self._actual_tokenizer]:
            if tok is None:
                continue
            if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
                if isinstance(tok.eos_token_id, list):
                    stop_tokens.update(tok.eos_token_id)
                else:
                    stop_tokens.add(tok.eos_token_id)
            if hasattr(tok, "eos_token_ids") and tok.eos_token_ids is not None:
                if isinstance(tok.eos_token_ids, (list, set, tuple)):
                    stop_tokens.update(tok.eos_token_ids)
                else:
                    # Handle case where eos_token_ids is a single int
                    stop_tokens.add(tok.eos_token_ids)
        return stop_tokens

    def _create_batch_generator(
        self, sampling_params: SamplingParams
    ) -> BatchGenerator:
        """Create a BatchGenerator with the given sampling parameters."""
        sampler = make_sampler(
            temp=sampling_params.temperature,
            top_p=sampling_params.top_p,
            min_p=sampling_params.min_p,
        )

        stop_tokens = self._get_stop_tokens()
        # Add custom stop token IDs
        if sampling_params.stop_token_ids:
            stop_tokens.update(sampling_params.stop_token_ids)

        save_interval = self.config.mid_prefill_save_interval
        mid_prefill_cb = None
        if self._prefix_cache is not None and (
            save_interval > 0 or self.turn_cache is not None
        ):
            mid_prefill_cb = self._make_mid_prefill_save_callback(save_interval)
            logger.info(f"[mid_prefill_cache] enabled, interval={save_interval}")

        # BatchGenerator needs the *live* decode-cache bit width — full-attention bits
        # is the right choice because that's what shows up in update_and_fetch when
        # mlx-lm allocates a fresh QuantizedKVCache. Sliding layers go through
        # RotatingKVCache which is always float; no separate kv_bits needed there.
        kv_bits = (
            self.config.kv_cache_bits_full
            if self.config.use_turn_cache and self.config.kv_cache_quantization
            else None
        )
        bg = _InstrumentedBatchGenerator(
            model=self.model,
            max_tokens=sampling_params.max_tokens,
            stop_tokens=stop_tokens,
            sampler=sampler,
            prefill_batch_size=self.config.prefill_batch_size,
            completion_batch_size=self.config.completion_batch_size,
            prefill_step_size=self.config.prefill_step_size,
            mid_prefill_callback=mid_prefill_cb,
            save_interval=save_interval,
            kv_cache_bits=kv_bits,
            kv_cache_group_size=self.config.kv_cache_quantization_group_size,
        )
        # mlx-lm >=0.31.x BatchGenerator natively interleaves prefill and
        # decode — chunked_prefill_tokens now only controls mid-prefill save
        # frequency (wired via _InstrumentedBatchGenerator in Task 2).
        logger.info(
            f"[batch_generator] prefill_step_size={self.config.prefill_step_size} "
            f"(native interleaving)"
        )

        # Install MTP if the model supports it
        if self.config.enable_mtp:
            if hasattr(self.model, "mtp") and self.model.mtp is not None:
                _install_mtp(
                    bg,
                    model=self.model,
                    num_draft_tokens=self.config.mtp_num_draft_tokens,
                    optimistic=self.config.mtp_optimistic,
                )
            else:
                logger.warning(
                    "[MTP] --enable-mtp is set but model has no MTP head "
                    "(model.mtp is None). MTP will be disabled."
                )

        return bg

    def _make_mid_prefill_save_callback(self, save_interval: int):
        """Create a callback for saving intermediate KV cache during chunked prefill.

        Dispatches to self._prefix_cache.on_prefill_checkpoint() which handles
        both memory-cache throttling/storage and turn-cache boundary capture.
        """

        def _mid_prefill_save(uid, processed_tokens, prompt_cache):
            request_id = self.uid_to_request_id.get(uid)
            if not request_id:
                return
            request = self.requests.get(request_id)
            if not request:
                return
            if self._prefix_cache is None:
                return
            extracted = self._prefix_cache.extract_cache(prompt_cache)
            if extracted:
                total = (request._cache_state.cached_tokens or 0) + processed_tokens
                self._prefix_cache.on_prefill_checkpoint(request, total, extracted)

        return _mid_prefill_save

    def _close_batch_generator(self) -> None:
        """Properly close BatchGenerator to restore wired_limit."""
        if self.batch_generator is not None:
            try:
                if hasattr(self.batch_generator, "close"):
                    self.batch_generator.close()
            except Exception as e:
                logger.debug(f"Error closing BatchGenerator: {e}")
            self.batch_generator = None

    def _ensure_batch_generator(self, sampling_params: SamplingParams) -> None:
        """Ensure BatchGenerator exists with compatible settings."""
        sampler_params = (
            sampling_params.temperature,
            sampling_params.top_p,
            sampling_params.min_p,
        )

        # Create new generator if needed or if sampling params changed
        if (
            self.batch_generator is None
            or self._current_sampler_params != sampler_params
        ):
            # If we have an existing generator with requests, we need to drain it first
            if self.batch_generator is not None and self.running:
                logger.warning(
                    "Sampling parameters changed with active requests. "
                    "New requests will use new parameters after current batch completes."
                )
                return

            # Keep prefix cache across BatchGenerator recreations.
            # KV cache entries depend only on the input tokens, not on
            # sampling params (temperature, top_p, min_p).  Since the
            # server runs a single model, the cache is always valid.
            if self.batch_generator is not None:
                logger.info(
                    "[batch_generator] recreating (sampler params changed), "
                    "keeping cache entries"
                )

            self._close_batch_generator()
            self.batch_generator = self._create_batch_generator(sampling_params)
            self._current_sampler_params = sampler_params

    def add_request(self, request: Request) -> None:
        """
        Add a new request to the scheduler.

        Args:
            request: The request to add
        """
        if request.request_id in self.requests:
            raise ValueError(f"Request {request.request_id} already exists")

        # Tokenize if needed
        if request.prompt_token_ids is None:
            if isinstance(request.prompt, str):
                # Handle both tokenizers and processors (for MLLM models)
                if hasattr(self.tokenizer, "encode"):
                    request.prompt_token_ids = self.tokenizer.encode(request.prompt)
                elif hasattr(self.tokenizer, "tokenizer") and hasattr(
                    self.tokenizer.tokenizer, "encode"
                ):
                    # Processor wraps tokenizer (e.g., Qwen3VLProcessor)
                    request.prompt_token_ids = self.tokenizer.tokenizer.encode(
                        request.prompt
                    )
                else:
                    raise AttributeError(
                        f"Tokenizer {type(self.tokenizer)} has no 'encode' method. "
                        "Continuous batching requires a tokenizer with encode support."
                    )
            else:
                request.prompt_token_ids = list(request.prompt)
            request.num_prompt_tokens = len(request.prompt_token_ids)

        # Cache fetch is deferred to _schedule_waiting() on the worker thread
        # to avoid MLX stream/thread mismatches (lazy ops must stay on the
        # thread that will evaluate them).

        # Initialise consolidated cache state
        request._cache_state = RequestCacheState()

        # Add to tracking
        self.requests[request.request_id] = request
        self.waiting.append(request)

        logger.debug(
            f"Added request {request.request_id} with {request.num_prompt_tokens} prompt tokens"
        )

    def abort_request(self, request_id: str) -> bool:
        """
        Queue request for abort. Thread-safe, called from any thread.

        The actual abort is deferred to the executor thread (inside step())
        to avoid race conditions with in-flight Metal GPU operations.

        Args:
            request_id: The request ID to abort

        Returns:
            True (abort is always enqueued)
        """
        self._pending_abort_ids.add(request_id)
        logger.info(f"[abort_request] {request_id[:12]} enqueued for deferred abort")
        return True

    def _process_pending_aborts(self) -> None:
        """Drain and process pending abort requests. Called from executor thread."""
        while self._pending_abort_ids:
            request_id = self._pending_abort_ids.pop()
            self._do_abort_request(request_id)

    def _do_abort_request(self, request_id: str) -> bool:
        """
        Actually abort a request. Must be called from the executor thread.

        Handles the case where the request was already removed from
        self.requests by _cleanup_request() but still lives in the
        BatchGenerator (e.g. in _partial or active_batch).

        Args:
            request_id: The request ID to abort

        Returns:
            True if any cleanup was performed, False otherwise
        """
        request = self.requests.get(request_id)
        was_waiting = False
        was_running = False
        removed_from_batch = False

        # Remove from waiting queue
        if request is not None and request.status == RequestStatus.WAITING:
            was_waiting = True
            try:
                self.waiting.remove(request)
            except ValueError:
                pass

        # Remove from running (BatchGenerator) — do this even if request
        # was already cleaned up from self.requests, because the UID may
        # still be live inside the BatchGenerator (_partial / active_batch).
        if request_id in self.request_id_to_uid:
            was_running = True
            uid = self.request_id_to_uid[request_id]
            if self.batch_generator is not None:
                self.batch_generator.remove([uid])
                removed_from_batch = True
            del self.uid_to_request_id[uid]
            del self.request_id_to_uid[request_id]

        if request_id in self.running:
            del self.running[request_id]

        # Credit in-flight tokens so dashboard metrics stay accurate
        # (without this, aborted requests' tokens vanish from /v1/status).
        if request is not None and request.num_output_tokens > 0:
            self.total_completion_tokens += request.num_output_tokens
            self.total_prompt_tokens += request.num_prompt_tokens

        if request is not None:
            request.set_finished(RequestStatus.FINISHED_ABORTED)
            # Release cache references so Metal buffers can be freed.
            # release() is a no-op if no leaf is pinned and also clears
            # the manager-owned turn_path.
            request._cache_state.cache = None
            request._cache_state.decoded_cache = None
            if self._prefix_cache is not None:
                self._prefix_cache.release(request)
        self.finished_req_ids.add(request_id)
        self._cleanup_detokenizer(request_id)

        # Flush Metal encoders after removing arrays from batch
        mx.clear_cache()

        logger.info(
            f"[abort_request] {request_id[:12]} ABORTED "
            f"was_waiting={was_waiting} was_running={was_running} "
            f"removed_from_batch={removed_from_batch} "
            f"remaining_running={len(self.running)} remaining_waiting={len(self.waiting)}"
        )
        return True

    def has_requests(self) -> bool:
        """Check if there are any pending or running requests."""
        return bool(self.waiting or self.running)

    def get_num_waiting(self) -> int:
        """Get number of waiting requests."""
        return len(self.waiting)

    def get_num_running(self) -> int:
        """Get number of running requests."""
        return len(self.running)

    def _schedule_waiting(self) -> List[Request]:
        """
        Move requests from waiting queue to running.

        Returns:
            List of requests that were scheduled
        """

        scheduled = []

        while self.waiting and len(self.running) < self.config.max_num_seqs:
            request = self.waiting.popleft()

            # Fetch cache on the worker thread so all MLX ops (dequantize,
            # reconstruct) are enqueued on the correct stream.
            if request._cache_state.remaining_tokens is None:
                hit_occurred = False
                if self._prefix_cache is not None:
                    hit_occurred = self._prefix_cache.fetch(request)
                    if hit_occurred:
                        self._log_cache_key(
                            "get", request.request_id, list(request.prompt_token_ids)
                        )
                        logger.info(
                            f"[cache_fetch] request={request.request_id[:12]} HIT "
                            f"cached={request._cache_state.cached_tokens} "
                            f"remaining={len(request._cache_state.remaining_tokens)}"
                        )
                    else:
                        self._log_cache_key(
                            "get", request.request_id, list(request.prompt_token_ids)
                        )
                        logger.info(
                            f"[cache_fetch] request={request.request_id[:12]} MISS "
                            f"prompt_tokens={len(request.prompt_token_ids)}"
                        )
                else:
                    # No prefix cache configured: RequestCacheState defaults
                    # already encode a miss (hit_type="miss", cached_tokens=0,
                    # turn_path=[], prefill_boundaries=[]). We only set
                    # remaining_tokens so downstream code that reads it works.
                    request._cache_state.remaining_tokens = request.prompt_token_ids

            # Ensure we have a batch generator
            self._ensure_batch_generator(request.sampling_params)

            if self.batch_generator is None:
                # Put back and try again later
                self.waiting.appendleft(request)
                break

            # Determine tokens to process and cache to use
            if request._cache_state.remaining_tokens:
                tokens_to_process = request._cache_state.remaining_tokens
            else:
                tokens_to_process = request.prompt_token_ids
            cache_to_use = request._cache_state.cache  # May be None

            # Create bounded cache when max_kv_size is configured and no cache exists
            if cache_to_use is None and self.config.max_kv_size > 0:
                from mlx_lm.models.cache import make_prompt_cache

                cache_to_use = make_prompt_cache(
                    self.model, max_kv_size=self.config.max_kv_size
                )

            # Validate cache before using it
            if cache_to_use is not None and self._prefix_cache is not None:
                if not self._prefix_cache.validate(cache_to_use):
                    logger.debug(
                        f"Request {request.request_id}: invalid cache detected, "
                        f"proceeding without cache"
                    )
                    cache_to_use = None
                    request._cache_state.cache = None
                    request._cache_state.cached_tokens = 0
                    request._cache_state.remaining_tokens = request.prompt_token_ids
                    tokens_to_process = request.prompt_token_ids

            # Build per-request logits_processors from repetition_penalty and
            # any caller-supplied extras (e.g. JSON schema constrained
            # decoding).
            rep_penalty = request.sampling_params.repetition_penalty
            extra_lp = request.sampling_params.logits_processors or []
            combined_lp: list = []
            if rep_penalty and rep_penalty != 1.0:
                combined_lp.extend(
                    make_logits_processors(repetition_penalty=rep_penalty)
                )
                logger.info(
                    f"[rep_penalty] request={request.request_id[:12]} "
                    f"penalty={rep_penalty}"
                )
            if extra_lp:
                combined_lp.extend(extra_lp)
                logger.info(
                    f"[logits_proc] request={request.request_id[:12]} "
                    f"extra_processors={len(extra_lp)}"
                )
            lp = combined_lp

            # Insert into BatchGenerator with optional cache.
            # Wrap in try/except: if cache shapes are incompatible
            # (e.g. stale entry after BatchGenerator recreation),
            # fall back to no-cache insert instead of crashing.
            insert_kwargs = {
                "max_tokens": [request.sampling_params.max_tokens],
                "caches": [cache_to_use] if cache_to_use else None,
                # Always pass logits_processors (even empty list) so that
                # mlx_lm BatchGenerator never stores None per-sequence.
                "logits_processors": [lp] if lp else [[]],
            }

            def _split_at_boundaries(tokens, boundaries):
                """Split token list at pre-adjusted boundary positions."""
                if not boundaries:
                    return [tokens]
                segments = []
                prev = 0
                for b in sorted(boundaries):
                    if 0 < b < len(tokens):
                        segments.append(tokens[prev:b])
                        prev = b
                segments.append(tokens[prev:])
                return [s for s in segments if s]

            prefill_bds = (
                request._cache_state.prefill_boundaries if request._cache_state else []
            )
            segments = _split_at_boundaries(tokens_to_process, prefill_bds)
            use_segments = len(segments) > 1

            try:
                if use_segments:
                    uids = self.batch_generator.insert_segments(
                        [segments],
                        **insert_kwargs,
                    )
                else:
                    uids = self.batch_generator.insert(
                        [tokens_to_process],
                        **insert_kwargs,
                    )
            except Exception as e:
                if cache_to_use is not None:
                    # Release the pinned leaf (release() is a no-op when
                    # nothing is pinned and also clears turn_path).
                    if self._prefix_cache is not None:
                        self._prefix_cache.release(request)

                    logger.warning(
                        f"[cache_insert_error] request={request.request_id[:12]} "
                        f"cache insert failed ({e}), retrying without cache"
                    )
                    # Reset scheduler-owned cache fields. turn_path is
                    # cache-manager-owned and was already cleared by release().
                    cache_to_use = None
                    request._cache_state.cache = None
                    request._cache_state.hit_type = "miss"
                    request._cache_state.cached_tokens = 0
                    request._cache_state.remaining_tokens = request.prompt_token_ids
                    # Recovery: full prefill without cache → no segments needed
                    request._cache_state.prefill_boundaries = []
                    tokens_to_process = request.prompt_token_ids
                    insert_kwargs["caches"] = None
                    uids = self.batch_generator.insert(
                        [tokens_to_process], **insert_kwargs
                    )
                else:
                    raise

            if uids:
                uid = uids[0]
                self.request_id_to_uid[request.request_id] = uid
                self.uid_to_request_id[uid] = request.request_id
                request.batch_uid = uid
                request.status = RequestStatus.RUNNING
                # Release the prompt cache reference now that BatchGenerator
                # has its own copy.  Holding this reference prevents MLX from
                # freeing the Metal buffers until the request object is GC'd,
                # which under sustained traffic can accumulate hundreds of GB
                # of wired memory (issue #442).
                request._cache_state.cache = None
                self.running[request.request_id] = request
                scheduled.append(request)

                self.total_prompt_tokens += request.num_prompt_tokens
                cache_info = (
                    f", {request._cache_state.cached_tokens} cached"
                    if request._cache_state.cached_tokens > 0
                    else ""
                )
                tokens_to_prefill = len(tokens_to_process)
                rep_info = (
                    f" rep_penalty={rep_penalty}"
                    if rep_penalty and rep_penalty != 1.0
                    else ""
                )
                logger.info(
                    f"[schedule] request={request.request_id[:12]} uid={uid} "
                    f"prompt_tokens={request.num_prompt_tokens} "
                    f"tokens_to_prefill={tokens_to_prefill}{cache_info} "
                    f"max_tokens={request.sampling_params.max_tokens}{rep_info} "
                    f"running={len(self.running)} waiting={len(self.waiting)}"
                )

        return scheduled

    def _process_batch_responses(
        self, responses: List[Any]
    ) -> Tuple[List[RequestOutput], Set[str]]:
        """
        Process responses from BatchGenerator.

        Args:
            responses: List of BatchGenerator.Response objects

        Returns:
            Tuple of (outputs, finished_request_ids)
        """
        outputs = []
        finished_ids = set()

        for response in responses:
            request_id = self.uid_to_request_id.get(response.uid)
            if request_id is None:
                continue

            request = self.running.get(request_id)
            if request is None:
                continue

            # Append token to request
            request.append_output_token(response.token)

            # Record first token time for TTFT metric
            if request.first_token_time is None and request.num_output_tokens > 0:
                import time as _time

                request.first_token_time = _time.time()

            # Decode the new token using streaming detokenizer (UTF-8 safe)
            if response.finish_reason == "stop":
                new_text = ""
            else:
                detok = self._get_detokenizer(request_id)
                detok.add_token(response.token)
                new_text = detok.last_segment

            # Create output
            output = RequestOutput(
                request_id=request_id,
                new_token_ids=[response.token],
                new_text=new_text,
                output_token_ids=list(request.output_token_ids),
                prompt_tokens=request.num_prompt_tokens,
                completion_tokens=request.num_output_tokens,
            )

            # Check if finished
            if response.finish_reason is not None:
                if response.finish_reason == "stop":
                    request.set_finished(RequestStatus.FINISHED_STOPPED)
                elif response.finish_reason == "length":
                    request.set_finished(RequestStatus.FINISHED_LENGTH_CAPPED)

                output.finished = True
                output.finish_reason = response.finish_reason
                finished_ids.add(request_id)

                # Finalize streaming detokenizer and get full output
                detok = self._detokenizer_pool.get(request_id)
                if detok is not None:
                    detok.finalize()
                    output.output_text = detok.text
                else:
                    output.output_text = self._decode_tokens(request.output_token_ids)
                request.output_text = output.output_text
                self._cleanup_detokenizer(request_id)

                # Extract cache for future reuse (critical for agentic multi-turn)
                if hasattr(response, "prompt_cache"):
                    try:
                        # prompt_cache may be callable or direct attribute
                        if callable(response.prompt_cache):
                            raw_cache = response.prompt_cache()
                        else:
                            raw_cache = response.prompt_cache

                        if raw_cache and self._prefix_cache is not None:
                            # Extract actual tensor states for cache persistence
                            # across BatchGenerator recreation
                            extracted = self._prefix_cache.extract_cache(raw_cache)
                            if extracted:
                                request._cache_state.decoded_cache = extracted
                            else:
                                request._cache_state.decoded_cache = raw_cache
                    except Exception as e:
                        logger.warning(
                            f"[paged_cache] request={request_id[:12]} extract exception: {e}"
                        )

                    # No second extraction needed — extract_cache handles format normalization

                    if request._cache_state.decoded_cache:
                        _full_tokens = list(request.prompt_token_ids) + list(
                            request.output_token_ids
                        )

                self.total_completion_tokens += request.num_output_tokens
                self.num_requests_processed += 1

                logger.debug(
                    f"Request {request_id} finished: {response.finish_reason}, "
                    f"{request.num_output_tokens} tokens"
                )

            outputs.append(output)

        return outputs, finished_ids

    def _cleanup_finished(self, finished_ids: Set[str]) -> None:
        """Clean up finished requests and store caches for reuse."""
        for request_id in finished_ids:
            request = self.running.get(request_id)

            # Store cache for future reuse
            if (
                request is not None
                and request.prompt_token_ids
                and self._prefix_cache is not None
            ):
                _store_cache = request._cache_state.decoded_cache
                if _store_cache:
                    _full_tokens = list(request.prompt_token_ids) + list(
                        request.output_token_ids
                    )
                    _store_tokens = _full_tokens
                    try:
                        self._prefix_cache.store(request, _store_tokens, _store_cache)
                    except Exception as e:
                        logger.debug(
                            f"[cache_store] store failed for {request_id}: {e}"
                        )
                try:
                    self._prefix_cache.release(request)
                except Exception as e:
                    logger.debug(f"[cache_store] release failed for {request_id}: {e}")

            # Evaluate stored cache tensors incrementally (per-layer) to prevent
            # a deferred batch evaluation spike when all lazy ops resolve at once.
            # This spreads the VRAM cost across smaller per-layer evaluations.
            if request is not None and request._cache_state.decoded_cache:
                for layer in request._cache_state.decoded_cache:
                    if isinstance(layer, dict) and "state" in layer:
                        keys, values = layer["state"]
                        mx.eval(keys, values)
                    elif hasattr(layer, "keys") and hasattr(layer, "values"):
                        keys_attr = layer.keys
                        values_attr = layer.values
                        if not callable(keys_attr) and not callable(values_attr):
                            mx.eval(keys_attr, values_attr)

            # Evaluate boundary-captured tensors (defensive against lazy MLX GC)
            for _state_attr in ("_sys_prompt_state", "_conv_end_state"):
                _state = (
                    getattr(request, _state_attr, None) if request is not None else None
                )
                if _state and isinstance(_state, list):
                    for layer_dict in _state:
                        if isinstance(layer_dict, dict) and "state" in layer_dict:
                            mx.eval(*layer_dict["state"])
            _b_states = (
                getattr(request, "_boundary_states", None)
                if request is not None
                else None
            )
            if _b_states and isinstance(_b_states, dict):
                for _bstate in _b_states.values():
                    if isinstance(_bstate, list):
                        for layer_dict in _bstate:
                            if isinstance(layer_dict, dict) and "state" in layer_dict:
                                mx.eval(*layer_dict["state"])

            # Release all cache references on the request so Metal buffers
            # can be freed.  The prefix cache (if any) holds its own copy;
            # keeping a second reference here pins the buffers in wired memory
            # until the request object is GC'd (issue #442).
            if request is not None:
                request._cache_state.cache = None
                request._cache_state.decoded_cache = None
                request._sys_prompt_state = None
                request._conv_end_state = None
                request._turn_boundary_states = {}
                request._boundary_states = {}

            # Remove from running
            if request_id in self.running:
                del self.running[request_id]

            # Remove UID mappings
            if request_id in self.request_id_to_uid:
                uid = self.request_id_to_uid[request_id]
                if uid in self.uid_to_request_id:
                    del self.uid_to_request_id[uid]
                del self.request_id_to_uid[request_id]

            # Track as finished
            self.finished_req_ids.add(request_id)

        # Free Metal command buffers after cleanup (prevents end-of-generation spike)
        if finished_ids:
            mx.clear_cache()

    def _is_cache_corruption_error(self, error: Exception) -> bool:
        """Check if an error indicates cache corruption."""
        error_str = str(error)
        return any(pattern in error_str for pattern in CACHE_CORRUPTION_PATTERNS)

    def _is_stream_thread_error(self, error: Exception) -> bool:
        """Check if an error indicates MLX stream/thread ownership mismatch."""
        error_str = str(error)
        return "no Stream(" in error_str or "no Stream(gpu" in error_str

    def _recover_from_cache_error(self) -> None:
        """Recover from cache corruption error."""
        # Properly close batch generator (this is the source of the corruption)
        self._close_batch_generator()
        self._current_sampler_params = None

        # Clear caches
        if self._prefix_cache is not None:
            self._prefix_cache.clear()

        # Clear UID mappings
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()

        logger.info("Cache recovery completed")

    def _recover_from_generation_error(self) -> Set[str]:
        """Recover from fatal generation error (OOM, Metal crash).

        Aborts all running requests and resets batch state.
        Unlike cache corruption recovery, does NOT reschedule —
        the request that OOMed would just OOM again.

        Returns:
            Set of aborted request IDs.
        """
        # Close batch generator (clears _partial state, active_batch)
        self._close_batch_generator()
        self._current_sampler_params = None

        # Abort all running requests
        aborted_ids: Set[str] = set()
        for request_id in list(self.running):
            request = self.running.get(request_id)
            if request is not None:
                request.set_finished(RequestStatus.FINISHED_ABORTED)
            aborted_ids.add(request_id)
            self.finished_req_ids.add(request_id)
        self.running.clear()
        self._detokenizer_pool.clear()

        # Clear UID mappings (batch generator is gone)
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()

        # Release Metal memory
        mx.clear_cache()

        logger.warning(
            f"[generation_error_recovery] aborted {len(aborted_ids)} running requests, "
            f"batch generator closed, Metal cache cleared"
        )
        return aborted_ids

    def _reschedule_running_requests(self) -> None:
        """Move running requests back to waiting queue for retry."""
        count = len(self.running)
        for request_id, request in list(self.running.items()):
            # Reset request state
            request.status = RequestStatus.WAITING
            request.batch_uid = None
            # Release any pinned leaf so refcounts don't leak when the
            # request is bounced back to waiting and re-fetches on retry.
            if self._prefix_cache is not None:
                self._prefix_cache.release(request)
            request._cache_state.cache = None
            request._cache_state.cached_tokens = 0
            request._cache_state.remaining_tokens = request.prompt_token_ids

            # Move to waiting queue (at front for priority)
            self.waiting.appendleft(request)
            del self.running[request_id]

        if count > 0:
            logger.info(f"Rescheduled {count} requests for retry")

    def step(self, max_retries: int = 1) -> SchedulerOutput:
        """
        Execute one scheduling step with automatic error recovery.

        This method:
        1. Schedules waiting requests into the batch
        2. Runs one generation step via BatchGenerator
        3. Processes outputs and handles finished requests
        4. Automatically recovers from cache corruption errors

        Args:
            max_retries: Number of times to retry on cache errors (default 1)

        Returns:
            SchedulerOutput with results of this step
        """
        output = SchedulerOutput()

        # Process pending aborts FIRST (in executor thread, safe for MLX)
        self._process_pending_aborts()

        for attempt in range(max_retries + 1):
            try:
                # Schedule waiting requests
                scheduled = self._schedule_waiting()
                output.scheduled_request_ids = [r.request_id for r in scheduled]
                output.num_scheduled_tokens = sum(
                    r.num_prompt_tokens for r in scheduled
                )

                # Run generation step if we have running requests
                if self.batch_generator is not None and self.running:
                    _gb = getattr(self.batch_generator, "_generation_batch", None)
                    if _gb is not None and _gb.uids:
                        for _e, _uid in enumerate(_gb.uids):
                            _rid = self.uid_to_request_id.get(_uid)
                            _req = self.running.get(_rid) if _rid else None
                            if _req is not None and self._prefix_cache is not None:
                                self._prefix_cache.update_n_minus_one(
                                    _req, _gb.prompt_cache, _e
                                )
                    # if logger.isEnabledFor(logging.DEBUG):
                    #     for req in self.running.values():
                    #         all_ids = list(req.prompt_token_ids) + list(req.output_token_ids)
                    #         text = self.tokenizer.decode(all_ids)
                    #         logger.debug(f"[decode_step] request_id={req.request_id}\n{text}")
                    result = self.batch_generator.next()
                    output.has_work = True

                    # mlx-lm >=0.31.x returns (prompt_responses, generation_responses);
                    # older versions returned a flat list.
                    if isinstance(result, tuple):
                        prompt_responses, responses = result
                        self._handle_prompt_segment_ends(prompt_responses)
                    else:
                        responses = result

                    if responses:
                        outputs, finished_ids = self._process_batch_responses(responses)
                        output.outputs = outputs
                        output.finished_request_ids = finished_ids
                        self._cleanup_finished(finished_ids)

                # Success - break out of retry loop
                break

            except TypeError as e:
                # Catch the NoneType error specifically
                if self._is_cache_corruption_error(e):
                    if attempt < max_retries:
                        logger.warning(
                            f"Cache corruption detected (attempt {attempt + 1}), "
                            f"performing recovery and retry..."
                        )
                        # Deep reset to recover
                        self._recover_from_cache_error()
                        # Re-add any running requests back to waiting
                        self._reschedule_running_requests()
                    else:
                        logger.error(
                            f"Cache corruption not recoverable after "
                            f"{max_retries + 1} attempts"
                        )
                        raise
                else:
                    raise
            except Exception as e:
                if self._is_stream_thread_error(e):
                    raise
                import traceback

                logger.error(
                    f"Error in batch generation step: {e}\n{traceback.format_exc()}"
                )
                # Recover from fatal errors (OOM, Metal crash) instead of
                # re-raising, which would cause infinite loop in engine_core.
                aborted_ids = self._recover_from_generation_error()
                for rid in aborted_ids:
                    output.outputs.append(
                        RequestOutput(
                            request_id=rid,
                            finished=True,
                            finish_reason="error",
                        )
                    )
                output.finished_request_ids = aborted_ids
                break

        # Clear finished tracking for next step
        old_finished = self.finished_req_ids
        self.finished_req_ids = set()

        # Adaptive interval: scale inversely with concurrency to prevent
        # Metal resource handle exhaustion under high-concurrency workloads.
        active_seqs = len(self.running)
        min_interval = max(4, self._clear_cache_interval // 4)
        effective_interval = max(
            min_interval, self._clear_cache_interval // max(1, active_seqs // 8)
        )

        self._step_count += 1
        if self._step_count % effective_interval == 0:
            # GenerationBatch.tokens is List[List[int]] — no lazy eval needed.
            mx.clear_cache()

        # Periodically log memory stats for monitoring
        if self._step_count % self._memory_log_interval == 0:
            try:
                if mx.metal.is_available():
                    active_gb = mx.get_active_memory() / 1e9
                    peak_gb = mx.get_peak_memory() / 1e9
                    cache_gb = mx.get_cache_memory() / 1e9
                    logger.info(
                        f"[Metal memory] active={active_gb:.1f}GB "
                        f"peak={peak_gb:.1f}GB cache={cache_gb:.1f}GB "
                        f"step={self._step_count} "
                        f"running={len(self.running)} waiting={len(self.waiting)}"
                    )
            except Exception:
                pass

        return output

    def get_request(self, request_id: str) -> Optional[Request]:
        """Get a request by ID."""
        return self.requests.get(request_id)

    def remove_finished_request(self, request_id: str) -> Optional[Request]:
        """Remove a finished request from tracking."""
        return self.requests.pop(request_id, None)

    def get_running_requests_info(self) -> List[Dict[str, Any]]:
        """Per-request details for status endpoint."""
        import time as _time

        now = _time.time()
        result = []

        # Waiting requests
        for req in self.waiting:
            result.append(
                {
                    "request_id": req.request_id,
                    "status": "waiting",
                    "phase": "queued",
                    "elapsed_s": round(now - req.arrival_time, 2),
                    "prompt_tokens": req.num_prompt_tokens,
                    "completion_tokens": 0,
                    "max_tokens": req.max_tokens,
                    "progress": 0.0,
                    "tokens_per_second": None,
                    "ttft_s": None,
                    "cache_hit_type": req.cache_hit_type,
                    "cached_tokens": req.cached_tokens,
                }
            )

        # Running requests
        for req in self.running.values():
            n_out = req.num_output_tokens
            elapsed = now - req.arrival_time

            # Phase detection
            if n_out == 0:
                phase = "prefill"
            else:
                phase = "generation"

            # Tokens per second (generation phase only)
            tok_s = None
            ttft = None
            if req.first_token_time is not None:
                ttft = round(req.first_token_time - req.arrival_time, 3)
                gen_elapsed = now - req.first_token_time
                if gen_elapsed > 0 and n_out > 0:
                    tok_s = round(n_out / gen_elapsed, 1)

            # Progress: completion_tokens / max_tokens
            progress = round(n_out / req.max_tokens, 3) if req.max_tokens > 0 else 0.0

            result.append(
                {
                    "request_id": req.request_id,
                    "status": "running",
                    "phase": phase,
                    "elapsed_s": round(elapsed, 2),
                    "prompt_tokens": req.num_prompt_tokens,
                    "completion_tokens": n_out,
                    "max_tokens": req.max_tokens,
                    "progress": min(progress, 1.0),
                    "tokens_per_second": tok_s,
                    "ttft_s": ttft,
                    "cache_hit_type": req.cache_hit_type,
                    "cached_tokens": req.cached_tokens,
                }
            )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        stats = {
            "num_waiting": len(self.waiting),
            "num_running": len(self.running),
            "num_requests_processed": self.num_requests_processed,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }
        # Include Metal memory stats
        try:
            if mx.metal.is_available():
                stats["metal_active_memory_gb"] = round(mx.get_active_memory() / 1e9, 2)
                stats["metal_peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
                stats["metal_cache_memory_gb"] = round(mx.get_cache_memory() / 1e9, 2)
        except Exception:
            pass

        # Include cache stats
        if self._prefix_cache is not None:
            stats["cache"] = self._prefix_cache.get_stats()
        return stats

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self._prefix_cache is not None:
            return self._prefix_cache.get_stats()
        return None

    def clear_runtime_caches(self) -> Dict[str, bool]:
        """Clear prefix-cache state without resetting scheduler/request state."""
        if self._prefix_cache is not None:
            self._prefix_cache.clear()
            return {"cache": True}
        return {}

    def reset(self) -> None:
        """Reset the scheduler state."""
        # Drain any pending deferred aborts
        self._pending_abort_ids.clear()

        # Abort all requests directly (reset is synchronous)
        for request_id in list(self.requests.keys()):
            self._do_abort_request(request_id)

        self.waiting.clear()
        self.running.clear()
        self.requests.clear()
        self.finished_req_ids.clear()
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()
        self._detokenizer_pool.clear()
        self._close_batch_generator()
        self._current_sampler_params = None

        # Clear caches
        self.clear_runtime_caches()

        # Close SSD tier on reset
        self.close_ssd_tier()

    def deep_reset(self) -> None:
        """
        Deep reset that clears ALL cache state including model-level caches.

        This is more aggressive than reset() and should be used when
        switching engines or recovering from errors.
        """
        # Standard reset first
        self.reset()

        # Clear any model-level cache state
        # MLX models may have internal cache references
        if hasattr(self.model, "cache"):
            self.model.cache = None

        # Some MLX models store cache in layers
        if hasattr(self.model, "layers"):
            for layer in self.model.layers:
                if hasattr(layer, "cache"):
                    layer.cache = None
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "cache"):
                    layer.self_attn.cache = None

        # Force garbage collection of any lingering cache objects
        import gc

        gc.collect()

        logger.info("Deep reset completed - all caches cleared")

    # -----------------------------------------------------------------
    # Cache persistence
    # -----------------------------------------------------------------

    def save_cache_to_disk(self, cache_dir: str) -> bool:
        """Save prefix cache to disk for persistence across restarts."""
        if self._prefix_cache is not None:
            return self._prefix_cache.save(cache_dir)
        return False

    def load_cache_from_disk(self, cache_dir: str) -> int:
        """Load prefix cache from disk. Returns number of entries loaded."""
        if self._prefix_cache is not None:
            return self._prefix_cache.load(cache_dir)
        return 0

    def clear_prefix_cache(self) -> None:
        """Clear the in-memory prefix cache (keeps disk cache untouched)."""
        if self._prefix_cache is not None:
            self._prefix_cache.clear()

    def close_ssd_tier(self) -> None:
        """Shut down the SSD cache tier if present."""
        if self._prefix_cache is not None:
            self._prefix_cache.close()

    def _handle_prompt_segment_ends(self, prompt_responses) -> None:
        """Save turn-cache state at each completed prompt segment boundary."""
        if self._prefix_cache is None:
            return
        for resp in prompt_responses:
            if not resp.end_of_segment or resp.end_of_prompt:
                continue
            uid = resp.uid
            request_id = self.uid_to_request_id.get(uid)
            if not request_id:
                continue
            request = self.requests.get(request_id)
            if not request:
                continue
            processed = resp.progress[0]
            bg = self.batch_generator
            pb = getattr(bg, "_prompt_batch", None)
            if pb is None or uid not in pb.uids:
                continue
            idx = pb.uids.index(uid)

            _cb_pre = mx.get_active_memory()
            per_uid_cache = pb.extract_cache(idx)
            mx.eval(*[c.keys for c in per_uid_cache if hasattr(c, "keys") and c.keys is not None],
                    *[c.values for c in per_uid_cache if hasattr(c, "values") and c.values is not None])
            _cb_post_extract = mx.get_active_memory()

            extracted = self._prefix_cache.extract_cache(per_uid_cache)
            _cb_post_extract_states = mx.get_active_memory()

            if extracted:
                total = (request._cache_state.cached_tokens or 0) + processed
                self._prefix_cache.on_prefill_checkpoint(request, total, extracted)
            _cb_post_checkpoint = mx.get_active_memory()

            del per_uid_cache, extracted
            import gc as _gc
            _gc.collect()
            mx.clear_cache()
            _cb_post_gc = mx.get_active_memory()

            logger.warning(
                "[memprobe:segend] processed=%d pre=%.2fGB "
                "post_extract=%.2fGB(+%.2f) post_states=%.2fGB(+%.2f) "
                "post_checkpoint=%.2fGB(+%.2f) post_gc=%.2fGB(net+%.2f)",
                processed,
                _cb_pre / 1e9,
                _cb_post_extract / 1e9,
                (_cb_post_extract - _cb_pre) / 1e9,
                _cb_post_extract_states / 1e9,
                (_cb_post_extract_states - _cb_post_extract) / 1e9,
                _cb_post_checkpoint / 1e9,
                (_cb_post_checkpoint - _cb_post_extract_states) / 1e9,
                _cb_post_gc / 1e9,
                (_cb_post_gc - _cb_pre) / 1e9,
            )
