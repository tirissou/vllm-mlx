# SPDX-License-Identifier: Apache-2.0
"""Roundtrip tests: TurnCacheManager prefill→store→fetch→decode ≈ pure prefill→decode.

Verifies via the public interface (fetch / on_prefill_checkpoint / store) that
using the cache system does not introduce numerical errors beyond the expected
int8 quantization tolerance.
"""

import math

import mlx.core as mx
import pytest

from vllm_mlx.kv_cache import RequestCacheState
from vllm_mlx.prefix_cache_adapters import TurnCacheManager
from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig

# int8 quantization introduces ~1% error; attention softmax averaging reduces it further
ATOL = 0.05


def _attention(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.swapaxes(-1, -2)) * scale
    probs = mx.softmax(scores, axis=-1)
    return probs @ v


def _make_manager():
    cfg = TurnPrefixCacheConfig(checkpoint_stride=0, max_memory_gb=4.0)
    return TurnCacheManager(TurnPrefixCache(cfg))


def _make_request(request_id, token_ids, boundaries):
    req = Request(
        request_id=request_id,
        prompt="",
        sampling_params=SamplingParams(),
        prompt_token_ids=token_ids,
        _turn_boundaries=boundaries,
    )
    req._cache_state = RequestCacheState()
    return req


class TestKVCacheRoundtrip:
    def test_cache_miss_on_first_request(self):
        """fetch returns None when nothing has been stored yet."""
        manager = _make_manager()
        req = _make_request("r1", [0, 1, 2, 3], boundaries=[2])
        assert manager.fetch(req) is None

    def test_cached_decode_attention_matches_fresh(self):
        """Decode step using TurnCacheManager fetch produces same attention as fresh prefill.

        Flow:
          req1: fetch (miss) → on_prefill_checkpoint at sys boundary → store
          req2: fetch (hit) → prefill remaining tokens manually → decode → compare
        """
        from mlx_lm.models.cache import KVCache

        manager = _make_manager()

        n_sys, n_user, n_heads, head_dim = 2, 3, 2, 64
        token_ids = list(range(n_sys + n_user))
        boundaries = [n_sys]

        mx.random.seed(0)
        k_per_tok = [mx.random.normal((1, n_heads, 1, head_dim)) for _ in token_ids]
        v_per_tok = [mx.random.normal((1, n_heads, 1, head_dim)) for _ in token_ids]
        q_decode = mx.random.normal((1, n_heads, 1, head_dim))
        k_decode = mx.random.normal((1, n_heads, 1, head_dim))
        v_decode = mx.random.normal((1, n_heads, 1, head_dim))
        mx.eval(*k_per_tok, *v_per_tok, q_decode, k_decode, v_decode)

        # --- Fresh path: prefill all tokens then decode ---
        kv_fresh = KVCache()
        for k, v in zip(k_per_tok, v_per_tok):
            kv_fresh.update_and_fetch(k, v)
        k_all, v_all = kv_fresh.update_and_fetch(k_decode, v_decode)
        logits_fresh = _attention(q_decode, k_all, v_all)
        mx.eval(logits_fresh)

        # --- Populate cache via public interface ---
        req1 = _make_request("r1", token_ids, boundaries)
        assert manager.fetch(req1) is None

        kv_sys = KVCache()
        for k, v in zip(k_per_tok[:n_sys], v_per_tok[:n_sys]):
            kv_sys.update_and_fetch(k, v)
        manager.on_prefill_checkpoint(
            req1, total_tokens_prefilled=n_sys, extracted_cache=[kv_sys]
        )

        kv_full = KVCache()
        for k, v in zip(k_per_tok, v_per_tok):
            kv_full.update_and_fetch(k, v)
        req1.output_token_ids = [999]
        manager.store(req1, cache=[kv_full])

        # --- Cached path: fetch, prefill remaining, decode ---
        req2 = _make_request("r2", token_ids, boundaries)
        hit = manager.fetch(req2)

        assert hit is not None
        assert hit.cached_tokens == n_sys
        assert hit.remaining_tokens == token_ids[n_sys:]

        # hit.cache[0] is a QuantizedKVCache covering the first n_sys tokens
        assembled = hit.cache[0]
        raw_state = assembled.state
        k_cached = mx.dequantize(
            *raw_state[0], group_size=assembled.group_size, bits=assembled.bits
        )
        v_cached = mx.dequantize(
            *raw_state[1], group_size=assembled.group_size, bits=assembled.bits
        )

        k_remaining = mx.concatenate(k_per_tok[n_sys:], axis=-2)
        v_remaining = mx.concatenate(v_per_tok[n_sys:], axis=-2)

        k_all_cached = mx.concatenate([k_cached, k_remaining, k_decode], axis=-2)
        v_all_cached = mx.concatenate([v_cached, v_remaining, v_decode], axis=-2)
        logits_cached = _attention(q_decode, k_all_cached, v_all_cached)
        mx.eval(logits_cached)

        manager.release(hit.handle)
        assert float(mx.max(mx.abs(logits_cached - logits_fresh))) < ATOL


class TestRotatingKVCacheRoundtrip:
    def test_cached_decode_attention_matches_fresh(self):
        """RotatingKVCache: fetch-assembled state produces same attention as fresh decode."""
        from mlx_lm.models.cache import RotatingKVCache

        manager = _make_manager()

        n_sys, n_user, max_size, n_heads, head_dim = 4, 2, 16, 2, 64
        # n_sys + n_user < max_size — buffer does not wrap in this test
        token_ids = list(range(n_sys + n_user))
        boundaries = [n_sys]

        mx.random.seed(1)
        k_per_tok = [mx.random.normal((1, n_heads, 1, head_dim)) for _ in token_ids]
        v_per_tok = [mx.random.normal((1, n_heads, 1, head_dim)) for _ in token_ids]
        q_decode = mx.random.normal((1, n_heads, 1, head_dim))
        k_decode = mx.random.normal((1, n_heads, 1, head_dim))
        v_decode = mx.random.normal((1, n_heads, 1, head_dim))
        mx.eval(*k_per_tok, *v_per_tok, q_decode, k_decode, v_decode)

        # --- Fresh path ---
        rk_fresh = RotatingKVCache(max_size=max_size, keep=0)
        for k, v in zip(k_per_tok, v_per_tok):
            rk_fresh.update_and_fetch(k, v)
        k_all, v_all = rk_fresh.update_and_fetch(k_decode, v_decode)
        logits_fresh = _attention(q_decode, k_all, v_all)
        mx.eval(logits_fresh)

        # --- Populate cache via public interface ---
        req1 = _make_request("r1", token_ids, boundaries)
        assert manager.fetch(req1) is None

        rk_sys = RotatingKVCache(max_size=max_size, keep=0)
        for k, v in zip(k_per_tok[:n_sys], v_per_tok[:n_sys]):
            rk_sys.update_and_fetch(k, v)
        manager.on_prefill_checkpoint(
            req1, total_tokens_prefilled=n_sys, extracted_cache=[rk_sys]
        )

        rk_full = RotatingKVCache(max_size=max_size, keep=0)
        for k, v in zip(k_per_tok, v_per_tok):
            rk_full.update_and_fetch(k, v)
        req1.output_token_ids = [999]
        manager.store(req1, cache=[rk_full])

        # --- Cached path: fetch, continue with assembled cache ---
        req2 = _make_request("r2", token_ids, boundaries)
        hit = manager.fetch(req2)

        assert hit is not None
        assert hit.cached_tokens == n_sys

        # For RotatingKVCache the assembled cache is a RotatingKVCache object;
        # update_and_fetch appends the remaining and decode tokens.
        assembled = hit.cache[0]
        assert isinstance(assembled, RotatingKVCache)

        for k, v in zip(k_per_tok[n_sys:], v_per_tok[n_sys:]):
            assembled.update_and_fetch(k, v)
        k_all_cached, v_all_cached = assembled.update_and_fetch(k_decode, v_decode)
        logits_cached = _attention(q_decode, k_all_cached, v_all_cached)
        mx.eval(logits_cached)

        manager.release(hit.handle)
        assert float(mx.max(mx.abs(logits_cached - logits_fresh))) < ATOL

    def test_wrapped_ring_buffer_cached_decode_matches_fresh(self):
        """RotatingKVCache with n_tokens > max_size: ring buffer has wrapped; restored
        cache must produce chronologically correct keys so decode attention matches fresh.

        This is a regression test for a bug in _segment where the raw offset (> max_size
        after wrapping) was passed to _linearize instead of offset % max_size, causing
        the ring buffer to be returned in physical order rather than chronological order.
        """
        from mlx_lm.models.cache import RotatingKVCache

        manager = _make_manager()

        max_size = 4  # small buffer so wrapping happens quickly
        n_sys = 6  # n_sys > max_size — the ring has wrapped by the time we checkpoint
        n_user = 2
        n_heads, head_dim = 1, 64
        token_ids = list(range(n_sys + n_user))
        boundaries = [n_sys]

        mx.random.seed(5)
        k_per_tok = [mx.random.normal((1, n_heads, 1, head_dim)) for _ in token_ids]
        v_per_tok = [mx.random.normal((1, n_heads, 1, head_dim)) for _ in token_ids]
        q_decode = mx.random.normal((1, n_heads, 1, head_dim))
        k_decode = mx.random.normal((1, n_heads, 1, head_dim))
        v_decode = mx.random.normal((1, n_heads, 1, head_dim))
        mx.eval(*k_per_tok, *v_per_tok, q_decode, k_decode, v_decode)

        # --- Fresh path ---
        rk_fresh = RotatingKVCache(max_size=max_size, keep=0)
        for k, v in zip(k_per_tok, v_per_tok):
            rk_fresh.update_and_fetch(k, v)
        k_all, v_all = rk_fresh.update_and_fetch(k_decode, v_decode)
        logits_fresh = _attention(q_decode, k_all, v_all)
        mx.eval(logits_fresh)

        # --- Cached path ---
        req1 = _make_request("r1", token_ids, boundaries)
        assert manager.fetch(req1) is None

        rk_sys = RotatingKVCache(max_size=max_size, keep=0)
        for k, v in zip(k_per_tok[:n_sys], v_per_tok[:n_sys]):
            rk_sys.update_and_fetch(k, v)
        manager.on_prefill_checkpoint(
            req1, total_tokens_prefilled=n_sys, extracted_cache=[rk_sys]
        )

        rk_full = RotatingKVCache(max_size=max_size, keep=0)
        for k, v in zip(k_per_tok, v_per_tok):
            rk_full.update_and_fetch(k, v)
        req1.output_token_ids = [999]
        manager.store(req1, cache=[rk_full])

        req2 = _make_request("r2", token_ids, boundaries)
        hit = manager.fetch(req2)

        assert hit is not None
        assert hit.cached_tokens == n_sys

        assembled = hit.cache[0]
        assert isinstance(assembled, RotatingKVCache)

        for k, v in zip(k_per_tok[n_sys:], v_per_tok[n_sys:]):
            assembled.update_and_fetch(k, v)
        k_all_cached, v_all_cached = assembled.update_and_fetch(k_decode, v_decode)
        logits_cached = _attention(q_decode, k_all_cached, v_all_cached)
        mx.eval(logits_cached)

        manager.release(hit.handle)
        assert float(mx.max(mx.abs(logits_cached - logits_fresh))) < ATOL


class TestMultiTurnRoundtrip:
    def test_second_turn_cached_prefill_matches_fresh_decode(self):
        """Turn-2 decode using turn-1 cached state matches full fresh-sequence decode.

        Conversation layout (token IDs = sequence positions):
          sys:   [0, 1]          (n_sys=2)
          user1: [2, 3]          (n_user1=2, turn-1 prompt)
          asst1: [4, 5]          (n_asst1=2, turn-1 output_token_ids)
          user2: [6, 7]          (n_user2=2, turn-2 prompt suffix)
          decode: position 8

        Turn 1 stores the KV covering sys+user1+asst1 (positions 0-5).
        Turn 2 fetch hits that stored state, prefills user2, then decodes.
        """
        from mlx_lm.models.cache import KVCache

        manager = _make_manager()

        n_sys, n_user1, n_asst1, n_user2 = 2, 2, 2, 2
        n_heads, head_dim = 2, 64
        n_cached = n_sys + n_user1 + n_asst1  # positions covered by stored cache
        n_total = n_cached + n_user2  # all turn-2 prefill positions

        # Token IDs equal position indices for clarity
        turn1_prompt_ids = list(range(n_sys + n_user1))  # [0,1,2,3]
        asst1_output_ids = list(range(n_sys + n_user1, n_cached))  # [4,5]
        turn2_ids = list(range(n_total))  # [0..7]

        # Boundaries: turn1 has sys boundary only; turn2 has sys + end-of-asst1
        boundaries_turn1 = [n_sys]
        boundaries_turn2 = [n_sys, n_cached]

        mx.random.seed(2)
        k_per_tok = [
            mx.random.normal((1, n_heads, 1, head_dim)) for _ in range(n_total)
        ]
        v_per_tok = [
            mx.random.normal((1, n_heads, 1, head_dim)) for _ in range(n_total)
        ]
        q_decode = mx.random.normal((1, n_heads, 1, head_dim))
        k_decode = mx.random.normal((1, n_heads, 1, head_dim))
        v_decode = mx.random.normal((1, n_heads, 1, head_dim))
        mx.eval(*k_per_tok, *v_per_tok, q_decode, k_decode, v_decode)

        # --- Fresh path: prefill all turn-2 tokens [0..7] then decode ---
        kv_fresh = KVCache()
        for k, v in zip(k_per_tok, v_per_tok):
            kv_fresh.update_and_fetch(k, v)
        k_all, v_all = kv_fresh.update_and_fetch(k_decode, v_decode)
        logits_fresh = _attention(q_decode, k_all, v_all)
        mx.eval(logits_fresh)

        # --- Turn 1: miss → checkpoint at sys → store (cache covers positions 0-5) ---
        req1 = _make_request("r1", turn1_prompt_ids, boundaries_turn1)
        assert manager.fetch(req1) is None

        kv_sys = KVCache()
        for k, v in zip(k_per_tok[:n_sys], v_per_tok[:n_sys]):
            kv_sys.update_and_fetch(k, v)
        manager.on_prefill_checkpoint(
            req1, total_tokens_prefilled=n_sys, extracted_cache=[kv_sys]
        )

        # kv_full covers the full turn-1 sequence: sys + user1 + asst1 (positions 0-5)
        kv_full = KVCache()
        for k, v in zip(k_per_tok[:n_cached], v_per_tok[:n_cached]):
            kv_full.update_and_fetch(k, v)
        req1.output_token_ids = asst1_output_ids
        manager.store(req1, cache=[kv_full])

        # --- Turn 2: fetch → cache covers positions 0-5; prefill user2; decode ---
        req2 = _make_request("r2", turn2_ids, boundaries_turn2)
        hit = manager.fetch(req2)

        assert hit is not None
        # Stored response node covers sys+user1+asst1 = n_cached positions
        assert hit.cached_tokens == n_cached
        assert hit.remaining_tokens == turn2_ids[n_cached:]

        assembled = hit.cache[0]
        raw_state = assembled.state
        k_cached_arr = mx.dequantize(
            *raw_state[0], group_size=assembled.group_size, bits=assembled.bits
        )
        v_cached_arr = mx.dequantize(
            *raw_state[1], group_size=assembled.group_size, bits=assembled.bits
        )

        k_remaining = mx.concatenate(k_per_tok[n_cached:], axis=-2)
        v_remaining = mx.concatenate(v_per_tok[n_cached:], axis=-2)

        k_all_cached = mx.concatenate([k_cached_arr, k_remaining, k_decode], axis=-2)
        v_all_cached = mx.concatenate([v_cached_arr, v_remaining, v_decode], axis=-2)
        logits_cached = _attention(q_decode, k_all_cached, v_all_cached)
        mx.eval(logits_cached)

        manager.release(hit.handle)
        assert float(mx.max(mx.abs(logits_cached - logits_fresh))) < ATOL
