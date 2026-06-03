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
        assert manager.fetch(req) is False

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
        assert manager.fetch(req1) is False

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
        assert manager.fetch(req2) is True
        cs = req2._cache_state

        assert cs.hit_type == "hit"
        assert cs.cached_tokens == n_sys
        assert cs.remaining_tokens == token_ids[n_sys:]

        # cs.cache[0] is a QuantizedKVCache covering the first n_sys tokens
        assembled = cs.cache[0]
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
        assert manager.fetch(req1) is False

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
        assert manager.fetch(req2) is True
        cs = req2._cache_state

        assert cs.cached_tokens == n_sys

        # For RotatingKVCache the assembled cache is a RotatingKVCache object;
        # update_and_fetch appends the remaining and decode tokens.
        assembled = cs.cache[0]
        assert isinstance(assembled, RotatingKVCache)

        for k, v in zip(k_per_tok[n_sys:], v_per_tok[n_sys:]):
            assembled.update_and_fetch(k, v)
        k_all_cached, v_all_cached = assembled.update_and_fetch(k_decode, v_decode)
        logits_cached = _attention(q_decode, k_all_cached, v_all_cached)
        mx.eval(logits_cached)

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
        assert manager.fetch(req1) is False

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
        assert manager.fetch(req2) is True
        cs = req2._cache_state

        assert cs.cached_tokens == n_sys

        assembled = cs.cache[0]
        assert isinstance(assembled, RotatingKVCache)

        for k, v in zip(k_per_tok[n_sys:], v_per_tok[n_sys:]):
            assembled.update_and_fetch(k, v)
        k_all_cached, v_all_cached = assembled.update_and_fetch(k_decode, v_decode)
        logits_cached = _attention(q_decode, k_all_cached, v_all_cached)
        mx.eval(logits_cached)

        assert float(mx.max(mx.abs(logits_cached - logits_fresh))) < ATOL


class TestRotatingKVCacheKeepRoundtrip:
    def test_keep_gt_zero_no_wrap(self):
        """RotatingKVCache with keep>0, buffer not wrapped: assembled cache matches fresh."""
        from mlx_lm.models.cache import RotatingKVCache

        manager = _make_manager()

        keep = 2
        max_size = 16
        n_sys, n_user = 4, 3
        n_heads, head_dim = 2, 64
        token_ids = list(range(n_sys + n_user))
        boundaries = [n_sys]

        mx.random.seed(10)
        k_per_tok = [mx.random.normal((1, n_heads, 1, head_dim)) for _ in token_ids]
        v_per_tok = [mx.random.normal((1, n_heads, 1, head_dim)) for _ in token_ids]
        q_decode = mx.random.normal((1, n_heads, 1, head_dim))
        k_decode = mx.random.normal((1, n_heads, 1, head_dim))
        v_decode = mx.random.normal((1, n_heads, 1, head_dim))
        mx.eval(*k_per_tok, *v_per_tok, q_decode, k_decode, v_decode)

        # --- Fresh path ---
        rk_fresh = RotatingKVCache(max_size=max_size, keep=keep)
        for k, v in zip(k_per_tok, v_per_tok):
            rk_fresh.update_and_fetch(k, v)
        k_all, v_all = rk_fresh.update_and_fetch(k_decode, v_decode)
        logits_fresh = _attention(q_decode, k_all, v_all)
        mx.eval(logits_fresh)

        # --- Populate cache ---
        req1 = _make_request("r1", token_ids, boundaries)
        assert manager.fetch(req1) is False

        rk_sys = RotatingKVCache(max_size=max_size, keep=keep)
        for k, v in zip(k_per_tok[:n_sys], v_per_tok[:n_sys]):
            rk_sys.update_and_fetch(k, v)
        manager.on_prefill_checkpoint(
            req1, total_tokens_prefilled=n_sys, extracted_cache=[rk_sys]
        )

        rk_full = RotatingKVCache(max_size=max_size, keep=keep)
        for k, v in zip(k_per_tok, v_per_tok):
            rk_full.update_and_fetch(k, v)
        req1.output_token_ids = [999]
        manager.store(req1, cache=[rk_full])

        # --- Cached path ---
        req2 = _make_request("r2", token_ids, boundaries)
        assert manager.fetch(req2) is True
        cs = req2._cache_state

        assembled = cs.cache[0]
        assert isinstance(assembled, RotatingKVCache)

        for k, v in zip(k_per_tok[n_sys:], v_per_tok[n_sys:]):
            assembled.update_and_fetch(k, v)
        k_all_cached, v_all_cached = assembled.update_and_fetch(k_decode, v_decode)
        logits_cached = _attention(q_decode, k_all_cached, v_all_cached)
        mx.eval(logits_cached)

        assert float(mx.max(mx.abs(logits_cached - logits_fresh))) < ATOL

    def test_keep_gt_zero_wrapped(self):
        """RotatingKVCache with keep>0 and ring wrapped at checkpoint: assembled cache matches fresh."""
        from mlx_lm.models.cache import RotatingKVCache

        manager = _make_manager()

        keep = 2
        max_size = 4
        n_sys = 7  # n_sys > max_size, ring has wrapped; keep slots are protected
        n_user = 2
        n_heads, head_dim = 1, 64
        token_ids = list(range(n_sys + n_user))
        boundaries = [n_sys]

        mx.random.seed(11)
        k_per_tok = [mx.random.normal((1, n_heads, 1, head_dim)) for _ in token_ids]
        v_per_tok = [mx.random.normal((1, n_heads, 1, head_dim)) for _ in token_ids]
        q_decode = mx.random.normal((1, n_heads, 1, head_dim))
        k_decode = mx.random.normal((1, n_heads, 1, head_dim))
        v_decode = mx.random.normal((1, n_heads, 1, head_dim))
        mx.eval(*k_per_tok, *v_per_tok, q_decode, k_decode, v_decode)

        # --- Fresh path ---
        rk_fresh = RotatingKVCache(max_size=max_size, keep=keep)
        for k, v in zip(k_per_tok, v_per_tok):
            rk_fresh.update_and_fetch(k, v)
        k_all, v_all = rk_fresh.update_and_fetch(k_decode, v_decode)
        logits_fresh = _attention(q_decode, k_all, v_all)
        mx.eval(logits_fresh)

        # --- Populate cache ---
        req1 = _make_request("r1", token_ids, boundaries)
        assert manager.fetch(req1) is False

        rk_sys = RotatingKVCache(max_size=max_size, keep=keep)
        for k, v in zip(k_per_tok[:n_sys], v_per_tok[:n_sys]):
            rk_sys.update_and_fetch(k, v)
        manager.on_prefill_checkpoint(
            req1, total_tokens_prefilled=n_sys, extracted_cache=[rk_sys]
        )

        rk_full = RotatingKVCache(max_size=max_size, keep=keep)
        for k, v in zip(k_per_tok, v_per_tok):
            rk_full.update_and_fetch(k, v)
        req1.output_token_ids = [999]
        manager.store(req1, cache=[rk_full])

        # --- Cached path ---
        req2 = _make_request("r2", token_ids, boundaries)
        assert manager.fetch(req2) is True
        cs = req2._cache_state

        assembled = cs.cache[0]
        assert isinstance(assembled, RotatingKVCache)

        for k, v in zip(k_per_tok[n_sys:], v_per_tok[n_sys:]):
            assembled.update_and_fetch(k, v)
        k_all_cached, v_all_cached = assembled.update_and_fetch(k_decode, v_decode)
        logits_cached = _attention(q_decode, k_all_cached, v_all_cached)
        mx.eval(logits_cached)

        assert float(mx.max(mx.abs(logits_cached - logits_fresh))) < ATOL


class TestMultiLayerRoundtrip:
    def test_multi_layer_kvcache_attention_matches_fresh(self):
        """Multiple KVCache layers: all layers must be restored correctly.

        Uses two independent KV layers (simulating two transformer layers).
        Each layer has distinct random keys/values; the test checks that both
        layers produce the same decode attention as a fresh full prefill.
        """
        from mlx_lm.models.cache import KVCache

        manager = _make_manager()

        n_layers = 2
        n_sys, n_user, n_heads, head_dim = 3, 2, 2, 64
        token_ids = list(range(n_sys + n_user))
        boundaries = [n_sys]

        mx.random.seed(20)
        # Per-layer, per-token KVs
        k_layers = [
            [mx.random.normal((1, n_heads, 1, head_dim)) for _ in token_ids]
            for _ in range(n_layers)
        ]
        v_layers = [
            [mx.random.normal((1, n_heads, 1, head_dim)) for _ in token_ids]
            for _ in range(n_layers)
        ]
        q_decode = [mx.random.normal((1, n_heads, 1, head_dim)) for _ in range(n_layers)]
        k_decode = [mx.random.normal((1, n_heads, 1, head_dim)) for _ in range(n_layers)]
        v_decode = [mx.random.normal((1, n_heads, 1, head_dim)) for _ in range(n_layers)]
        mx.eval(*[t for layer in k_layers for t in layer])
        mx.eval(*[t for layer in v_layers for t in layer])
        mx.eval(*q_decode, *k_decode, *v_decode)

        # --- Fresh path: prefill all tokens for each layer, then decode ---
        logits_fresh = []
        for li in range(n_layers):
            kv = KVCache()
            for k, v in zip(k_layers[li], v_layers[li]):
                kv.update_and_fetch(k, v)
            k_all, v_all = kv.update_and_fetch(k_decode[li], v_decode[li])
            logits_fresh.append(_attention(q_decode[li], k_all, v_all))
        mx.eval(*logits_fresh)

        # --- Populate cache ---
        req1 = _make_request("r1", token_ids, boundaries)
        assert manager.fetch(req1) is False

        kv_sys_layers = []
        for li in range(n_layers):
            kv = KVCache()
            for k, v in zip(k_layers[li][:n_sys], v_layers[li][:n_sys]):
                kv.update_and_fetch(k, v)
            kv_sys_layers.append(kv)
        manager.on_prefill_checkpoint(
            req1, total_tokens_prefilled=n_sys, extracted_cache=kv_sys_layers
        )

        kv_full_layers = []
        for li in range(n_layers):
            kv = KVCache()
            for k, v in zip(k_layers[li], v_layers[li]):
                kv.update_and_fetch(k, v)
            kv_full_layers.append(kv)
        req1.output_token_ids = [999]
        manager.store(req1, cache=kv_full_layers)

        # --- Cached path ---
        req2 = _make_request("r2", token_ids, boundaries)
        assert manager.fetch(req2) is True
        cs = req2._cache_state

        assert cs.cached_tokens == n_sys
        assert len(cs.cache) == n_layers

        logits_cached = []
        for li in range(n_layers):
            assembled = cs.cache[li]
            raw_state = assembled.state
            k_cached = mx.dequantize(
                *raw_state[0], group_size=assembled.group_size, bits=assembled.bits
            )
            v_cached = mx.dequantize(
                *raw_state[1], group_size=assembled.group_size, bits=assembled.bits
            )
            k_remaining = mx.concatenate(k_layers[li][n_sys:], axis=-2)
            v_remaining = mx.concatenate(v_layers[li][n_sys:], axis=-2)
            k_all = mx.concatenate([k_cached, k_remaining, k_decode[li]], axis=-2)
            v_all = mx.concatenate([v_cached, v_remaining, v_decode[li]], axis=-2)
            logits_cached.append(_attention(q_decode[li], k_all, v_all))
        mx.eval(*logits_cached)

        for li in range(n_layers):
            assert float(mx.max(mx.abs(logits_cached[li] - logits_fresh[li]))) < ATOL, (
                f"Layer {li} attention mismatch"
            )


class TestFullCacheHitRoundtrip:
    def test_zero_remaining_tokens_kvcache(self):
        """Fetch where cached tokens == len(prompt_token_ids): remaining_tokens is empty.

        The assembled cache alone should reproduce the fresh decode attention.

        Setup: turn 1 stores sys+user1+asst1 in the trie.  Turn 2's prompt is
        sys+user1+asst1 (the assistant output from turn 1 appears verbatim in
        the turn-2 prompt, as happens in real chat history).  The trie match
        reaches the stored response node which covers all turn-2 prompt tokens,
        so cached_tokens == len(prompt_token_ids) and remaining_tokens == [].
        """
        from mlx_lm.models.cache import KVCache

        manager = _make_manager()

        # Turn 1: sys=[0,1], user1=[2,3,4], output=[5] (asst1)
        # Turn 2 prompt: sys+user1+asst1 = [0,1,2,3,4,5]  — no new user tokens
        n_sys, n_user1, n_heads, head_dim = 2, 3, 2, 64
        asst1_token = 5
        turn1_ids = list(range(n_sys + n_user1))          # [0,1,2,3,4]
        turn2_ids = turn1_ids + [asst1_token]              # [0,1,2,3,4,5]
        n_total = len(turn2_ids)
        boundaries_turn1 = [n_sys]                         # [2]
        boundaries_turn2 = [n_sys]                         # sys only; no extra boundary needed

        mx.random.seed(30)
        k_per_tok = [mx.random.normal((1, n_heads, 1, head_dim)) for _ in turn2_ids]
        v_per_tok = [mx.random.normal((1, n_heads, 1, head_dim)) for _ in turn2_ids]
        q_decode = mx.random.normal((1, n_heads, 1, head_dim))
        k_decode = mx.random.normal((1, n_heads, 1, head_dim))
        v_decode = mx.random.normal((1, n_heads, 1, head_dim))
        mx.eval(*k_per_tok, *v_per_tok, q_decode, k_decode, v_decode)

        # --- Fresh path: prefill all turn-2 tokens then decode ---
        kv_fresh = KVCache()
        for k, v in zip(k_per_tok, v_per_tok):
            kv_fresh.update_and_fetch(k, v)
        k_all, v_all = kv_fresh.update_and_fetch(k_decode, v_decode)
        logits_fresh = _attention(q_decode, k_all, v_all)
        mx.eval(logits_fresh)

        # --- Turn 1: populate cache ---
        req1 = _make_request("r1", turn1_ids, boundaries_turn1)
        assert manager.fetch(req1) is False

        kv_sys = KVCache()
        for k, v in zip(k_per_tok[:n_sys], v_per_tok[:n_sys]):
            kv_sys.update_and_fetch(k, v)
        manager.on_prefill_checkpoint(
            req1, total_tokens_prefilled=n_sys, extracted_cache=[kv_sys]
        )

        # kv_full must cover all tokens including the decoded asst1 output token,
        # since that KV is generated during turn-1 decode and must be stored.
        kv_full = KVCache()
        for k, v in zip(k_per_tok, v_per_tok):  # all n_total tokens
            kv_full.update_and_fetch(k, v)
        req1.output_token_ids = [asst1_token]
        manager.store(req1, cache=[kv_full])

        # --- Turn 2: fetch — cached node covers all turn-2 prompt tokens ---
        req2 = _make_request("r2", turn2_ids, boundaries_turn2)
        assert manager.fetch(req2) is True
        cs = req2._cache_state

        assert cs.cached_tokens == n_total, (
            f"expected cached_tokens={n_total}, got {cs.cached_tokens}; "
            "response node in trie should cover all turn-2 prompt tokens"
        )
        assert cs.remaining_tokens == []

        assembled = cs.cache[0]
        raw_state = assembled.state
        k_cached = mx.dequantize(
            *raw_state[0], group_size=assembled.group_size, bits=assembled.bits
        )
        v_cached = mx.dequantize(
            *raw_state[1], group_size=assembled.group_size, bits=assembled.bits
        )

        k_all_cached = mx.concatenate([k_cached, k_decode], axis=-2)
        v_all_cached = mx.concatenate([v_cached, v_decode], axis=-2)
        logits_cached = _attention(q_decode, k_all_cached, v_all_cached)
        mx.eval(logits_cached)

        assert float(mx.max(mx.abs(logits_cached - logits_fresh))) < ATOL


class TestRotatingCacheSecondWrapRoundtrip:
    def test_remaining_tokens_cause_second_wrap(self):
        """RotatingKVCache: appending remaining tokens after fetch causes the ring to wrap again.

        Layout: max_size=4, n_sys=6 (wrapped at checkpoint), n_user=4 (n_sys+n_user > 2*max_size).
        After fetch the assembled cache holds max_size tokens; appending n_user tokens wraps again.
        """
        from mlx_lm.models.cache import RotatingKVCache

        manager = _make_manager()

        max_size = 4
        n_sys = 6   # wraps once during checkpoint prefill
        n_user = 5  # n_sys + n_user = 11 > 2*max_size; appending causes another wrap
        n_heads, head_dim = 1, 64
        token_ids = list(range(n_sys + n_user))
        boundaries = [n_sys]

        mx.random.seed(40)
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

        # --- Populate cache ---
        req1 = _make_request("r1", token_ids, boundaries)
        assert manager.fetch(req1) is False

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

        # --- Cached path ---
        req2 = _make_request("r2", token_ids, boundaries)
        assert manager.fetch(req2) is True
        cs = req2._cache_state

        assert cs.cached_tokens == n_sys

        assembled = cs.cache[0]
        assert isinstance(assembled, RotatingKVCache)

        for k, v in zip(k_per_tok[n_sys:], v_per_tok[n_sys:]):
            assembled.update_and_fetch(k, v)
        k_all_cached, v_all_cached = assembled.update_and_fetch(k_decode, v_decode)
        logits_cached = _attention(q_decode, k_all_cached, v_all_cached)
        mx.eval(logits_cached)

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
        assert manager.fetch(req1) is False

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
        assert manager.fetch(req2) is True
        cs = req2._cache_state

        # Stored response node covers sys+user1+asst1 = n_cached positions
        assert cs.cached_tokens == n_cached
        assert cs.remaining_tokens == turn2_ids[n_cached:]

        assembled = cs.cache[0]
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

        assert float(mx.max(mx.abs(logits_cached - logits_fresh))) < ATOL

    def test_second_turn_rotating_cache_matches_fresh_decode(self):
        """Multi-turn with RotatingKVCache: turn-2 decode matches full fresh prefill.

        Conversation layout (token IDs = sequence positions):
          sys:   [0, 1, 2, 3]      (n_sys=4)
          user1: [4, 5]            (n_user1=2)
          asst1: [6, 7]            (n_asst1=2, stored as output_token_ids)
          user2: [8, 9]            (n_user2=2)
          decode: position 10

        Turn 1 stores RotatingKVCache covering sys+user1+asst1 (positions 0-7).
        Turn 2 fetch hits that stored state, prefills user2 into the assembled
        RotatingKVCache, then decodes — result must match fresh full prefill.
        """
        from mlx_lm.models.cache import RotatingKVCache

        manager = _make_manager()

        max_size = 16  # buffer does not wrap; focus is on multi-turn path
        n_sys, n_user1, n_asst1, n_user2 = 4, 2, 2, 2
        n_heads, head_dim = 2, 64
        n_cached = n_sys + n_user1 + n_asst1
        n_total = n_cached + n_user2

        turn1_prompt_ids = list(range(n_sys + n_user1))
        asst1_output_ids = list(range(n_sys + n_user1, n_cached))
        turn2_ids = list(range(n_total))

        boundaries_turn1 = [n_sys]
        boundaries_turn2 = [n_sys, n_cached]

        mx.random.seed(50)
        k_per_tok = [mx.random.normal((1, n_heads, 1, head_dim)) for _ in range(n_total)]
        v_per_tok = [mx.random.normal((1, n_heads, 1, head_dim)) for _ in range(n_total)]
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

        # --- Turn 1: miss → checkpoint at sys → store ---
        req1 = _make_request("r1", turn1_prompt_ids, boundaries_turn1)
        assert manager.fetch(req1) is False

        rk_sys = RotatingKVCache(max_size=max_size, keep=0)
        for k, v in zip(k_per_tok[:n_sys], v_per_tok[:n_sys]):
            rk_sys.update_and_fetch(k, v)
        manager.on_prefill_checkpoint(
            req1, total_tokens_prefilled=n_sys, extracted_cache=[rk_sys]
        )

        rk_full = RotatingKVCache(max_size=max_size, keep=0)
        for k, v in zip(k_per_tok[:n_cached], v_per_tok[:n_cached]):
            rk_full.update_and_fetch(k, v)
        req1.output_token_ids = asst1_output_ids
        manager.store(req1, cache=[rk_full])

        # --- Turn 2: fetch → assembled covers 0-7; prefill user2; decode ---
        req2 = _make_request("r2", turn2_ids, boundaries_turn2)
        assert manager.fetch(req2) is True
        cs = req2._cache_state

        assert cs.cached_tokens == n_cached
        assert cs.remaining_tokens == turn2_ids[n_cached:]

        assembled = cs.cache[0]
        assert isinstance(assembled, RotatingKVCache)

        for k, v in zip(k_per_tok[n_cached:], v_per_tok[n_cached:]):
            assembled.update_and_fetch(k, v)
        k_all_cached, v_all_cached = assembled.update_and_fetch(k_decode, v_decode)
        logits_cached = _attention(q_decode, k_all_cached, v_all_cached)
        mx.eval(logits_cached)

        assert float(mx.max(mx.abs(logits_cached - logits_fresh))) < ATOL
