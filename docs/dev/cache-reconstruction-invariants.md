# Cache Reconstruction Invariants

Pitfalls that have caused production bugs during reconstruction of live cache objects
from trie-stored `KVLayerSegment` data. Check this file before touching `_assemble`,
`BatchQuantizedKVCache`, or anything that feeds `update_and_fetch` for the first time.

---

## 1. `_assemble` must emit step-aligned `BatchQuantizedKVCache`

**Rule:** `_assemble` must call `BatchQuantizedKVCache.from_quantized_arrays()` and
pre-pad the buffer to the next `step=256` boundary.

**Why it matters:** `BatchQuantizedKVCache.update_and_fetch` enters the `expand_quant`
branch — a full `mx.concatenate` of the existing buffer with a fresh zero-block — whenever
`(prev + num_steps) > keys.packed.shape[2]`. A buffer allocated at exactly `n_tokens`
triggers this on the very first decode step. At 60k tokens × 7 full-attention layers the
transient peak is ~1.8 GB, which OOMs a Gemma 4 26B A4B server.

**Pre-padded allocation** ensures `keys.packed.shape[2] == next_multiple_of_256(n_tokens)`,
so the first decode step lands on the cheap in-place assignment branch.

**Correct pattern:**
```python
step = BatchQuantizedKVCache.step          # 256
padded_len = ((n_tokens + step - 1) // step) * step
pad = padded_len - n_tokens

def _pad_qa(qa):
    if pad == 0:
        return qa
    return QuantizedArray(*[
        mx.concatenate(
            [c, mx.zeros((*c.shape[:-2], pad, c.shape[-1]), dtype=c.dtype)],
            axis=-2,
        )
        for c in qa
    ])

cache = BatchQuantizedKVCache.from_quantized_arrays(
    keys=_pad_qa(layer.keys),
    values=_pad_qa(layer.values),
    n_tokens=n_tokens,
    group_size=group_size,
    bits=bits,
)
```

**ADR reference:** ADR-0005 specified `from_quantized_arrays` as the construction path;
emitting plain `QuantizedKVCache` deviates from this contract and will cause the spike.

**Regression test:** `tests/test_cache_hit_oom_repro.py` — measures Metal peak memory
during the first decode step at production scale (60k tokens, 7 layers).

---

## 2. `BatchQuantizedKVCache.merge` fast-path requires a plain `QuantizedKVCache` input

**Rule:** The single-cache fast path in `BatchQuantizedKVCache.merge([c])` must early-return
if `c` is already a `BatchQuantizedKVCache`.

**Why it matters:** `BatchQuantizedKVCache` stores `offset` and `_idx` differently from
`QuantizedKVCache`:
- `QuantizedKVCache.offset` — plain `int`
- `BatchQuantizedKVCache.offset` — `mx.array([n])`

The old fast path did `result._idx = c.offset` and `result.offset = mx.array([c.offset])`.
When `c` is a `BatchQuantizedKVCache`, this nests the array:
`result.offset = mx.array([mx.array([n])])`, which is a rank-2 scalar. All subsequent
`update_and_fetch` calls use `_idx` as a slice index — a corrupted `_idx` causes the cache
to return 0 tokens silently.

**Correct guard (already in place):**
```python
if len(caches) == 1:
    c = caches[0]
    if isinstance(c, cls):   # ← this guard
        return c
    ...
```

**How to detect regression:** `test_cache_parity` in `tests/test_prefix_cache_parity.py`
will time out or produce empty output (0 decoded tokens) if this guard is removed.

---

## 3. ADR-0005 type contract: always `BatchQuantizedKVCache`, never `QuantizedKVCache`

**Rule:** Cache reconstruction must never emit plain `QuantizedKVCache` for standard
(non-rotating) KV layers. ADR-0005 mandates `BatchQuantizedKVCache.from_quantized_arrays`.

**Why it matters:** `QuantizedKVCache` is a single-sequence, non-batched type. It does not
have step-aligned allocation, does not support `left_padding`, and is incompatible with
`BatchQuantizedKVCache.merge`. Returning it from `_assemble` is always a bug.

**How to verify:** Integration and unit tests assert `isinstance(cache, BatchQuantizedKVCache)`
— not just shape or `_idx`. If a test only checks `.offset` or `.shape`, it cannot detect
a type-level regression.
