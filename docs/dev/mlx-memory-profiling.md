# MLX Memory Profiling

How to measure Metal memory allocation during cache operations — useful when diagnosing
OOM reports or verifying that a reconstruction fix doesn't spike peak memory.

---

## API

```python
import mlx.core as mx

mx.eval(*arrays)           # materialize lazy graphs before measuring
resident = mx.get_active_memory()

mx.reset_peak_memory()     # zero the high-water mark
pre = mx.get_active_memory()

# ... operation under test ...

mx.eval(*output_arrays)
peak_delta = mx.get_peak_memory() - pre
```

`get_active_memory()` — bytes currently resident in Metal.  
`get_peak_memory()` — high-water mark since last `reset_peak_memory()`.  
`reset_peak_memory()` — zeroes the high-water mark without freeing memory.

---

## Pattern: first-decode-step spike test

The dangerous window is `update_and_fetch` on a freshly reconstructed cache.
If `keys.packed.shape[2] == n_tokens` (not step-aligned), the first call triggers
`expand_quant` — a `mx.concatenate` that doubles peak allocation transiently.

```python
mx.eval(*cache_arrays)
mx.reset_peak_memory()
pre = mx.get_active_memory()

k = mx.zeros((1, N_KV_HEADS, 1, HEAD_DIM), dtype=mx.bfloat16)
v = mx.zeros((1, N_KV_HEADS, 1, HEAD_DIM), dtype=mx.bfloat16)
out_k, out_v = cache.update_and_fetch(k, v)
mx.eval(*flatten_arrays(out_k), *flatten_arrays(out_v))

spike = mx.get_peak_memory() - pre
```

A step-aligned cache should spike < 1% of cache size.
An unaligned cache spikes ~100% (full buffer doubling).

---

## Realistic test dimensions for Gemma 4 26B A4B

| Parameter | Value |
|-----------|-------|
| N_TOKENS | 60 000 (not step-aligned: 60000 % 256 = 96) |
| N_KV_HEADS | 4 |
| HEAD_DIM | 512 (global full-attn layers) |
| BITS | 8 |
| GROUP_SIZE | 64 |
| N_FULL_LAYERS | 7 (1 full-attn per 5 layers in 35-layer model) |

These values are encoded in `tests/test_cache_hit_oom_repro.py`.
Use them as the reference for "production scale" when writing new spike tests.

---

## Handling `QuantizedArray` in flatten helpers

`BatchQuantizedKVCache` returns `QuantizedArray(packed, scales, biases)` namedtuples,
not bare `mx.array`. A flatten helper must unpack both:

```python
def _flatten_arrays(x):
    if isinstance(x, mx.array):
        yield x
    elif hasattr(x, "packed"):           # QuantizedArray namedtuple
        yield x.packed; yield x.scales; yield x.biases
    else:
        for v in x:
            yield from _flatten_arrays(v)
```
