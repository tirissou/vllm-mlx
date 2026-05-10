# TurnPrefixCache: Real KV State Wiring

**Date**: 2026-05-10  
**Status**: Approved  
**Branch**: fix-reasoning_content-input

## Problem

TurnPrefixCache was added as a trie-based conversation-turn cache, but every `turn_cache.insert()` call passes `recurrent_state=None`. The trie correctly matches system prompt segments across sessions, but `find_checkpoint_ancestor` always returns `None` (no state), so every request re-prefills from scratch. The cache provides zero computation savings.

Additionally, `memory_aware_cache` and `turn_cache` both initialize simultaneously — `memory_aware_cache` wins every fetch/store because it is checked first in the `elif` chain, making `turn_cache` dead code.

## Goals

1. Fix mutual exclusion: `memory_aware_cache` must not initialize when `use_turn_cache=True`.
2. Store real KV/recurrent state in `TurnNode.recurrent_state` for both system and user segments.
3. Restore state at fetch so BatchGenerator skips already-cached tokens.
4. Persist state across server restarts via `save()`/`load()`.
5. Support transformer (KVCache), SSM (MambaCache/ArraysCache), and hybrid models.
6. (Deferred) Quantize stored state tensors to reduce memory footprint.

## Non-Goals

- Replacing the `kv_arrays` quantization path in TurnNode (unused, left as-is).
- Changing trie structure, match logic, or eviction policy.
- Enabling `memory_aware_cache` and `turn_cache` simultaneously.

## Architecture Decision: `recurrent_state` format

Store `_extract_cache_states` output — `List[Dict{state, meta_state, class_name, class_ref}]` — in `TurnNode.recurrent_state`. This is the natural bridge between what `mid_prefill_save` produces and what `_reconstruct_cache_from_states` consumes.

**Rejected alternatives**:
- Store live cache objects: save/load must re-extract from objects; incompatible with `_QuantizedCacheWrapper`; quantization harder to add later.
- New `kv_state` field on TurnNode: two nearly-identical fields, all existing code touching `recurrent_state` needs updating.

## Data Flow

### Store path

```
generation ends
    │
    ├─ mid_prefill fired at prefix_boundary (during prefill)
    │       └─ _extract_cache_states(prompt_cache)
    │               └─ request._sys_prompt_state = List[Dict]
    │
    └─ store-side code (scheduler.py ~line 2521)
            ├─ segment[0] (system): insert(recurrent_state=request._sys_prompt_state)
            └─ segment[1] (user):   insert(recurrent_state=_extract_cache_states(_extracted_cache))
```

### Fetch path

```
new request arrives
    └─ turn_cache.match(segments)
            ├─ MISS → full prefill (unchanged)
            └─ HIT → find_checkpoint_ancestor(path)
                    └─ ancestor.recurrent_state = List[Dict]
                            └─ _reconstruct_cache_from_states(recurrent_state)
                                    └─ request.prompt_cache = List[KVCache/MambaCache]
                                            └─ BatchGenerator skips cached tokens
```

### Cross-session system prompt hit (primary bug fix)

```
Session 1: [sys:500tok] + [user:20tok] → full prefill
    → mid_prefill fires at tok 500 → sys node gets List[Dict] state
    → user node gets List[Dict] state (full prompt end-state)

Session 2: [sys:500tok] + [user:15tok] → match()
    → sys node HIT, user node MISS
    → find_checkpoint_ancestor → sys node (is_permanent_checkpoint=True, has state)
    → reconstruct → request.prompt_cache = sys KV state
    → BatchGenerator processes only 15 user tokens
```

## Component Changes

### `scheduler.py`

**Change 0** — `__init__` line 1199: gate out `memory_aware_cache` initialization when `use_turn_cache=True`.
```python
elif self.config.use_memory_aware_cache and not self.config.use_turn_cache:
```

**Change 1+2** — `_make_mid_prefill_save_callback`: add turn_cache branch. When `at_prefix_boundary` and `self.turn_cache is not None`, call `_extract_cache_states(prompt_cache)` and store in `request._sys_prompt_state`. Existing `memory_aware_cache` branch unchanged (now mutually exclusive after Change 0). Install callback when either cache is active (not just memory_aware_cache).

**Change 3** — store-side loop (~line 2521): pass real state per segment:
- System segment (i==0, matched_depth==0): `recurrent_state = request._sys_prompt_state`
- User segment (last): `recurrent_state = _extract_cache_states(request._extracted_cache)`
- Middle segments (multi-turn beyond two): `recurrent_state = None` (future work)

**Change 4** — fetch side (~line 2017): reconstruct before use:
```python
raw_state = ancestor.recurrent_state if ancestor else None
request.prompt_cache = _reconstruct_cache_from_states(raw_state) if raw_state else None
```

**Change 7** — eval loop: after existing eval of `_extracted_cache`, add eval of tensors in `request._sys_prompt_state` (if set) to prevent lazy MLX evaluation after source cache objects are released.

### `turn_prefix_cache.py`

**Change 5** — `save()`: detect dict-format `recurrent_state` via `isinstance(state[0], dict)`. Serialize as:
- `ext_{i}_state_{j}` — tensor members of the `state` tuple, per layer
- `ext_{i}_meta_{j}` — meta_state strings as numpy bytes, per layer  
- `ext_{i}_class` — class_name as numpy bytes

Use `ext_` prefix to distinguish from existing SSM `r_` format (backward compatibility).

**Change 6** — `load()`: detect `ext_` keys. Reconstruct `List[Dict]`: look up `class_ref` via `getattr(importlib.import_module("mlx_lm.models.cache"), class_name, None)`. Restore `state` as tuple of `mx.array`, `meta_state` as tuple of strings.

**Change 8** — `_node_data_bytes`: for dict-format `recurrent_state`, sum `arr.nbytes` for each tensor in each dict's `state` tuple, so LRU eviction budget is accurate.

## Serialization Format

Two coexisting formats in safetensors files:

| Key pattern | Format | Meaning |
|-------------|--------|---------|
| `r_{k}_{m}` | existing | SSM raw tensor format (unchanged) |
| `ext_{i}_state_{j}` | new | layer i, state tuple member j |
| `ext_{i}_meta_{j}` | new | layer i, meta_state string j (as bytes) |
| `ext_{i}_class` | new | layer i, class_name (as bytes) |

Detection: if any `ext_` key exists → dict format. If only `r_` keys → legacy SSM format.

## Quantization (Deferred — Change 10)

`TurnPrefixCacheConfig.kv_dtype = "int8"` already exists but currently applies only to `kv_arrays` (unused). After the main implementation:

- `insert()`: if `kv_dtype == "int8"` and `recurrent_state` is dict-format, quantize each `state` tensor pair via `mx.quantize`. Store quantization scales alongside state tensors.
- Fetch side: dequantize before `_reconstruct_cache_from_states`.
- Test: assert `node._node_data_bytes` after quantized insert < 60% of unquantized equivalent for same input.

Memory motivation: a 36-layer model with 1000-token system prompt stores ~147MB unquantized (bfloat16). int8 halves this; int4 quarters it.

## Notes on Quantization Non-interaction

`_QuantizedCacheWrapper` (from `memory_aware_cache`) is never present in the turn_cache pipeline. `request._extracted_cache` in standard (non-block-aware) mode contains raw `BatchKVCache`/`KVCache` objects. `_extract_cache_states` correctly handles both. No special casing needed.

## Constraint: TurnPrefixCache Requires Chunked Prefill

`mid_prefill_save` is wired into the chunked prefill path (`_install_chunked_prefill`). When `prefix_boundary > 0`, the batch generator explicitly uses it as the first chunk boundary (line ~585: `n_to_process = min(_pb or chunk_size, remaining)`), so mid_prefill fires at the system/user boundary regardless of whether the system prompt is shorter than the normal chunk size.

However, if chunked prefill is completely disabled (`chunked_prefill_tokens=0`, the default), the whole prompt is processed in one pass and mid_prefill never fires — system segment state is never captured.

**Enforcement**: `Scheduler.__init__` raises `ValueError` if `use_turn_cache=True` and `chunked_prefill_tokens=0`:

```python
if self.config.use_turn_cache and self.config.chunked_prefill_tokens == 0:
    raise ValueError(
        "TurnPrefixCache requires --chunked-prefill-tokens to be set. "
        "Set --chunked-prefill-tokens 8192 or higher."
    )
```

This is **Change 0b** (added to the execution order after Change 0).

## Execution Order

0 → 0b → 1+2 → 3 → 4 → 5 → 6 → 7 → 8 → tests → (deferred: 10)

## Tests

**Cross-session HIT** (integration): construct two requests with identical system prefix tokens but different user tokens. After first request completes, second request must show `[turn_cache] HIT` log with `cached_tokens == len(sys_tokens)`.

**Save/load round-trip** (unit): populate a TurnNode with dict-format `recurrent_state`, call `save()`, clear the trie, call `load()`, assert `_reconstruct_cache_from_states(node.recurrent_state)` returns valid cache objects with correct offset.

**Memory accounting** (unit): assert `_node_data_bytes` returns a value consistent with tensor sizes in dict-format `recurrent_state`.

**Config validation** (unit): assert `Scheduler.__init__` raises `ValueError` when `use_turn_cache=True` and `chunked_prefill_tokens=0`.

**(Deferred) Quantization memory reduction**: assert quantized node uses <60% memory of unquantized equivalent.
