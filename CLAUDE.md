# Claude guidance for vllm-mlx

## Domain glossary and architecture

Read `CONTEXT.md` at the start of any session that touches the cache layer, scheduler,
or spill/promote pipeline. It defines the canonical terms used throughout the codebase
(`KVLayerSegment`, `merge_strategy`, `hot path`, `pinning`, etc.) and will prevent
misreading variable names.

## Architectural decisions

`docs/adr/` contains accepted ADRs. Before changing cache storage format, reconstruction
logic, or the `_segment`/`_assemble` pipeline, check the relevant ADR:

- **ADR-0003** — why there is no cache orchestrator (responsibilities stay in Scheduler)
- **ADR-0004** — why the decode step stays synchronous
- **ADR-0005** — why KV segments are stored in mlx-lm native group-quantized format and
  why `_assemble` must emit `BatchQuantizedKVCache.from_quantized_arrays`, not `QuantizedKVCache`
- **ADR-0008** — why model-specific behavior lives behind one `Architecture` class per
  `model_type` instead of scattered monkey-patches and `model_type` switches

## Developer notes (`docs/dev/`)

Operational notes from debugging sessions — read these before touching the relevant code:

| File | Read when… |
|------|------------|
| `docs/dev/cache-reconstruction-invariants.md` | Touching `_assemble`, `BatchQuantizedKVCache`, or anything that feeds `update_and_fetch` for the first time on a reconstructed cache |
| `docs/dev/mlx-memory-profiling.md` | Writing a test that measures Metal peak memory, diagnosing OOM reports, or verifying that a reconstruction fix doesn't introduce allocation spikes |

## Tests

- `tests/test_cache_hit_oom_repro.py` — Metal peak-memory regression test at production
  scale (60k tokens, Gemma 4 26B A4B dimensions). Skipped on non-Metal hardware.
- `tests/test_cache_translator.py` — unit tests for `_segment`/`_assemble` pipeline and
  `KVLayerSegment.concat`.
- `tests/test_turn_prefix_cache_integration.py` — end-to-end trie round-trip tests.

Run `pytest tests/` before any cache-layer commit.
