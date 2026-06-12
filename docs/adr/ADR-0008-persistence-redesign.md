# ADR-0008: SSD persistence redesign (CacheDiskStore at the manager seam)

**Status:** Accepted
**Date:** 2026-06-11
**Companion spec:** `docs/superpowers/specs/2026-06-11-ssd-persistence-redesign-design.md`
**Companion plan:** `docs/superpowers/plans/2026-06-11-ssd-persistence-redesign.md`

## Context

The pre-existing SSD persistence layer was buried inside `TurnPrefixCache`
(`_spill_to_ssd`, `_promote_from_ssd`, `save`, `load`) and duplicated by
`MemoryAwarePrefixCache.save_to_disk` / `load_from_disk`. The two paths
never converged on a single disk format; both were buggy and unused. The
NumPy-backed `safetensors` variant could not represent `bfloat16` and
forced lossy conversions on spill. ADR-0003 declined an
`SSDOffloadedCache` decorator because two cache shapes had to be
supported simultaneously; with `MemoryAwarePrefixCache` deprecated in the
ADR-0003 addendum, only one cache shape remains.

## Decision

- Introduce a `CacheDiskStore` protocol and a single shipping
  `FilesystemCacheDiskStore` implementation.
- Disk concerns move from `TurnPrefixCache` up to `TurnCacheManager`.
  The trie becomes pure in-memory.
- Spill and promote share one disk format with save/load. Disk inherits
  the per-layer quant policy via self-describing `metadata["bits"]`.
- A single `_DISK_FORMAT_VERSION = 1`. No migration code — mismatch is
  fatal with instructions.

See the companion spec for full details on the protocol, disk format,
LRU ordering, error handling, and config surface.

## Consequences

- `vllm_mlx/ssd_cache.py` and `vllm_mlx/memory_cache.py` are deleted.
- `TurnPrefixCache` shrinks substantially (~−350 LOC); `TurnCacheManager`
  grows (~+250 LOC); a new `cache_disk_store.py` module appears (~400 LOC).
- The `BatchedEngine.save_cache_to_disk` / `load_cache_from_disk` API
  changes from `(cache_dir: str)` to no-arg, with the disk location
  configured via `SchedulerConfig.kv_cache_disk_dir`.
- New CLI flags: `--kv-cache-disk-dir`, `--kv-cache-disk-max-bytes`,
  `--kv-cache-load-on-startup`, `--kv-cache-save-on-shutdown`. Legacy
  `--ssd-*` flags are migration-errored.

## Relationship to prior ADRs

- **ADR-0003** still constrains us against a cache orchestrator; the
  redesign respects that by keeping responsibilities at the
  `CacheManager` seam, not in a new wrapper layer.
- **ADR-0004** keeps `step()` synchronous; promote remains sync on access.
- **ADR-0005** is unaffected — segments still ride the same group-quantized
  format; disk is a byte-level pass-through.
- **ADR-0007** (per-layer-type quant) supplies the self-describing
  `metadata["bits"]` that makes the disk format inherit the quant policy.
