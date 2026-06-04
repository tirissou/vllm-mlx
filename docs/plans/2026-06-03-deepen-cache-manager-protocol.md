# Deepen CacheManager Protocol — Scheduler Cache Decoupling (Corrected)

> **REQUIRED SUB-SKILL:** Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Make the Scheduler only interact with caching through the `CacheManager` protocol, removing all direct references to concrete cache backends.

**Architecture:**
- Add `validate()`, `extract_cache()`, `save()`, `load()`, `close()` methods to the `CacheManager` protocol with no-op defaults.
- Implement these on `TurnCacheManager` (the only functional adapter). `save()` and `load()` are updated (already exist, add error handling); `validate()`, `extract_cache()`, `close()` are genuinely new.
- Remove deprecated backends (`PrefixCacheManager`, `PagedCacheManager`, `MemoryAwarePrefixCache`) and their test files.
- Refactor the Scheduler to use only `self._prefix_cache` (a `CacheManager` reference) for all cache operations.
- Remove orphaned free functions `extract_cache_states()` and `validate_cache()` from `kv_cache.py` (zero callers after refactor).

**Tech Stack:** Python, pytest, MLX.

**Constraints (from ADRs):**
- ADR-0003: No `CacheOrchestrator`. Deepen adapters instead.
- ADR-0004: `step()` must remain synchronous.
- ADR-0005: Group-quantized trie storage format (already implemented).

---

### Task 0: Add new methods to CacheManager protocol

**Files:**
- Modify: `vllm_mlx/prefix_cache_adapters.py`

**Step 1: Add imports**

Add these imports at the top of `prefix_cache_adapters.py` (after existing imports, before class definitions):

```python
from .kv_cache import validate_cache, extract_cache_states
```

**Step 2: Add new methods to CacheManager base class**

Add these methods after the existing no-op defaults (after `update_n_minus_one`), before the `TurnCacheManager` class:

```python
    def validate(self, cache: list) -> bool:
        """Validate cache state. Returns True if valid and usable."""
        return True

    def extract_cache(self, raw_cache: list) -> list | None:
        """Extract cache state from raw cache objects.

        Returns list of layer state dicts, or None on failure.
        Called during cleanup to prepare cache for storage.
        """
        return None

    def save(self, cache_dir: str) -> bool:
        """Persist cache to disk. Returns True on success."""
        return False

    def load(self, cache_dir: str) -> int:
        """Load cache from disk. Returns entries loaded."""
        return 0

    def close(self) -> None:
        """Cleanup resources (SSD threads, file handles, etc.)."""
        pass
```

**Step 3: Run test to verify protocol still works**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
python -c "from vllm_mlx.prefix_cache_adapters import CacheManager; print('OK')"
```
Expected: `OK` (no import errors)

**Step 4: Run existing adapter tests**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
python -m pytest tests/test_prefix_cache_adapters.py -v --tb=short
```
Expected: All existing tests pass (new methods have no-op defaults).

**Step 5: Commit**

```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
git add vllm_mlx/prefix_cache_adapters.py
git commit -m "feat: add validate, extract_cache, save, load, close to CacheManager protocol"
```

---

### Task 1: Implement new methods on TurnCacheManager

**Files:**
- Modify: `vllm_mlx/prefix_cache_adapters.py`

> **Note:** `save()` and `load()` already exist on `TurnCacheManager`. This task updates them (adds error handling) and adds only `validate()`, `extract_cache()`, `close()` (genuinely new).

**Step 1: Add tests for new methods**

In `tests/test_prefix_cache_adapters.py`, add these tests after the existing TurnCacheManager tests:

```python
# ── validate() ───────────────────────────────────────────────────────────────

def test_turn_cache_manager_validate_valid_cache():
    """validate() returns True for valid cache (list of layers with keys/values)."""
    inner = MagicMock()
    inner.root = MagicMock(n_tokens=0)
    adapter = TurnCacheManager(inner)
    valid_cache = [MagicMock(keys=MagicMock(shape=(1, 8, 4, 64)), values=MagicMock(shape=(1, 8, 4, 64)))]
    assert adapter.validate(valid_cache) is True


def test_turn_cache_manager_validate_none_cache():
    """validate() returns False for None cache."""
    inner = MagicMock()
    adapter = TurnCacheManager(inner)
    assert adapter.validate(None) is False


def test_turn_cache_manager_validate_empty_cache():
    """validate() returns False for empty list cache."""
    inner = MagicMock()
    adapter = TurnCacheManager(inner)
    assert adapter.validate([]) is False


def test_turn_cache_manager_validate_none_layer():
    """validate() returns False if any layer is None."""
    inner = MagicMock()
    adapter = TurnCacheManager(inner)
    assert adapter.validate([None]) is False


# ── extract_cache() ──────────────────────────────────────────────────────────

def test_turn_cache_manager_extract_cache_valid():
    """extract_cache() forwards to extract_cache_states."""
    inner = MagicMock()
    adapter = TurnCacheManager(inner)
    raw_cache = [MagicMock(state=(MagicMock(), MagicMock()), meta_state=())]
    result = adapter.extract_cache(raw_cache)
    # Should return list of dicts or None
    assert result is None or isinstance(result, list)


def test_turn_cache_manager_extract_cache_empty():
    """extract_cache() returns None for empty list."""
    inner = MagicMock()
    adapter = TurnCacheManager(inner)
    assert adapter.extract_cache([]) is None


# ── save() / load() (updated with error handling) ────────────────────────────

def test_turn_cache_manager_save_forwards_to_inner():
    """save() forwards to inner.save()."""
    inner = MagicMock()
    inner.save.return_value = True
    adapter = TurnCacheManager(inner)
    result = adapter.save("/tmp/cache")
    assert result is True
    inner.save.assert_called_once_with("/tmp/cache")


def test_turn_cache_manager_save_fails():
    """save() returns False when inner.save() raises."""
    inner = MagicMock()
    inner.save.side_effect = OSError("disk full")
    adapter = TurnCacheManager(inner)
    result = adapter.save("/tmp/cache")
    assert result is False


def test_turn_cache_manager_load_forwards_to_inner():
    """load() forwards to inner.load()."""
    inner = MagicMock()
    inner.load.return_value = 42
    adapter = TurnCacheManager(inner)
    result = adapter.load("/tmp/cache")
    assert result == 42


def test_turn_cache_manager_load_fails():
    """load() returns 0 when inner.load() raises."""
    inner = MagicMock()
    inner.load.side_effect = OSError("disk full")
    adapter = TurnCacheManager(inner)
    result = adapter.load("/tmp/cache")
    assert result == 0


# ── close() ──────────────────────────────────────────────────────────────────

def test_turn_cache_manager_close_is_noop():
    """close() does nothing (TurnPrefixCache has no external resources)."""
    inner = MagicMock()
    adapter = TurnCacheManager(inner)
    adapter.close()  # must not raise
    inner.close.assert_not_called()
```

**Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
python -m pytest tests/test_prefix_cache_adapters.py::test_turn_cache_manager_validate_valid_cache -v
```
Expected: `FAIL` — `TurnCacheManager` doesn't have `validate` method (uses base class default which always returns True).

**Step 3: Implement methods on TurnCacheManager**

Find the existing `save()` and `load()` methods at the end of `TurnCacheManager` (after `on_prefill_checkpoint`). Replace them with error-handling versions, and add `validate()`, `extract_cache()`, `close()` after them:

```python
    # ── Updated: save() / load() with error handling ──────────────────────────

    def save(self, cache_dir: str) -> bool:
        try:
            return self._inner.save(cache_dir)
        except Exception:
            return False

    def load(self, cache_dir: str) -> int:
        try:
            self._inner.load(cache_dir)
            return 0
        except Exception:
            return 0

    # ── New: validate, extract_cache, close ──────────────────────────────────

    def validate(self, cache: list) -> bool:
        return validate_cache(cache)

    def extract_cache(self, raw_cache: list) -> list | None:
        return extract_cache_states(raw_cache)

    def close(self) -> None:
        pass
```

**Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
python -m pytest tests/test_prefix_cache_adapters.py -v --tb=short
```
Expected: All tests pass.

**Step 5: Commit**

```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
git add vllm_mlx/prefix_cache_adapters.py tests/test_prefix_cache_adapters.py
git commit -m "feat: implement validate, extract_cache, close on TurnCacheManager; add error handling to save/load"
```

---

### Task 2: Remove deprecated backends from _build_prefix_cache

**Files:**
- Modify: `vllm_mlx/scheduler.py`

**Step 1: Write a test that verifies simplified _build_prefix_cache**

In `tests/test_prefix_cache_adapters.py`, update the existing `TestBuildPrefixCache` test. First, let me check what the test expects:

The existing test checks `bundle.adapter`, `bundle.turn_cache`, `bundle.memory_aware_cache`, `bundle.prefix_cache`. After our changes, only `adapter` and `turn_cache` will exist.

Run the existing test to see what fails:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
python -m pytest tests/test_prefix_cache_adapters.py::TestBuildPrefixCache -v --tb=short
```
Expected: May pass or fail depending on current state. We'll update it.

**Step 2: Update _PrefixCacheBundle**

Replace the existing `_PrefixCacheBundle` dataclass:

```python
@dataclass
class _PrefixCacheBundle:
    """All prefix-cache objects produced by _build_prefix_cache."""

    adapter: "CacheManager | None" = None
    turn_cache: "TurnPrefixCache | None" = None
```

**Step 3: Update _build_prefix_cache**

Replace the entire `_build_prefix_cache` function:

```python
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
        bundle.adapter = TurnCacheManager(
            turn_cache,
            kv_bits=config.kv_cache_quantization_bits,
            kv_group_size=config.kv_cache_quantization_group_size,
        )
        logger.info(
            f"TurnPrefixCache enabled: stride={config.turn_cache_stride} "
            f"memory={config.turn_cache_memory_gb}GB"
        )
    else:
        logger.info("Prefix cache disabled (use_turn_cache=False)")

    return bundle
```

**Step 4: Update the test**

Update `TestBuildPrefixCache::test_turn_cache_config_returns_turn_cache_adapter` in `tests/test_prefix_cache_adapters.py`:

```python
    def test_turn_cache_config_returns_turn_cache_adapter(self):
        from unittest.mock import MagicMock, patch
        from vllm_mlx.scheduler import _build_prefix_cache

        mock_tc = MagicMock()
        with patch("vllm_mlx.turn_prefix_cache.TurnPrefixCache", return_value=mock_tc):
            bundle = _build_prefix_cache(
                self._config(use_turn_cache=True), model=object()
            )

        assert isinstance(bundle.adapter, TurnCacheManager)
        assert bundle.turn_cache is mock_tc
        # Deprecated fields removed
        assert not hasattr(bundle, 'memory_aware_cache')
        assert not hasattr(bundle, 'prefix_cache')
```

**Step 5: Run tests**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
python -m pytest tests/test_prefix_cache_adapters.py::TestBuildPrefixCache -v --tb=short
```
Expected: PASS.

**Step 6: Commit**

```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
git add vllm_mlx/scheduler.py tests/test_prefix_cache_adapters.py
git commit -m "refactor: remove deprecated backends from _build_prefix_cache, keep only TurnPrefixCache"
```

---

### Task 3: Simplify _init_cache_bundle

**Files:**
- Modify: `vllm_mlx/scheduler.py`

**Step 1: Update _init_cache_bundle**

Replace the entire `_init_cache_bundle` method:

```python
def _init_cache_bundle(self) -> None:
    """Initialize cache backend attributes from SchedulerConfig.

    Called from __init__. The Scheduler only knows about CacheManager
    through self._prefix_cache.
    """
    if self.config.enable_prefix_cache:
        _bundle = _build_prefix_cache(self.config, self.model)
        self._prefix_cache = _bundle.adapter
        self.turn_cache = _bundle.turn_cache
```

**Step 2: Run scheduler cache fetch tests**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
python -m pytest tests/test_scheduler_cache_fetch.py -v --tb=short
```
Expected: Some tests may fail because they reference deprecated backends. We'll fix those in Task 4.

**Step 3: Commit**

```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
git add vllm_mlx/scheduler.py
git commit -m "refactor: simplify _init_cache_bundle to only set _prefix_cache and turn_cache"
```

---

### Task 4: Remove deprecated backend references from Scheduler

**Files:**
- Modify: `vllm_mlx/scheduler.py`

**Step 1: Remove deprecated imports**

Remove these imports from the top of `scheduler.py`:

```diff
- from .memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
- from .paged_cache import PagedCacheManager
- from .ssd_cache import SSDCacheConfig, SSDCacheTier
- from .prefix_cache import BlockAwarePrefixCache, PrefixCacheManager
```

**Step 2: Remove deprecated instance attributes**

In `__init__`, remove these lines from the cache-related section:
```diff
- self.prefix_cache: Optional[PrefixCacheManager] = None
- self.paged_cache_manager: Optional[PagedCacheManager] = None
- self.block_aware_cache: Optional[BlockAwarePrefixCache] = None
- self._ssd_offloaded_cache = None
```

In `_init_cache_bundle`, remove these lines:
```diff
- self.memory_aware_cache: Optional[MemoryAwarePrefixCache] = None
- self._ssd_tier: Optional[SSDCacheTier] = None
```

**Step 3: Run tests**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
python -m pytest tests/test_scheduler_cache_fetch.py -v --tb=short
```
Expected: Some tests may reference deprecated backends. We'll handle those.

**Step 4: Commit**

```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
git add vllm_mlx/scheduler.py
git commit -m "refactor: remove deprecated backend imports and instance attributes from Scheduler"
```

---

### Task 5: Replace direct validate_cache() call in _schedule_waiting

**Files:**
- Modify: `vllm_mlx/scheduler.py`

**Step 1: Update the validation branch in _schedule_waiting**

Find the validation block in `_schedule_waiting` (around line ~1350). Replace:

```python
# Validate cache before using it
if cache_to_use is not None and not validate_cache(cache_to_use):
```

With:

```python
# Validate cache before using it
if cache_to_use is not None and self._prefix_cache is not None:
    if not self._prefix_cache.validate(cache_to_use):
```

**Step 2: Run scheduler cache fetch tests**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
python -m pytest tests/test_scheduler_cache_fetch.py -v --tb=short
```
Expected: Tests that mock `_prefix_cache.validate()` should pass. Tests that patch `vllm_mlx.scheduler.validate_cache` need updating.

**Step 3: Update test patches**

In `tests/test_scheduler_cache_fetch.py`, find all occurrences of:
```python
with patch("vllm_mlx.scheduler.validate_cache", return_value=True):
```
Replace with:
```python
with patch("vllm_mlx.scheduler.validate_cache", return_value=True):
    # Keep the patch for backward compatibility during transition
```
Actually, since we're removing the direct call, we need to remove these patches. The mock on `_prefix_cache` will handle validation.

**Step 4: Commit**

```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
git add vllm_mlx/scheduler.py tests/test_scheduler_cache_fetch.py
git commit -m "refactor: replace direct validate_cache() call with CacheManager.validate()"
```

---

### Task 6: Replace direct extract_cache_states() calls

**Files:**
- Modify: `vllm_mlx/scheduler.py`

**Step 1: Update _process_batch_responses**

Find the cache extraction block in `_process_batch_responses` (around line ~1500). Replace:

```python
if self.block_aware_cache is not None:
    extracted_cache = extract_cache_states(raw_cache)
    if extracted_cache:
        request._cache_state.decoded_cache = extracted_cache
else:
    request._cache_state.decoded_cache = raw_cache
```

With:

```python
if self._prefix_cache is not None and raw_cache:
    extracted = self._prefix_cache.extract_cache(raw_cache)
    if extracted:
        request._cache_state.decoded_cache = extracted
    else:
        request._cache_state.decoded_cache = raw_cache
else:
    request._cache_state.decoded_cache = raw_cache
```

Find the second extraction block (around line ~1530):
```python
if request._cache_state.decoded_cache and not isinstance(
    request._cache_state.decoded_cache[0], dict
):
    request._cache_state.decoded_cache = extract_cache_states(
        request._cache_state.decoded_cache
    )
```

Replace with (remove entirely — `extract_cache` handles format normalization):
```python
# No second extraction needed — extract_cache handles format
```

**Step 2: Update _handle_prompt_segment_ends**

Find the extraction block in `_handle_prompt_segment_ends` (around line ~2000). Replace:

```python
extracted = extract_cache_states(per_uid_cache)
```

With:

```python
extracted = self._prefix_cache.extract_cache(per_uid_cache)
```

**Step 3: Run tests**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
python -m pytest tests/test_scheduler_cache_fetch.py tests/test_prefix_cache_adapters.py -v --tb=short
```
Expected: Tests pass.

**Step 4: Commit**

```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
git add vllm_mlx/scheduler.py
git commit -m "refactor: replace direct extract_cache_states() calls with CacheManager.extract_cache()"
```

---

### Task 7: Simplify cache persistence methods

**Files:**
- Modify: `vllm_mlx/scheduler.py`

**Step 1: Update save_cache_to_disk**

Replace the entire method:

```python
def save_cache_to_disk(self, cache_dir: str) -> bool:
    """Save prefix cache to disk for persistence across restarts."""
    if self._prefix_cache is not None:
        return self._prefix_cache.save(cache_dir)
    return False
```

**Step 2: Update load_cache_from_disk**

Replace the entire method:

```python
def load_cache_from_disk(self, cache_dir: str) -> int:
    """Load prefix cache from disk. Returns number of entries loaded."""
    if self._prefix_cache is not None:
        return self._prefix_cache.load(cache_dir)
    return 0
```

**Step 3: Update clear_prefix_cache**

Replace the entire method:

```python
def clear_prefix_cache(self) -> None:
    """Clear the in-memory prefix cache (keeps disk cache untouched)."""
    if self._prefix_cache is not None:
        self._prefix_cache.clear()
```

**Step 4: Update close_ssd_tier**

Replace the entire method:

```python
def close_ssd_tier(self) -> None:
    """Shut down the SSD cache tier if present."""
    if self._prefix_cache is not None:
        self._prefix_cache.close()
```

**Step 5: Run tests**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
python -m pytest tests/test_scheduler_cache_fetch.py tests/test_prefix_cache_adapters.py -v --tb=short
```
Expected: Tests pass.

**Step 6: Commit**

```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
git add vllm_mlx/scheduler.py
git commit -m "refactor: simplify cache persistence methods through CacheManager protocol"
```

---

### Task 8: Update _ensure_batch_generator cache entry counting

**Files:**
- Modify: `vllm_mlx/scheduler.py`

**Step 1: Update cache entry counting**

Find the cache entry counting block in `_ensure_batch_generator` (around line ~900). Replace:

```python
if self.batch_generator is not None:
    n_entries = 0
    if self.memory_aware_cache is not None:
        n_entries = len(self.memory_aware_cache._entries)
    elif self.prefix_cache is not None:
        n_entries = (
            len(self.prefix_cache)
            if hasattr(self.prefix_cache, "__len__")
            else 0
        )
    logger.info(
        f"[batch_generator] recreating (sampler params changed), "
        f"keeping {n_entries} cache entries"
    )
```

With:

```python
if self.batch_generator is not None:
    logger.info(
        "[batch_generator] recreating (sampler params changed), "
        "keeping cache entries"
    )
```

**Step 2: Run tests**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
python -m pytest tests/test_scheduler_cache_fetch.py -v --tb=short
```
Expected: Tests pass.

**Step 3: Commit**

```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
git add vllm_mlx/scheduler.py
git commit -m "refactor: remove deprecated cache entry counting from _ensure_batch_generator"
```

---

### Task 9: Remove deprecated test files

**Files:**
- Delete: `tests/test_prefix_cache.py`
- Delete: `tests/test_memory_cache.py`
- Delete: `tests/test_paged_cache.py`
- Delete: `tests/test_paged_cache_benefits.py`
- Delete: `tests/test_paged_cache_real_inference.py`
- Delete: `tests/test_paged_cache_real_model.py`

**Step 1: Verify no remaining references**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
rg 'test_prefix_cache\.py|test_memory_cache\.py|test_paged_cache' tests/ --glob '!test_prefix_cache_adapters.py'
```
Expected: No results (or only references in test_prefix_cache_adapters.py which we keep).

**Step 2: Remove files**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
rm tests/test_prefix_cache.py
rm tests/test_memory_cache.py
rm tests/test_paged_cache.py
rm tests/test_paged_cache_benefits.py
rm tests/test_paged_cache_real_inference.py
rm tests/test_paged_cache_real_model.py
```

**Step 3: Run remaining tests**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
python -m pytest tests/test_prefix_cache_adapters.py tests/test_scheduler_cache_fetch.py tests/test_kv_cache.py tests/test_ssd_cache.py tests/test_ssd_offloaded_cache.py tests/test_cache_disk_store.py -v --tb=short
```
Expected: All pass.

**Step 4: Commit**

```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
git add -A
git rm tests/test_prefix_cache.py tests/test_memory_cache.py tests/test_paged_cache.py tests/test_paged_cache_benefits.py tests/test_paged_cache_real_inference.py tests/test_paged_cache_real_model.py
git commit -m "chore: remove deprecated backend test files (PrefixCacheManager, PagedCacheManager, MemoryAwarePrefixCache)"
```

---

### Task 10: Remove orphaned free functions from kv_cache.py

> **New task (corrected plan).** `extract_cache_states()` and `validate_cache()` are only called from scheduler.py, which now routes through `self._prefix_cache.validate()` / `self._prefix_cache.extract_cache()`. These free functions are orphaned.

**Files:**
- Modify: `vllm_mlx/kv_cache.py`

**Step 1: Verify zero callers remain**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
rg 'extract_cache_states\(' vllm_mlx/ | grep -v 'def extract_cache_states' | grep -v 'extract_cache_states()' | grep -v 'extract_cache_states(' | head -10
rg 'validate_cache\(' vllm_mlx/ | grep -v 'def validate_cache' | head -10
```
Expected: No results (both are orphaned after scheduler refactor).

**Step 2: Remove the functions**

Remove `extract_cache_states()` and `validate_cache()` from `vllm_mlx/kv_cache.py`. Keep `extract_layer_state()` — it's still used by `prefix_cache_adapters.py`.

**Step 3: Run tests**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
python -m pytest tests/test_prefix_cache_adapters.py tests/test_scheduler_cache_fetch.py tests/test_kv_cache.py -v --tb=short
```
Expected: All pass (tests that referenced `extract_cache_states` or `validate_cache` directly should have been updated in earlier tasks).

**Step 4: Commit**

```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
git add vllm_mlx/kv_cache.py
git commit -m "refactor: remove orphaned extract_cache_states() and validate_cache() from kv_cache.py"
```

---

### Task 11: Final verification

**Files:**
- Run: full test suite

**Step 1: Run full test suite**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```
Expected: All tests pass. Any failures indicate remaining references to deprecated backends.

**Step 2: Run specific cache-related tests**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
python -m pytest tests/test_prefix_cache_adapters.py tests/test_scheduler_cache_fetch.py tests/test_kv_cache.py tests/test_cache_disk_store.py tests/test_ssd_cache.py tests/test_ssd_offloaded_cache.py tests/test_memory_cache_mlx.py -v --tb=short
```
Expected: All pass.

**Step 3: Verify Scheduler imports**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
python -c "
from vllm_mlx.scheduler import Scheduler, SchedulerConfig
from vllm_mlx.prefix_cache_adapters import CacheManager
print('Scheduler imports OK')
print('CacheManager has validate:', hasattr(CacheManager, 'validate'))
print('CacheManager has extract_cache:', hasattr(CacheManager, 'extract_cache'))
print('CacheManager has save:', hasattr(CacheManager, 'save'))
print('CacheManager has load:', hasattr(CacheManager, 'load'))
print('CacheManager has close:', hasattr(CacheManager, 'close'))
"
```
Expected: All print statements succeed.

**Step 4: Verify Scheduler only knows about CacheManager**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
rg 'self\.(memory_aware_cache|prefix_cache|paged_cache_manager|block_aware_cache|_ssd_tier|_ssd_offloaded_cache)\.' vllm_mlx/scheduler.py
```
Expected: No results.

**Step 5: Verify no direct calls to orphaned functions remain**

Run:
```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
rg 'extract_cache_states\(' vllm_mlx/scheduler.py
rg 'validate_cache\(' vllm_mlx/scheduler.py
```
Expected: No results (all calls now go through `self._prefix_cache`).

**Step 6: Final commit**

```bash
cd /Users/tibo/Projects/vllm-mlx/test-pi-subagents
git add -A
git commit -m "chore: final verification — Scheduler only knows CacheManager protocol"
```

---

## Summary

**Files modified:**
- `vllm_mlx/prefix_cache_adapters.py` — Add 5 methods to CacheManager, implement on TurnCacheManager (3 new, 2 updated with error handling)
- `vllm_mlx/scheduler.py` — Remove deprecated backends, simplify all cache operations through protocol
- `vllm_mlx/kv_cache.py` — Remove orphaned `extract_cache_states()` and `validate_cache()` (Task 10, new)

**Files deleted:**
- `tests/test_prefix_cache.py`
- `tests/test_memory_cache.py`
- `tests/test_paged_cache.py`
- `tests/test_paged_cache_benefits.py`
- `tests/test_paged_cache_real_inference.py`
- `tests/test_paged_cache_real_model.py`

**Net effect:**
- Scheduler goes from knowing about 8 concrete backends to knowing about 1 (`CacheManager`)
- ~200 lines removed from scheduler.py
- ~30 lines added to adapters (3 new methods, 2 updated)
- All cache operations flow through the protocol
- Deprecated backends removed
- Orphaned free functions removed from kv_cache.py

**Verification:** Full test suite + grep for remaining deprecated references + grep for orphaned function calls.