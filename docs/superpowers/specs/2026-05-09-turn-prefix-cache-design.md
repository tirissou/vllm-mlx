# TurnPrefixCache Design

**Date:** 2026-05-09  
**Branch:** fix-reasoning_content-input  
**Status:** Awaiting implementation

---

## Motivation

The two existing cache modes in vllm-mlx have complementary weaknesses for hybrid models like Qwen3.6-27B (qwen3_5 architecture: 16 full-attention + 48 recurrent layers):

| | MemoryAwarePrefixCache | BlockAwarePrefixCache |
|---|---|---|
| Sharing | None (full copy per entry) | Block-level KV deduplication |
| Recurrent checkpoints | 1 per entry (session end) | Terminal block only |
| Mid-session branching | No | No |
| 150K token session cost | ~9.2 GB (bf16) | ~9.2 GB (bf16) |

Neither handles hybrid recurrent state correctly at scale. `TurnPrefixCache` is a third cache mode that replaces both for the hybrid model use case.

**Memory target:** 150K token session under 10 GB RAM.

---

## Architecture

### Trie structure

A trie where each node represents one **conversation segment**: a single post-normalization message (`<|im_start|>role...content<|im_end|>` block). The root is a sentinel. Children are identified by a **context-sensitive hash**:

```
context_hash(node) = hash(parent.context_hash || node.token_ids)
```

Context-sensitivity is required because KV values depend on the full preceding context — identical content at different trie depths has different KV values and must not be shared.

### TurnNode

```python
@dataclass
class TurnNode:
    token_ids: list[int]
    context_hash: int
    kv_arrays: list[array] | SSDRef | None  # per-layer int8 KV; None = root
    kv_scales: list[float] | None           # dequantization scale per layer
    recurrent_state: array | SSDRef | None  # checkpoint if applicable
    tokens_since_checkpoint: int            # cumulative from nearest permanent checkpoint ancestor at creation time; not updated when ancestor temp checkpoints are pruned
    children: dict[int, TurnNode]           # context_hash → child
    ref_count: int                          # pinned while > 0 (active requests)
    last_used: float                        # unix timestamp; updated on traversal
```

---

## Segment definition

**Input:** post-`_normalize_messages()` message list. Each `Message` object becomes one segment. Normalization merges consecutive same-role messages before the cache sees them, so the cache operates on the merged view.

**Tokenization:** each segment is tokenized in its conversational context (not in isolation) to produce correct token IDs including chat template special tokens. Segment boundaries are the role-delimiter positions in the full token sequence.

**Tool call handling:** parallel tool calls from one assistant turn are already merged into a single `assistant` message by `_normalize_messages()`. Multiple consecutive `tool` result messages are similarly merged into one `tool` node. This is correct: tool outputs are request-specific and rarely produce cache hits regardless of granularity.

---

## Recurrent checkpointing

### Policy: N-token stride (configurable)

`checkpoint_stride: int` — minimum cumulative tokens since the last permanent checkpoint before a new one is written. Setting `checkpoint_stride = 0` checkpoints every node (eager).

**Two checkpoint types:**

| Type | Condition | Lifetime |
|---|---|---|
| Temporary (leaf) | node is a leaf | freed when node gains its first child AND tokens_since_checkpoint < N |
| Permanent (stride) | tokens_since_checkpoint >= N, OR system prompt node | kept until node is RAM-evicted |

**On leaf → inner node transition:** if `tokens_since_checkpoint < checkpoint_stride`, free `recurrent_state`. The node retains KV arrays for prefix matching and gap reconstruction.

**System prompt node:** always receives a permanent checkpoint regardless of stride.

### Example (N=512, turns of ~150 tokens)

```
[sys_prompt: 500 tok, Δ=0]    permanent checkpoint (always)
  └─[user_1: 50 tok, Δ=50]    temp → pruned when asst_1 added
      └─[asst_1: 150 tok, Δ=200]  temp → pruned when user_2 added
          └─[user_2: 50 tok, Δ=250]  temp → pruned
              └─[asst_2: 200 tok, Δ=450]  temp → pruned
                  └─[user_3: 100 tok, Δ=550]  permanent checkpoint (≥512) + temp
                      └─[asst_3: 150 tok, Δ=150]  temp (leaf)
```

---

## KV storage

**Format:** int8, quantized from bf16 on write, dequantized to bf16 before attention. A per-layer scale factor stored alongside. Per-tensor scaling (simplest; per-channel can be added later for accuracy).

**Memory:** `150,000 tokens × 32 KB/token (int8) = 4.7 GB` — well under the 10 GB target, leaving headroom for recurrent checkpoints and model weights.

**Why int8:** the 16 full-attention layers cost 64 KB/token at bf16. At 150K tokens that is 9.15 GB — nearly the full budget before any recurrent checkpoints. int8 halves this to 4.7 GB.

---

## Prefix matching

```
match(segments) → (matched_node, has_recurrent_state)

walk from root:
  for each segment:
    h = context_hash(current_node.context_hash, segment.token_ids)
    if h not in current_node.children: break
    current_node = current_node.children[h]
    current_node.last_used = now()           ← propagates to all ancestors naturally
    current_node.ref_count += 1              ← pinned for request duration; decremented on request completion
  return current_node
```

`last_used` is updated on every node during the downward walk — no upward propagation step needed. All ancestors of the matched node are visited and timestamped in one pass.

### Reconstruction when matched node has no recurrent state

1. Walk up from matched node to the nearest ancestor with `recurrent_state` (permanent checkpoint or leaf checkpoint)
2. Re-run model forward pass for the gap tokens (matched_node.depth − checkpoint_ancestor.depth turns)
3. KV arrays for the gap are already in the trie: attention is cheap (no KV recomputation). Only the recurrent layers need rebuilding.
4. Gap is bounded by `checkpoint_stride / avg_tokens_per_turn` turns.

---

## Eviction

**Pure LRU, no priority for checkpoints.** A node is evictable iff `ref_count == 0` AND it has no children (leaf).

**Eviction heap:** min-heap keyed by `last_used`, containing only current evictable leaves.

**Algorithm:**
1. Pop LRU leaf from heap
2. Free `kv_arrays` and `recurrent_state` (or cancel pending SSD write)
3. Remove node from parent's `children`
4. Decrement parent's notional child count; if parent is now a leaf and `ref_count == 0`: add parent to heap
5. Repeat until memory is within budget

---

## Disk persistence (across server restarts)

**SQLite index** (`turn_cache_index.db`) + one safetensors file per node.

### SQLite schema

```sql
CREATE TABLE nodes (
    context_hash            INTEGER PRIMARY KEY,
    parent_hash             INTEGER,          -- 0 for root children
    token_ids_blob          BLOB NOT NULL,
    kv_file_path            TEXT NOT NULL,
    recurrent_file_path     TEXT,             -- NULL if no checkpoint
    last_used               REAL NOT NULL,
    tokens_since_checkpoint INTEGER NOT NULL,
    is_permanent_checkpoint INTEGER NOT NULL  -- 0 or 1
);
```

**Save (on shutdown):** async writer thread processes a queue; writes temp file then renames atomically for crash consistency.

**Load (on startup):** read all rows, reconstruct trie by linking `context_hash → parent_hash`. One pass sufficient since parent hashes are stored. Nodes with missing files are skipped and logged.

**Format versioning:** `_CACHE_FORMAT_VERSION` constant; mismatched version on load → reject and start empty.

---

## SSD offloading (hot/cold KV tiering)

The trie structure (hashes, metadata, child pointers) **always stays in RAM** — it is tiny (kilobytes per node) and required for prefix matching without touching disk.

Only `kv_arrays` and `recurrent_state` are tiered.

### SSDRef sentinel

```python
@dataclass
class SSDRef:
    file_path: str
    size_bytes: int
```

When a node's KV is spilled to SSD, `kv_arrays` is replaced with an `SSDRef`. Prefix matching still works (trie traversal does not read KV). Data is loaded only when a request needs it.

### On RAM eviction with SSD enabled

1. Async-write KV arrays + recurrent state to SSD
2. Replace `kv_arrays` / `recurrent_state` with `SSDRef`
3. Node stays in trie; data is on disk

### On cache hit with SSDRef

1. Reserve RAM budget (block until space is available or timeout)
2. Read from SSD synchronously — NVMe reads (~2 ms for a typical segment) are ~100× faster than re-prefill (~200 ms), so always wait
3. Promote arrays back to RAM, clear `SSDRef`
4. Only fall back to re-prefill if SSD read **errors out** (file missing or corrupt)

Partial promotion: if `kv_file` is corrupt but `recurrent_file` is intact, load recurrent state and recompute KV during prefill. Vice versa is not useful (KV without recurrent state requires re-run anyway).

---

## Scheduler integration

`TurnPrefixCache` replaces the `prefix_cache` field in the scheduler. Enabled via `--use-turn-cache` CLI flag. Existing `--use-paged-cache` and default memory cache remain untouched.

### Two scheduler touch points

**`_fetch_prefix_cache` (before prefill):**
```
segments = split_messages_into_segments(request.messages)
matched_node, has_recurrent = cache.match(segments)
if not has_recurrent:
    checkpoint_ancestor = find_nearest_checkpoint(matched_node)
    schedule_gap_rerun(checkpoint_ancestor, matched_node)
request.cached_tokens = matched_node.depth_in_tokens
request.prompt_cache  = matched_node.recurrent_state
```

**`_process_batch_responses` (after generation):**
```
new_segments = input_segments_beyond_match + [generated_output_segment]
for segment in new_segments:
    node = cache.insert(parent_node, segment.token_ids, segment.kv_arrays, segment.recurrent_state)
    apply_checkpoint_prune_rules(node)
```

---

## Configuration

```python
@dataclass
class TurnPrefixCacheConfig:
    checkpoint_stride: int = 512   # tokens between permanent checkpoints; 0 = every node
    max_memory_gb: float = 8.0     # RAM budget for KV + recurrent state
    kv_dtype: str = "int8"         # "bf16" for accuracy; "int8" for 2× memory savings
    persist_dir: str | None = None # None = disabled; saves trie on shutdown, loads on start
    ssd_max_gb: float = 0.0        # 0 = disabled; enables hot/cold KV tiering to SSD
```

---

## Files affected

| File | Change |
|---|---|
| `vllm_mlx/turn_prefix_cache.py` | New — `TurnNode`, `SSDRef`, `TurnPrefixCache` |
| `vllm_mlx/scheduler.py` | Replace `prefix_cache` with `TurnPrefixCache` under `--use-turn-cache` |
| `vllm_mlx/cli.py` | Add `--use-turn-cache`, `--turn-cache-stride`, `--turn-cache-ssd-gb` flags |
| `tests/test_turn_prefix_cache.py` | New — unit + integration tests |

---

## Memory budget summary (150K token session)

| Component | Size |
|---|---|
| KV cache (int8, 150K tokens) | 4.7 GB |
| Permanent recurrent checkpoints (sparse, ~N=512) | ~300 MB |
| Leaf recurrent state (1 active leaf) | 75 MB |
| Trie structure (100 nodes × metadata) | < 1 MB |
| **Total** | **~5.1 GB** |

Leaves 4.9 GB headroom for model weights and concurrent requests within a 10 GB budget.
