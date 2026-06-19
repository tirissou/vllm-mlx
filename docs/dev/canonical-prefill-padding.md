# Canonical prefill padding — operator notes

## What it does

Every prefill forward pass through the model runs with `S == prefill_step_size`.
Sub-canonical chunks (from boundary-aware splitting or short follow-up turns)
are right-padded before the forward and the resulting pad rows are trimmed
from full-attention caches and physically sliced off rotating-cache buffers.

`TurnCacheManager.store()` is a no-op: decoded K, V (computed at M=1, the
worst kernel regime) never enter the trie. Promotion happens exclusively
through `on_prefill_checkpoint()` at turn boundaries during prefill.

## Picking `--prefill-step-size`

Run the probe CLI against your deployed model:

```
python scripts/find_canonical_m.py --model <model-path-or-hf-id> \
    --batch-sizes 1,<your-prefill-batch-size>
```

Set `--prefill-step-size` to the value reported on the `Recommended:` line.

## Release note

> Decoded tokens are no longer cached. Cache hits land at prefill boundaries
> only; the prior turn's assistant response is re-prefilled on the next turn.
> This trades a one-time per-cache-hit cost (replay of the prior assistant
> response, ~500 tokens for typical conversations) for stable multi-turn
> quality.

## Validating post-deployment

- Replay a 3-turn cache-hit scenario; quality should remain stable across
  10+ turns instead of slowly degrading.
- Inspect `_log_segment_breakdown` log lines: no node sizes should match
  `output_token_ids` lengths (would indicate `store()` is still promoting).
- Optional: set `VLLM_MLX_VERIFY_FETCH_KV=1`; `verify_kv` diff magnitudes
  should drop meaningfully if decoded K,V no longer pollute the cache.
