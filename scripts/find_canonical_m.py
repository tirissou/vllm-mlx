#!/usr/bin/env python3
"""Recommend prefill_step_size by probing canonical chunking regimes.

Loads an mlx-lm-compatible model and runs run_chunking_probe across a grid
of (N, batch_size). Computes per-B canonical bands, intersects them across
batch sizes, and recommends the largest M in the intersection capped at the
smallest sliding max_size.

Exit code: 0 on success, 1 if no M is canonical across all probed batch sizes.
"""

from __future__ import annotations

import argparse
import json
import sys

from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, RotatingKVCache

from vllm_mlx.canonical_m_probe import (
    compute_canonical_band,
    intersect_per_batch_bands,
    run_chunking_probe,
)


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _sliding_max_size(model) -> int | None:
    cache = make_prompt_cache(model)
    sizes = [c.max_size for c in cache if isinstance(c, RotatingKVCache)]
    return min(sizes) if sizes else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe an mlx-lm model and recommend prefill_step_size."
    )
    parser.add_argument("--model", required=True, help="mlx-lm path or HF id")
    parser.add_argument("--n", default="512,1024,2048,4096",
                        help="Comma-separated N values to probe.")
    parser.add_argument("--batch-sizes", default="1,2,4,8",
                        help="Comma-separated batch sizes to probe.")
    parser.add_argument("--json", action="store_true",
                        help="Also emit a JSON summary after the human report.")
    args = parser.parse_args()

    n_values = _parse_csv_ints(args.n)
    batch_sizes = _parse_csv_ints(args.batch_sizes)

    print(f"Loading model: {args.model} ...", flush=True)
    model, _ = load(args.model)

    sliding_max = _sliding_max_size(model)
    cache = make_prompt_cache(model)
    n_layers = len(cache)
    n_sliding = sum(1 for c in cache if isinstance(c, RotatingKVCache))
    n_full = n_layers - n_sliding
    del cache

    print(f"Model: {args.model}")
    print(f"Layers: {n_layers} (KVCache: {n_full}, RotatingKVCache: {n_sliding})")
    print(f"Sliding max_size: {sliding_max}")
    print()
    print("Per-batch-size canonical bands:")

    per_batch_bands: dict[int, list[int]] = {}
    for B in batch_sizes:
        band_for_B: set[int] = set()
        # Probe each N independently, union zero-diff M values found.
        for N in n_values:
            try:
                probe = run_chunking_probe(model, n_tokens=N, batch_size=B)
            except ValueError as exc:
                print(f"  B={B} N={N} skipped: {exc}")
                continue
            band_for_B.update(compute_canonical_band(probe))
        sorted_band = sorted(band_for_B)
        per_batch_bands[B] = sorted_band
        if sorted_band:
            print(f"  B={B:<3} band: M ∈ {sorted_band}")
        else:
            print(f"  B={B:<3} band: (no canonical M found)")
    print()

    intersection = intersect_per_batch_bands(per_batch_bands)
    if sliding_max is not None:
        intersection = [M for M in intersection if M <= sliding_max]

    if not intersection:
        print("Intersection (canonical across all B): (empty)")
        print()
        print("FAILURE: no M is canonical across all probed batch sizes.")
        print("Lower --prefill-batch-size or investigate per-B kernel regimes "
              "separately.")
        if args.json:
            print()
            json.dump({
                "model": args.model,
                "n_layers": n_layers,
                "n_sliding": n_sliding,
                "n_full": n_full,
                "sliding_max_size": sliding_max,
                "per_batch_bands": {str(k): v for k, v in per_batch_bands.items()},
                "canonical_intersection": [],
                "error": "empty_intersection",
            }, sys.stdout, indent=2)
            print()
        return 1

    recommended = max(intersection)
    print(f"Intersection (canonical across all B): M ∈ {intersection}")
    if sliding_max is not None:
        print(f"Sliding max_size constraint: M <= {sliding_max}")
    print()
    print(f"Recommended: --prefill-step-size {recommended}")

    if args.json:
        print()
        json.dump({
            "model": args.model,
            "n_layers": n_layers,
            "n_sliding": n_sliding,
            "n_full": n_full,
            "sliding_max_size": sliding_max,
            "per_batch_bands": {str(k): v for k, v in per_batch_bands.items()},
            "canonical_intersection": intersection,
            "recommended_prefill_step_size": recommended,
        }, sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
