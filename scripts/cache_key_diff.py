#!/usr/bin/env python3
"""
Diff two cache key log entries to find where their token sequences diverge.

Usage:
    python scripts/cache_key_diff.py LOG            # diff last two entries
    python scripts/cache_key_diff.py LOG --list     # list all entries
    python scripts/cache_key_diff.py LOG 3 7        # diff entries #3 and #7
    python scripts/cache_key_diff.py LOG 3 7 --model mlx-community/Qwen3-8B-4bit
"""

import argparse
import json
import sys
from datetime import datetime


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                r["_lineno"] = lineno
                r["_idx"] = len(records) + 1
                records.append(r)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping line {lineno}: {e}", file=sys.stderr)
    return records


def common_prefix_len(a: list, b: list) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3] if ts else "?"


def fmt_tokens(tokens: list[int], start: int, count: int) -> str:
    chunk = tokens[start : start + count]
    return " ".join(str(t) for t in chunk)


def list_records(records: list[dict]) -> None:
    print(f"{'#':>4}  {'ln':>5}  {'op':>4}  {'n_tokens':>8}  {'time':>12}  request_id")
    print("-" * 80)
    for r in records:
        print(
            f"{r['_idx']:>4}  {r['_lineno']:>5}  {r.get('op', '?'):>4}"
            f"  {r.get('n_tokens', 0):>8}  {fmt_ts(r.get('ts', 0)):>12}"
            f"  {r.get('request_id', '?')}"
        )


def decode_tokens(tokens: list[int], tokenizer) -> str:
    try:
        return tokenizer.decode(tokens)
    except Exception:
        return "(decode failed)"


def diff_records(a: dict, b: dict, context: int, tokenizer=None) -> None:
    ta = a.get("tokens", [])
    tb = b.get("tokens", [])
    overlap = common_prefix_len(ta, tb)
    max_len = max(len(ta), len(tb), 1)
    pct = overlap / max_len * 100

    bar_width = 60
    filled = round(overlap / max_len * bar_width)
    bar = "=" * filled + "-" * (bar_width - filled)

    print(f"\nA  #{a['_idx']:>4} (line {a['_lineno']:>5})  op={a.get('op','?'):>4}"
          f"  tokens={len(ta):>6}  {fmt_ts(a.get('ts',0))}  {a.get('request_id','?')}")
    print(f"B  #{b['_idx']:>4} (line {b['_lineno']:>5})  op={b.get('op','?'):>4}"
          f"  tokens={len(tb):>6}  {fmt_ts(b.get('ts',0))}  {b.get('request_id','?')}")

    print(f"\nOverlap  {overlap} / {max_len} tokens  ({pct:.1f}%)")
    print(f"[{bar}]")

    if len(ta) != len(tb):
        print(f"Length delta  {len(tb) - len(ta):+d}  (A={len(ta)}, B={len(tb)})")

    is_prefix = overlap == min(len(ta), len(tb))
    if is_prefix and len(ta) != len(tb):
        shorter_name = "A" if len(ta) < len(tb) else "B"
        longer_tokens = tb if len(ta) < len(tb) else ta
        print(f"\nNo divergence — {shorter_name} is a prefix of the other.")
        print(f"Continuation (+{context} tokens from idx {overlap}):")
        print(f"  {fmt_tokens(longer_tokens, overlap, context)}")
        if tokenizer:
            decoded = decode_tokens(longer_tokens[overlap : overlap + context], tokenizer)
            print(f"  decoded: {decoded!r}")
    elif is_prefix:
        print("\nSequences are identical.")
    else:
        pre_start = max(0, overlap - 5)
        print(f"\nFirst divergence at token index {overlap}:")
        if pre_start < overlap:
            print(f"  shared tail (idx {pre_start}-{overlap-1}): {fmt_tokens(ta, pre_start, overlap - pre_start)}")
        print(f"  A[{overlap}:+{context}]: {fmt_tokens(ta, overlap, context)}")
        print(f"  B[{overlap}:+{context}]: {fmt_tokens(tb, overlap, context)}")
        if tokenizer:
            print(f"  A decoded: {decode_tokens(ta[overlap:overlap+context], tokenizer)!r}")
            print(f"  B decoded: {decode_tokens(tb[overlap:overlap+context], tokenizer)!r}")


def load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Warning: could not load tokenizer for {model_name!r}: {e}", file=sys.stderr)
        return None


def resolve_index(spec: str, records: list[dict]) -> int:
    """Resolve a 1-based record index. Negative values count from end."""
    n = int(spec)
    if n < 0:
        n = len(records) + n + 1
    if not (1 <= n <= len(records)):
        print(f"Error: index {n} out of range (1-{len(records)})", file=sys.stderr)
        sys.exit(1)
    return n - 1


def main():
    parser = argparse.ArgumentParser(
        description="Diff two cache key log entries to find token sequence divergence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("file", help="Cache key log file (JSON lines)")
    parser.add_argument("a", nargs="?", help="First record index (1-based; default: second-to-last)")
    parser.add_argument("b", nargs="?", help="Second record index (1-based; default: last)")
    parser.add_argument("--list", "-l", action="store_true", help="List all records and exit")
    parser.add_argument("--context", "-c", type=int, default=40,
                        help="Tokens to show around divergence point (default: 40)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="HuggingFace model name to decode tokens (optional)")
    args = parser.parse_args()

    records = load_records(args.file)
    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    if args.list:
        list_records(records)
        return

    if len(records) < 2 and args.a is None:
        print("Error: need at least 2 records (or specify indices explicitly)", file=sys.stderr)
        sys.exit(1)

    idx_a = resolve_index(args.a, records) if args.a is not None else len(records) - 2
    idx_b = resolve_index(args.b, records) if args.b is not None else len(records) - 1

    tokenizer = load_tokenizer(args.model) if args.model else None

    diff_records(records[idx_a], records[idx_b], context=args.context, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
