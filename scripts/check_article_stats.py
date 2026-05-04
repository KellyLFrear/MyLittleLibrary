#!/usr/bin/env python3
"""
check_article_stats.py

Purpose:
- Read outputs/article_stats.jsonl
- Print quick summaries so you can judge whether your candidate-selection settings make sense
- Optionally simulate alternate coverage windows without rerunning analyze_articles.py

This is a debugging / inspection helper, not a core pipeline generator.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Read command-line options.

    --preview-rows shows sample records.
    --window min:max can be repeated to test alternate candidate windows.
    """
    parser = argparse.ArgumentParser(description="Inspect article_stats.jsonl output.")
    parser.add_argument("--input", default="outputs/article_stats.jsonl", help="Path to article_stats.jsonl")
    parser.add_argument("--preview-rows", type=int, default=3, help="How many rows to preview")
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        help='Optional alternate candidate window in the form "min:max", e.g. 0.45:0.70. Can be repeated.',
    )
    return parser.parse_args()


def parse_window(value: str) -> tuple[float, float]:
    """
    Parse a coverage window string like "0.45:0.70" into floats.

    Also validates that min <= max.
    """
    try:
        low_s, high_s = value.split(":", 1)
        low = float(low_s)
        high = float(high_s)
    except Exception as exc:
        raise SystemExit(f"Invalid --window value: {value!r}. Use format min:max, e.g. 0.45:0.70") from exc

    if low > high:
        raise SystemExit(f"Invalid --window value: {value!r}. min cannot be greater than max.")
    return low, high


def main() -> None:
    """
    Load the JSONL file, summarize it by reading level, and optionally simulate
    new candidate counts for alternate coverage windows.
    """
    args = parse_args()
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Input: {path}")
    print(f"Rows loaded: {len(rows)}")

    # Optionally print a few example rows so you can inspect the JSON structure.
    preview_n = max(0, args.preview_rows)
    if preview_n:
        print(f"\nFirst {min(preview_n, len(rows))} rows:")
        for row in rows[:preview_n]:
            print(json.dumps(row, indent=2, ensure_ascii=False))

    # Track counts and stats by level.
    by_level = Counter()
    candidate_by_level = Counter()
    coverage = defaultdict(list)
    readability = defaultdict(list)

    for row in rows:
        level = row.get("Level", "UNKNOWN")
        by_level[level] += 1

        if row.get("Candidate", False):
            candidate_by_level[level] += 1

        cov = row.get("Coverage_Ratio")
        if isinstance(cov, (int, float)):
            coverage[level].append(float(cov))

        fk = row.get("Flesch_Kincaid_Grade")
        if isinstance(fk, (int, float)):
            readability[level].append(float(fk))

    print("\nRows by level:")
    for level in sorted(by_level):
        print(f"  {level}: {by_level[level]}")

    print("\nCandidate rows by level:")
    for level in sorted(by_level):
        print(f"  {level}: {candidate_by_level[level]}")

    print("\nCoverage ratio stats:")
    for level in sorted(coverage):
        arr = coverage[level]
        print(f"  {level}: min={min(arr):.4f} max={max(arr):.4f} avg={sum(arr)/len(arr):.4f}")

    print("\nReadability stats:")
    for level in sorted(readability):
        arr = readability[level]
        print(f"  {level}: min={min(arr):.2f} max={max(arr):.2f} avg={sum(arr)/len(arr):.2f}")

    # Simulate alternate windows without changing the original JSONL file.
    if args.window:
        print("\nSimulated candidate rows for alternate windows:")
        for raw in args.window:
            low, high = parse_window(raw)
            sim = Counter()
            for row in rows:
                cov = row.get("Coverage_Ratio")
                level = row.get("Level", "UNKNOWN")
                if isinstance(cov, (int, float)) and low <= float(cov) <= high:
                    sim[level] += 1
            print(f"  window {low:.2f}-{high:.2f}:")
            for level in sorted(by_level):
                print(f"    {level}: {sim[level]}")


if __name__ == "__main__":
    main()
