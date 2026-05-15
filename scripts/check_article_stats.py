#!/usr/bin/env python3
"""
check_article_stats.py

Purpose:
- Summarize outputs/article_stats.jsonl.
- Simulate alternate coverage windows.
- Validate that article_stats.jsonl matches the MyLittleLibrary vocabulary logic.

Project vocabulary logic:
- The vocab TXT files are exclusive bands:
    beginner_1000.txt      = beginner-only words
    intermediate_3000.txt  = intermediate-only words
    advanced_6000.txt      = advanced-only words

- Student known-word coverage is cumulative:
    Beginner      = beginner band
    Intermediate  = beginner + intermediate bands
    Advanced      = beginner + intermediate + advanced bands

Typical use:
    python scripts/check_article_stats.py --input outputs/article_stats.jsonl

Fast check on a large file:
    python scripts/check_article_stats.py \
      --input outputs/article_stats.jsonl \
      --max-recompute 1000

Simulate candidate windows:
    python scripts/check_article_stats.py \
      --input outputs/article_stats.jsonl \
      --window 0.85:0.97 \
      --window 0.75:0.95
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

LEVELS = ("Beginner", "Intermediate", "Advanced")
WORD_RE = re.compile(r"[a-zA-Z']+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize and validate article_stats.jsonl against exclusive vocab "
            "bands and cumulative known-word coverage."
        )
    )
    parser.add_argument(
        "--input",
        default="outputs/article_stats.jsonl",
        help="Path to article_stats.jsonl. Default: outputs/article_stats.jsonl",
    )
    parser.add_argument(
        "--beginner-vocab",
        default="data/vocab/beginner_1000.txt",
        help="Beginner vocab band TXT. Default: data/vocab/beginner_1000.txt",
    )
    parser.add_argument(
        "--intermediate-vocab",
        default="data/vocab/intermediate_3000.txt",
        help="Intermediate vocab band TXT. Default: data/vocab/intermediate_3000.txt",
    )
    parser.add_argument(
        "--advanced-vocab",
        default="data/vocab/advanced_6000.txt",
        help="Advanced vocab band TXT. Default: data/vocab/advanced_6000.txt",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=3,
        help="How many JSONL rows to preview. Use 0 to disable. Default: 3",
    )
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        help=(
            'Optional alternate candidate window in the form "min:max", '
            'for example 0.85:0.97. Can be repeated.'
        ),
    )
    parser.add_argument(
        "--coverage-tolerance",
        type=float,
        default=1e-9,
        help="Allowed floating-point difference when recomputing Coverage_Ratio. Default: 1e-9",
    )
    parser.add_argument(
        "--max-recompute",
        type=int,
        default=0,
        help=(
            "Maximum rows to recompute for Coverage_Ratio/New_Word_Count. "
            "Use 0 to recompute all rows. Default: 0"
        ),
    )
    parser.add_argument(
        "--skip-recompute",
        action="store_true",
        help="Skip recomputing Coverage_Ratio/New_Word_Count from Text. Faster, but weaker validation.",
    )
    parser.add_argument(
        "--preview-errors",
        type=int,
        default=5,
        help="How many example errors to print per validation category. Default: 5",
    )
    return parser.parse_args()


def parse_window(value: str) -> tuple[float, float]:
    try:
        low_s, high_s = value.split(":", 1)
        low = float(low_s)
        high = float(high_s)
    except Exception as exc:
        raise SystemExit(
            f"Invalid --window value: {value!r}. Use format min:max, e.g. 0.85:0.97"
        ) from exc

    if low > high:
        raise SystemExit(f"Invalid --window value: {value!r}. min cannot be greater than max.")
    return low, high


def load_word_list(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Vocab file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(str(text).lower())


def load_vocab_sets(args: argparse.Namespace) -> tuple[dict[str, set[str]], dict[str, set[str]], list[str]]:
    beginner_band = load_word_list(args.beginner_vocab)
    intermediate_band = load_word_list(args.intermediate_vocab)
    advanced_band = load_word_list(args.advanced_vocab)

    bands = {
        "Beginner": beginner_band,
        "Intermediate": intermediate_band,
        "Advanced": advanced_band,
    }

    known_sets = {
        "Beginner": beginner_band,
        "Intermediate": beginner_band | intermediate_band,
        "Advanced": beginner_band | intermediate_band | advanced_band,
    }

    overlap_errors: list[str] = []
    overlaps = {
        "Beginner ∩ Intermediate": beginner_band & intermediate_band,
        "Beginner ∩ Advanced": beginner_band & advanced_band,
        "Intermediate ∩ Advanced": intermediate_band & advanced_band,
    }

    print("Vocabulary band sizes:")
    for level in LEVELS:
        print(f"  {level} band: {len(bands[level]):,}")

    print("\nVocabulary band overlap:")
    for label, words in overlaps.items():
        print(f"  {label}: {len(words):,}")
        if words:
            sample = ", ".join(sorted(words)[: args.preview_errors])
            overlap_errors.append(f"{label} has overlap. Sample: {sample}")

    print("\nCumulative known-word totals:")
    for level in LEVELS:
        print(f"  {level}: {len(known_sets[level]):,}")

    return bands, known_sets, overlap_errors


def numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def boolish(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def add_error(errors: dict[str, list[str]], category: str, message: str, limit: int) -> None:
    if len(errors[category]) < limit:
        errors[category].append(message)


def print_numeric_stats(title: str, values_by_level: dict[str, list[float]], decimals: int = 4) -> None:
    print(f"\n{title}:")
    for level in LEVELS:
        arr = values_by_level.get(level, [])
        if not arr:
            print(f"  {level}: no numeric values")
            continue
        print(
            f"  {level}: "
            f"min={min(arr):.{decimals}f} "
            f"avg={mean(arr):.{decimals}f} "
            f"max={max(arr):.{decimals}f}"
        )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    windows = [parse_window(raw) for raw in args.window]
    _, known_sets, overlap_errors = load_vocab_sets(args)

    errors: dict[str, list[str]] = defaultdict(list)
    for msg in overlap_errors:
        errors["vocab overlap"].append(msg)

    required_fields = {"ID", "Title", "Text", "Level", "Coverage_Ratio", "New_Word_Count"}

    total_rows = 0
    invalid_json_lines = 0
    preview_rows: list[dict[str, Any]] = []

    rows_by_level: Counter[str] = Counter()
    candidate_by_level: Counter[str] = Counter()
    simulated_candidates: list[Counter[str]] = [Counter() for _ in windows]

    coverage_by_level: dict[str, list[float]] = defaultdict(list)
    readability_by_level: dict[str, list[float]] = defaultdict(list)
    new_words_by_level: dict[str, list[float]] = defaultdict(list)
    total_words_by_level: dict[str, list[float]] = defaultdict(list)

    # Store only the small amount needed for article-level monotonicity checks.
    # This avoids keeping every full Text field in memory for large runs.
    article_levels: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    recompute_limit = None
    if not args.skip_recompute and args.max_recompute > 0:
        recompute_limit = args.max_recompute

    recompute_rows_checked = 0
    coverage_mismatches = 0
    new_word_mismatches = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_json_lines += 1
                add_error(
                    errors,
                    "invalid json",
                    f"line {line_number}: {exc}",
                    args.preview_errors,
                )
                continue

            total_rows += 1
            if len(preview_rows) < max(0, args.preview_rows):
                preview_rows.append(row)

            missing = sorted(required_fields - set(row))
            if missing:
                add_error(
                    errors,
                    "missing fields",
                    f"line {line_number}: missing {', '.join(missing)}",
                    args.preview_errors,
                )

            level = row.get("Level", "UNKNOWN")
            level_str = str(level)
            rows_by_level[level_str] += 1

            if boolish(row.get("Candidate")):
                candidate_by_level[level_str] += 1

            cov = numeric(row.get("Coverage_Ratio"))
            if cov is not None:
                coverage_by_level[level_str].append(cov)
                for index, (low, high) in enumerate(windows):
                    if low <= cov <= high:
                        simulated_candidates[index][level_str] += 1

            fk = numeric(row.get("Flesch_Kincaid_Grade"))
            if fk is not None:
                readability_by_level[level_str].append(fk)

            nwc = numeric(row.get("New_Word_Count"))
            if nwc is not None:
                new_words_by_level[level_str].append(nwc)

            tw = numeric(row.get("Total_Words"))
            if tw is not None:
                total_words_by_level[level_str].append(tw)

            article_id = str(row.get("ID"))
            if level_str in LEVELS:
                if level_str in article_levels[article_id]:
                    add_error(
                        errors,
                        "duplicate article-level rows",
                        f"ID={article_id} has duplicate {level_str} rows",
                        args.preview_errors,
                    )

                article_levels[article_id][level_str] = {
                    "line": line_number,
                    "title": row.get("Title", ""),
                    "coverage": cov,
                    "new_word_count": int(row.get("New_Word_Count", -1))
                    if isinstance(row.get("New_Word_Count"), int)
                    else row.get("New_Word_Count"),
                }

            # Optional stronger validation: recompute coverage directly from Text.
            if not args.skip_recompute and (recompute_limit is None or recompute_rows_checked < recompute_limit):
                if level_str in LEVELS:
                    tokens = tokenize_words(str(row.get("Text", "")))
                    if tokens:
                        recompute_rows_checked += 1
                        known_words = known_sets[level_str]

                        expected_cov = sum(1 for word in tokens if word in known_words) / len(tokens)
                        actual_cov = row.get("Coverage_Ratio")

                        if (
                            not isinstance(actual_cov, (int, float))
                            or isinstance(actual_cov, bool)
                            or abs(float(actual_cov) - expected_cov) > args.coverage_tolerance
                        ):
                            coverage_mismatches += 1
                            add_error(
                                errors,
                                "coverage recompute",
                                (
                                    f"line {line_number} ID={row.get('ID')} {level_str}: "
                                    f"stored={actual_cov} expected={expected_cov:.12f}"
                                ),
                                args.preview_errors,
                            )

                        expected_new_count = len({word for word in tokens if word not in known_words})
                        actual_new_count = row.get("New_Word_Count")

                        if actual_new_count != expected_new_count:
                            new_word_mismatches += 1
                            add_error(
                                errors,
                                "new-word recompute",
                                (
                                    f"line {line_number} ID={row.get('ID')} {level_str}: "
                                    f"stored={actual_new_count} expected={expected_new_count}"
                                ),
                                args.preview_errors,
                            )

    print(f"\nInput: {input_path}")
    print(f"Rows loaded: {total_rows:,}")

    if invalid_json_lines:
        print(f"Invalid JSON lines: {invalid_json_lines:,}")

    if preview_rows:
        print(f"\nFirst {len(preview_rows)} row(s):")
        for row in preview_rows:
            print(json.dumps(row, indent=2, ensure_ascii=False))

    print("\nRows by level:")
    for level in list(LEVELS) + sorted(k for k in rows_by_level if k not in LEVELS):
        print(f"  {level}: {rows_by_level[level]:,}")

    print("\nCandidate rows by level:")
    for level in list(LEVELS) + sorted(k for k in candidate_by_level if k not in LEVELS):
        print(f"  {level}: {candidate_by_level[level]:,}")

    print_numeric_stats("Coverage ratio stats", coverage_by_level, decimals=4)
    print_numeric_stats("New-word-count stats", new_words_by_level, decimals=2)
    print_numeric_stats("Total-word stats", total_words_by_level, decimals=2)
    print_numeric_stats("Readability stats", readability_by_level, decimals=2)

    if windows:
        print("\nSimulated candidate rows for alternate windows:")
        for (low, high), counter in zip(windows, simulated_candidates):
            print(f"  window {low:.2f}-{high:.2f}:")
            for level in LEVELS:
                print(f"    {level}: {counter[level]:,}")

    missing_level_count = 0
    coverage_monotonicity_failures = 0
    new_word_monotonicity_failures = 0

    for article_id, level_map in article_levels.items():
        missing_levels = [level for level in LEVELS if level not in level_map]
        if missing_levels:
            missing_level_count += 1
            add_error(
                errors,
                "missing article levels",
                f"ID={article_id} missing level(s): {', '.join(missing_levels)}",
                args.preview_errors,
            )
            continue

        b_cov = level_map["Beginner"].get("coverage")
        i_cov = level_map["Intermediate"].get("coverage")
        a_cov = level_map["Advanced"].get("coverage")

        if not all(isinstance(x, float) and math.isfinite(x) for x in (b_cov, i_cov, a_cov)):
            add_error(
                errors,
                "coverage values",
                f"ID={article_id} has missing/non-numeric coverage value(s)",
                args.preview_errors,
            )
        elif not (b_cov <= i_cov <= a_cov):
            coverage_monotonicity_failures += 1
            title = level_map["Beginner"].get("title", "")
            add_error(
                errors,
                "coverage monotonicity",
                (
                    f"ID={article_id} title={title!r}: "
                    f"Beginner={b_cov:.6f}, Intermediate={i_cov:.6f}, Advanced={a_cov:.6f}"
                ),
                args.preview_errors,
            )

        b_new = level_map["Beginner"].get("new_word_count")
        i_new = level_map["Intermediate"].get("new_word_count")
        a_new = level_map["Advanced"].get("new_word_count")

        if not all(isinstance(x, int) for x in (b_new, i_new, a_new)):
            add_error(
                errors,
                "new-word values",
                f"ID={article_id} has missing/non-integer New_Word_Count value(s)",
                args.preview_errors,
            )
        elif not (b_new >= i_new >= a_new):
            new_word_monotonicity_failures += 1
            title = level_map["Beginner"].get("title", "")
            add_error(
                errors,
                "new-word monotonicity",
                (
                    f"ID={article_id} title={title!r}: "
                    f"Beginner={b_new}, Intermediate={i_new}, Advanced={a_new}"
                ),
                args.preview_errors,
            )

    print("\nArticle-level monotonicity checks:")
    print(f"  Articles checked: {len(article_levels):,}")
    print(f"  Articles missing one or more levels: {missing_level_count:,}")
    print(f"  Coverage monotonicity failures: {coverage_monotonicity_failures:,}")
    print(f"  New-word monotonicity failures: {new_word_monotonicity_failures:,}")

    if args.skip_recompute:
        print("\nRecompute check: skipped")
    else:
        print("\nRecompute check:")
        print(f"  Rows checked: {recompute_rows_checked:,}")
        print(f"  Coverage mismatches: {coverage_mismatches:,}")
        print(f"  New-word-count mismatches: {new_word_mismatches:,}")

    errors = defaultdict(list, {category: messages for category, messages in errors.items() if messages})

    print("\nValidation result:")
    if not errors:
        print("  PASS: article stats match exclusive vocab bands and cumulative coverage logic.")
        return 0

    print("  FAIL: validation found issues.")
    for category, messages in errors.items():
        print(f"\n{category}:")
        for message in messages:
            print(f"  - {message}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
