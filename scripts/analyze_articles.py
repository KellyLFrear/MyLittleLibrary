#!/usr/bin/env python3
"""
Analyze cleaned Wikipedia articles against beginner/intermediate/advanced vocabulary lists.

Features:
- configurable parquet input path via --input
- configurable JSONL output path via --output
- optional vocab file overrides for each band
- configurable coverage window for candidate filtering
- validates required columns before running

Default behavior matches the original project layout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import textstat


DEFAULT_INPUT = "data/processed/wiki_clean.parquet"
DEFAULT_OUTPUT = "outputs/article_stats.jsonl"
DEFAULT_VOCABS = {
    "Beginner": "data/vocab/beginner_1000.txt",
    "Intermediate": "data/vocab/intermediate_3000.txt",
    "Advanced": "data/vocab/advanced_6000.txt",
}

WORD_RE = re.compile(r"[a-zA-Z']+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cleaned articles against vocabulary lists and write JSONL stats."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input cleaned parquet file. Default: data/processed/wiki_clean.parquet",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output JSONL path. Default: outputs/article_stats.jsonl",
    )
    parser.add_argument(
        "--beginner-vocab",
        default=DEFAULT_VOCABS["Beginner"],
        help="Path to beginner vocab TXT file.",
    )
    parser.add_argument(
        "--intermediate-vocab",
        default=DEFAULT_VOCABS["Intermediate"],
        help="Path to intermediate vocab TXT file.",
    )
    parser.add_argument(
        "--advanced-vocab",
        default=DEFAULT_VOCABS["Advanced"],
        help="Path to advanced vocab TXT file.",
    )
    parser.add_argument(
        "--coverage-min",
        type=float,
        default=0.90,
        help="Minimum coverage ratio for candidate articles. Default: 0.90",
    )
    parser.add_argument(
        "--coverage-max",
        type=float,
        default=0.97,
        help="Maximum coverage ratio for candidate articles. Default: 0.97",
    )
    return parser.parse_args()


# Function That Loads a Vocabulary File And Returns A Set Of Lowercase Words

def load_word_list(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


# Function To Tokenize Text Into Lowercase Words Using Regex

def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


# Function To Compute Flesch-Kincaid Grade Level Using textstat

def flesch_kincaid_grade(text: str) -> float:
    return textstat.flesch_kincaid_grade(text)


def validate_columns(df: pd.DataFrame) -> None:
    required = {"title", "clean_text"}
    missing = required - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise SystemExit(
            f"Input parquet is missing required column(s): {missing_str}. "
            "Expected at least: title, clean_text"
        )


def ensure_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "id" in df.columns:
        return df
    df = df.copy()
    df["id"] = range(len(df))
    return df


def iter_vocab_paths(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "Beginner": args.beginner_vocab,
        "Intermediate": args.intermediate_vocab,
        "Advanced": args.advanced_vocab,
    }


# Function To Analyze Articles And Save Results As JSON Lines

def analyze_articles(
    input_path: str,
    output_path: str,
    vocab_paths: Dict[str, str],
    coverage_min: float,
    coverage_max: float,
) -> None:
    if coverage_min > coverage_max:
        raise SystemExit("--coverage-min cannot be greater than --coverage-max")

    input_file = Path(input_path)
    if not input_file.exists():
        raise SystemExit(f"Input parquet does not exist: {input_path}")

    for level, vocab_path in vocab_paths.items():
        if not Path(vocab_path).exists():
            raise SystemExit(f"{level} vocab file does not exist: {vocab_path}")

    print("Running analysis for all levels...")

    df = pd.read_parquet(input_file)
    validate_columns(df)
    df = ensure_id_column(df)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for level, vocab_path in vocab_paths.items():
            print(f"Processing {level} level...")
            known_words = load_word_list(vocab_path)

            for _, row in df.iterrows():
                text = row["clean_text"]
                tokens = tokenize_words(text)

                if not tokens:
                    continue

                total_words = len(tokens)
                unique_words = len(set(tokens))
                known_count = sum(1 for w in tokens if w in known_words)
                coverage_ratio = known_count / total_words

                article_new_words = sorted({w for w in tokens if w not in known_words})
                new_word_count = len(article_new_words)
                readability = flesch_kincaid_grade(text)
                is_candidate = coverage_min <= coverage_ratio <= coverage_max

                record = {
                    "ID": row["id"],
                    "Title": row["title"],
                    "Level": level,
                    "Total_Words": total_words,
                    "Unique_Words": unique_words,
                    "Coverage_Ratio": coverage_ratio,
                    "New_Word_Count": new_word_count,
                    "New_Words": ", ".join(article_new_words[:30]),
                    "Flesch_Kincaid_Grade": readability,
                    "Candidate": is_candidate,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved JSONL output to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    analyze_articles(
        input_path=args.input,
        output_path=args.output,
        vocab_paths=iter_vocab_paths(args),
        coverage_min=args.coverage_min,
        coverage_max=args.coverage_max,
    )
