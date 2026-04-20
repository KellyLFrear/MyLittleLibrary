#!/usr/bin/env python3
from __future__ import annotations

"""
preprocess_wiki.py

Purpose:
- Load the raw Wikipedia parquet file created by download_wiki.py
- Normalize text formatting
- Remove disambiguation pages
- Remove very short stub articles
- Save a cleaner parquet file for downstream analysis

Why this matters:
The raw sample may contain pages that are not useful for reading-level work,
such as "(disambiguation)" pages or very short articles with too little text.
"""

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """
    Read command-line options for preprocessing.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess Wikipedia parquet by normalizing text, removing "
            "disambiguation pages and short stub articles, and writing a cleaned parquet."
        )
    )
    parser.add_argument(
        "--input",
        default="data/raw/wiki_sample.parquet",
        help="Input parquet file path. Default: data/raw/wiki_sample.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/processed/wiki_clean.parquet",
        help="Output parquet file path. Default: data/processed/wiki_clean.parquet",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=150,
        help="Minimum word count required to keep an article. Default: 150",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    """
    Normalize article text.

    What it does:
    - Returns an empty string if the value is not text
    - Normalizes Unicode so characters are represented consistently
    - Collapses repeated whitespace into single spaces
    - Trims leading/trailing spaces
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_disambiguation(title: str, text: str) -> bool:
    """
    Identify common Wikipedia disambiguation pages.

    Heuristic used here:
    - title contains "(disambiguation)"
    - text contains the phrase "may refer to:"
    """
    t = (title or "").lower()
    x = (text or "").lower()
    return "(disambiguation)" in t or "may refer to:" in x


def is_stub(text: str, min_words: int = 150) -> bool:
    """
    Treat very short articles as stubs.

    A stub here simply means the cleaned article has fewer than min_words.
    """
    return len((text or "").split()) < min_words


def preprocess_wiki(input_path: str, output_path: str, min_words: int = 150) -> None:
    """
    Run the full preprocessing step.

    Steps:
    1. Check that the input parquet exists.
    2. Load the parquet into a DataFrame.
    3. Verify required columns exist.
    4. Create a cleaned version of the text.
    5. Remove disambiguation pages and short stubs.
    6. Add a word_count column.
    7. Save the result to a new parquet file.
    """
    print("Running Preprocessing...")

    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)

    required_columns = {"title", "text"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input parquet is missing required columns: {missing_text}")

    # Create a cleaned text column that later scripts will use.
    df["clean_text"] = df["text"].apply(clean_text)

    # Keep only rows that are not disambiguation pages and not too short.
    df = df[
        (~df.apply(lambda row: is_disambiguation(row["title"], row["clean_text"]), axis=1))
        & (~df["clean_text"].apply(lambda x: is_stub(x, min_words=min_words)))
    ].copy()

    # Store a simple word count for convenience/debugging.
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} cleaned articles to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    preprocess_wiki(args.input, args.output, args.min_words)
