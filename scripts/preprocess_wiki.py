#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
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


# Function To Clean Text By Normalizing Unicode, Removing Extra Whitespace, And Stripping Leading/Trailing Spaces

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Function To Identify Disambiguation Pages Based On Title And Text Content

def is_disambiguation(title: str, text: str) -> bool:
    t = (title or "").lower()
    x = (text or "").lower()
    return "(disambiguation)" in t or "may refer to:" in x


# Function To Identify Stub Articles Based On Word Count (Simple Heuristic)

def is_stub(text: str, min_words: int = 150) -> bool:
    return len((text or "").split()) < min_words


# Main Function To Preprocess Wikipedia Data By Cleaning Text, Removing Disambiguation Pages And Stubs, And Saving The Cleaned Data

def preprocess_wiki(input_path: str, output_path: str, min_words: int = 150) -> None:
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

    df["clean_text"] = df["text"].apply(clean_text)

    df = df[
        (~df.apply(lambda row: is_disambiguation(row["title"], row["clean_text"]), axis=1))
        & (~df["clean_text"].apply(lambda x: is_stub(x, min_words=min_words)))
    ].copy()

    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} cleaned articles to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    preprocess_wiki(args.input, args.output, args.min_words)
