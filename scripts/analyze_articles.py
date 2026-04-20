#!/usr/bin/env python3
"""
analyze_articles.py

Purpose:
- Compare each tokenized Wikipedia article against the project vocabulary lists
- Compute article coverage for Beginner / Intermediate / Advanced
- Estimate readability with Flesch-Kincaid grade
- Record optional LLM tokenizer metadata if present in the parquet input
- Mark whether an article falls inside a chosen coverage window
- Write one JSON record per article-level pair to article_stats.jsonl

Important idea:
Each tokenized article is analyzed three separate times:
1) once using the beginner vocabulary
2) once using the intermediate vocabulary
3) once using the advanced vocabulary

That is why you saw the same number of rows for all three levels.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict

import pandas as pd
import textstat


DEFAULT_INPUT = "data/processed/wiki_tokenized.parquet"
DEFAULT_OUTPUT = "outputs/article_stats.jsonl"
DEFAULT_VOCABS = {
    "Beginner": "data/vocab/beginner_1000.txt",
    "Intermediate": "data/vocab/intermediate_3000.txt",
    "Advanced": "data/vocab/advanced_6000.txt",
}

# Basic word tokenizer pattern for the vocabulary-coverage portion of the analysis:
# - keeps alphabetic words
# - allows apostrophes inside words
#
# Note:
# This script still uses word-level tokens for vocabulary coverage because the
# project vocab lists are stored as whole words. LLM subword tokenization is
# handled earlier in the pipeline by tokenize_articles.py.
WORD_RE = re.compile(r"[a-zA-Z']+")


def parse_args() -> argparse.Namespace:
    """
    Read command-line arguments for the analysis step.
    """
    parser = argparse.ArgumentParser(
        description="Analyze tokenized articles against vocabulary lists and write JSONL stats."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input tokenized parquet file. Default: data/processed/wiki_tokenized.parquet",
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


def load_word_list(path: str) -> set[str]:
    """
    Load a one-word-per-line vocabulary text file into a lowercase set.

    Using a set makes membership checks fast:
    `word in known_words`
    """
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def tokenize_words(text: str) -> list[str]:
    """
    Split article text into lowercase word tokens using regex.
    """
    return WORD_RE.findall(text.lower())


def flesch_kincaid_grade(text: str) -> float:
    """
    Compute readability using textstat's Flesch-Kincaid grade level.
    """
    return textstat.flesch_kincaid_grade(text)


def validate_columns(df: pd.DataFrame) -> None:
    """
    Make sure the input parquet has the columns required by this script.

    Required columns:
    - title
    - clean_text

    Optional columns that may come from tokenize_articles.py:
    - llm_token_count
    - llm_tokenizer_name
    - llm_input_ids_json
    - llm_attention_mask_json
    """
    required = {"title", "clean_text"}
    missing = required - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise SystemExit(
            f"Input parquet is missing required column(s): {missing_str}. "
            "Expected at least: title, clean_text"
        )


def ensure_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee there is an id column.

    If the source parquet already has `id`, keep it.
    Otherwise generate a simple sequential id.
    """
    if "id" in df.columns:
        return df
    df = df.copy()
    df["id"] = range(len(df))
    return df


def iter_vocab_paths(args: argparse.Namespace) -> Dict[str, str]:
    """
    Package the three vocabulary paths into a single dictionary.
    """
    return {
        "Beginner": args.beginner_vocab,
        "Intermediate": args.intermediate_vocab,
        "Advanced": args.advanced_vocab,
    }


def analyze_articles(
    input_path: str,
    output_path: str,
    vocab_paths: Dict[str, str],
    coverage_min: float,
    coverage_max: float,
) -> None:
    """
    Analyze each article against each vocabulary list.

    For every (article, level) pair this script computes:
    - Total_Words: total number of word-level tokens
    - Unique_Words: number of distinct word-level tokens
    - Coverage_Ratio: fraction of tokens already in the known-word list
    - New_Word_Count: number of unique unknown words
    - New_Words: a preview of up to 30 unknown words
    - Flesch_Kincaid_Grade: readability estimate
    - LLM_Token_Count: number of subword tokens if present in the tokenized parquet
    - Tokenizer_Name: tokenizer name used earlier in the pipeline if present
    - Word_to_LLM_Token_Ratio: rough comparison between word count and LLM token count
    - Candidate: whether the coverage ratio falls inside the chosen window
    """
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
        # Loop once per level. This is why all three levels end up with the same
        # number of rows: every article is evaluated for every level.
        for level, vocab_path in vocab_paths.items():
            print(f"Processing {level} level...")
            known_words = load_word_list(vocab_path)

            # Loop through all cleaned/tokenized articles.
            for _, row in df.iterrows():
                text = row["clean_text"]
                tokens = tokenize_words(text)

                # Skip empty rows after tokenization.
                if not tokens:
                    continue

                total_words = len(tokens)
                unique_words = len(set(tokens))

                # Count how many running-word tokens appear in the current known-word list.
                known_count = sum(1 for w in tokens if w in known_words)
                coverage_ratio = known_count / total_words

                # Collect unique words that are NOT known for the current level.
                article_new_words = sorted({w for w in tokens if w not in known_words})
                new_word_count = len(article_new_words)

                readability = flesch_kincaid_grade(text)

                # Newly added:
                # If tokenize_articles.py was run earlier, the parquet may already
                # contain the number of LLM subword tokens for this article.
                # Keep it optional so this script can still run on older parquet files.
                llm_token_count = None
                if "llm_token_count" in df.columns and pd.notna(row["llm_token_count"]):
                    llm_token_count = int(row["llm_token_count"])

                # Newly added:
                # Record which tokenizer created the subword tokens when that metadata
                # is available in the input parquet.
                tokenizer_name = ""
                if "llm_tokenizer_name" in df.columns and pd.notna(row["llm_tokenizer_name"]):
                    tokenizer_name = str(row["llm_tokenizer_name"])

                # Newly added:
                # This ratio is a quick diagnostic showing how many word-level tokens
                # correspond to each LLM subword token count for the same article.
                word_to_llm_ratio = None
                if llm_token_count:
                    word_to_llm_ratio = total_words / llm_token_count

                # Candidate means the article falls inside the target "sweet spot"
                # for known-word coverage.
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
                    "LLM_Token_Count": llm_token_count,
                    "Tokenizer_Name": tokenizer_name,
                    "Word_to_LLM_Token_Ratio": word_to_llm_ratio,
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
