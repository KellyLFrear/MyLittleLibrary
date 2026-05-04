#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


DEFAULT_INPUT = "data/processed/wiki_clean.parquet"
DEFAULT_OUTPUT = "data/processed/wiki_tokenized.parquet"
DEFAULT_TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize cleaned articles with a Hugging Face / LLaMA tokenizer."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input cleaned parquet")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output tokenized parquet")
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="HF model id or local tokenizer directory"
    )
    parser.add_argument("--max-length", type=int, default=2048, help="Max LLM tokens per article")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    return parser.parse_args()


def ensure_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "id" in df.columns:
        return df
    df = df.copy()
    df["id"] = range(len(df))
    return df


def validate_columns(df: pd.DataFrame) -> None:
    required = {"title", "clean_text"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(sorted(missing))}")


def tokenize_articles(
    input_path: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    max_length: int = 2048,
    batch_size: int = 32,
) -> None:
    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists():
        raise SystemExit(f"Input parquet not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    validate_columns(df)
    df = ensure_id_column(df)

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    except OSError as exc:
        raise SystemExit(
            "Unable to load tokenizer. If the tokenizer repo is gated, your account must be "
            "approved for that specific model. Try a public tokenizer with: "
            "--tokenizer sentence-transformers/all-MiniLM-L6-v2"
        ) from exc

    # Some causal LLM tokenizers do not define a pad token by default.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    ids_json = []
    attn_json = []
    token_counts = []

    texts = df["clean_text"].fillna("").astype(str).tolist()

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        enc = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )

        for input_ids, attention_mask in zip(enc["input_ids"], enc["attention_mask"]):
            ids_json.append(json.dumps(input_ids))
            attn_json.append(json.dumps(attention_mask))
            token_counts.append(len(input_ids))

    out_df = df.copy()
    out_df["llm_tokenizer_name"] = tokenizer_name
    out_df["llm_input_ids_json"] = ids_json
    out_df["llm_attention_mask_json"] = attn_json
    out_df["llm_token_count"] = token_counts

    out_df.to_parquet(out_path, index=False)
    print(f"Saved tokenized articles to {out_path}")
    print(f"Rows: {len(out_df)}")
    print(f"Tokenizer: {tokenizer_name}")


def main() -> None:
    args = parse_args()
    tokenize_articles(
        input_path=args.input,
        output_path=args.output,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()