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
    parser.add_argument(
        "--gguf-path",
        default=None,
        help=(
            "Path to a local GGUF model file. When provided, the model's built-in "
            "tokenizer is used instead of a Hugging Face tokenizer — no HF account "
            "or internet access required. Example: "
            "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        ),
    )
    return parser.parse_args()


class GGUFTokenizerWrapper:
    """
    Thin wrapper around llama_cpp.Llama that exposes a tokenizer-like
    __call__ interface compatible with the existing encoding loop.

    The model is loaded on CPU with a 1-token context window — this is
    enough to access the vocabulary and tokenize text without consuming
    any GPU memory.
    """

    def __init__(self, gguf_path: str, max_length: int) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise SystemExit(
                "llama-cpp-python is required to use --gguf-path. Install it with:\n"
                "  pip install llama-cpp-python"
            ) from exc

        model_path = Path(gguf_path)
        if not model_path.exists():
            raise SystemExit(f"GGUF file not found: {gguf_path}")

        self._llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=0,   # CPU-only; no VRAM needed for tokenization
            n_ctx=1,          # minimal context; we only need the tokenizer
            verbose=False,
        )
        self._max_length = max_length
        self.tokenizer_name = model_path.name

    def __call__(self, texts: list[str]) -> dict:
        """
        Tokenize a batch of texts, returning a dict with 'input_ids' and
        'attention_mask' lists-of-lists, mirroring the HF tokenizer output
        consumed by the encoding loop below.
        """
        all_ids: list[list[int]] = []
        all_masks: list[list[int]] = []

        for text in texts:
            ids = self._llm.tokenize(
                text.encode("utf-8", errors="replace"),
                add_bos=True,
                special=False,
            )[: self._max_length]
            all_ids.append(ids)
            all_masks.append([1] * len(ids))

        return {"input_ids": all_ids, "attention_mask": all_masks}


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
    gguf_path: str | None = None,
) -> None:
    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists():
        raise SystemExit(f"Input parquet not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    validate_columns(df)
    df = ensure_id_column(df)

    DEFAULT_GGUF_FALLBACK = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

    if gguf_path is not None:
        # Use the vocabulary embedded in the local GGUF file.
        # No Hugging Face account, internet access, or GPU memory required.
        tokenizer = GGUFTokenizerWrapper(gguf_path, max_length)
        resolved_tokenizer_name = tokenizer.tokenizer_name
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            # Some causal LLM tokenizers do not define a pad token by default.
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            resolved_tokenizer_name = tokenizer_name
        except OSError:
            fallback = Path(DEFAULT_GGUF_FALLBACK)
            if fallback.exists():
                print(
                    f"Warning: could not load HF tokenizer '{tokenizer_name}' "
                    f"(repo may be gated). Falling back to local GGUF: {fallback}"
                )
                tokenizer = GGUFTokenizerWrapper(str(fallback), max_length)
                resolved_tokenizer_name = tokenizer.tokenizer_name
            else:
                raise SystemExit(
                    f"Unable to load tokenizer '{tokenizer_name}'. If the repo is gated, "
                    "your account must be approved for that model.\n"
                    "Try a public tokenizer with: --tokenizer sentence-transformers/all-MiniLM-L6-v2\n"
                    f"Or place a GGUF model at '{DEFAULT_GGUF_FALLBACK}' for automatic fallback."
                )

    ids_json = []
    attn_json = []
    token_counts = []

    texts = df["clean_text"].fillna("").astype(str).tolist()

    use_gguf = isinstance(tokenizer, GGUFTokenizerWrapper)

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        if use_gguf:
            enc = tokenizer(batch)
        else:
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
    out_df["llm_tokenizer_name"] = resolved_tokenizer_name
    out_df["llm_input_ids_json"] = ids_json
    out_df["llm_attention_mask_json"] = attn_json
    out_df["llm_token_count"] = token_counts

    out_df.to_parquet(out_path, index=False)
    print(f"Saved tokenized articles to {out_path}")
    print(f"Rows: {len(out_df)}")
    print(f"Tokenizer: {resolved_tokenizer_name}")


def main() -> None:
    args = parse_args()
    tokenize_articles(
        input_path=args.input,
        output_path=args.output,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gguf_path=args.gguf_path,
    )


if __name__ == "__main__":
    main()