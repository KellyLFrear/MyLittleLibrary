#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Wikipedia sample without forcing a full local fetch for small test runs."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="How many articles to save.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/wiki_sample.parquet",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling the streaming dataset.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=10000,
        help="Shuffle buffer size for the streaming dataset.",
    )
    parser.add_argument(
        "--config",
        default="20231101.en",
        help="Wikipedia config/version to use from Hugging Face datasets.",
    )
    return parser.parse_args()



def download_wiki(sample_size: int, output_path: str, seed: int, buffer_size: int, config: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print("Downloading Wikipedia dataset with streaming...")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        config,
        split="train",
        streaming=True,
    )

    shuffled = dataset.shuffle(seed=seed, buffer_size=max(buffer_size, sample_size))
    rows = []

    for item in itertools.islice(shuffled, sample_size):
        rows.append(
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "text": item.get("text"),
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)

    print(f"Downloaded {len(df)} samples from Wikipedia and saved to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    download_wiki(
        sample_size=args.sample_size,
        output_path=args.output,
        seed=args.seed,
        buffer_size=args.buffer_size,
        config=args.config,
    )
