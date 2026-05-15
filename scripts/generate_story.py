"""
Story Generation Script
========================
Generates a short story personalised to a student's vocabulary level and known
word list using the local llama.cpp model.

Usage
-----
    # Profile-driven (from DB)
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_story.py \\
        --user-id 1 \\
        --filename models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

    # Manual level (no DB required)
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_story.py \\
        --level intermediate \\
        --filename models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

    # Fine-tuned options
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_story.py \\
        --user-id 2 \\
        --topic "a mysterious island" \\
        --genre fantasy \\
        --challenge high \\
        --target-words 600 \\
        --filename models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

Challenge levels
----------------
  low    — almost all known words, 1-3 new words introduced gently
  medium — mostly familiar, 5-8 new words woven in naturally  (default)
  high   — up to --max-new-vocab new words, one level above current

Genres
------
  adventure, mystery, fantasy, sci-fi, slice-of-life
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.pipeline import LlamaCppGenerator
from src.rag.student_profile import StudentProfile


def _load_profile(args, db_path: str) -> tuple[str, set]:
    """Return (level, known_words) either from DB or manual --level flag."""
    if args.user_id is not None:
        from src.db.connection import get_db
        with get_db(db_path) as conn:
            profile = StudentProfile.from_db(args.user_id, conn)
        return profile.base_level, profile.known_words()
    else:
        return args.level, set()


def _print_story(result, time_elapsed) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {result.title}")
    print(f"  Level: {result.vocab_level}  |  Genre: {result.genre}  |  Challenge: {result.challenge}")
    print(f"  Time elapsed: {time_elapsed:.2f} seconds")
    print("=" * width)
    print()
    # Wrap story body for readability
    for paragraph in result.story.split("\n"):
        paragraph = paragraph.strip()
        if paragraph:
            print(textwrap.fill(paragraph, width=width))
            print()

    if result.new_vocab:
        print("-" * width)
        print("New vocabulary in this story:")
        for entry in result.new_vocab:
            word = entry.get("word", "")
            defn = entry.get("definition", "")
            print(f"  • {word}: {defn}")
        print()

    if result.challenge_note:
        print(f"Challenge note: {result.challenge_note}")

    print("=" * width)
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate a personalised story using the local llama model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # -- Input mode -----------------------------------------------------------
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--user-id", type=int,
        help="Load student profile from the database (requires --db-path)",
    )
    input_group.add_argument(
        "--level", choices=["beginner", "intermediate", "advanced"],
        help="Manually specify vocabulary level (no DB required)",
    )

    # -- Story parameters (fine-tuning knobs) ---------------------------------
    p.add_argument(
        "--topic", default=None,
        help="Optional theme or topic hint, e.g. 'a mysterious island'",
    )
    p.add_argument(
        "--genre",
        choices=["adventure", "mystery", "fantasy", "sci-fi", "slice-of-life"],
        default="adventure",
        help="Story genre (default: adventure)",
    )
    p.add_argument(
        "--challenge", choices=["low", "medium", "high"], default="medium",
        help=(
            "Vocabulary challenge level: "
            "low = 1-3 new words, medium = 5-8 new words (default), "
            "high = up to --max-new-vocab new words"
        ),
    )
    p.add_argument(
        "--target-words", type=int, default=400,
        help="Approximate story length in words (default: 400)",
    )
    p.add_argument(
        "--max-new-vocab", type=int, default=10,
        help="Hard cap on new vocabulary words when --challenge high (default: 10)",
    )

    # -- Model parameters -----------------------------------------------------
    p.add_argument(
        "--filename", default="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        help="Path to GGUF file (local path) or filename in --repo-id",
    )
    p.add_argument(
        "--repo-id", default="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        help="HuggingFace repo ID (used only if --filename is not a local path)",
    )
    p.add_argument(
        "--n-gpu-layers", type=int, default=-1,
        help="GPU layers to offload (-1 = all)",
    )
    p.add_argument(
        "--context-length", type=int, default=4096,
        help="LLM context window size in tokens (default: 4096)",
    )
    p.add_argument(
        "--cuda-visible-devices", default=None,
        help="Override CUDA_VISIBLE_DEVICES, e.g. '0'",
    )

    # -- DB -------------------------------------------------------------------
    p.add_argument(
        "--db-path", default="data/library.db",
        help="Path to the SQLite database (default: data/library.db)",
    )
    time_start = time.time()

    args = p.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    db_path = str(Path(__file__).resolve().parent.parent / args.db_path)

    # -- Load student profile -------------------------------------------------
    try:
        level, known_words = _load_profile(args, db_path)
    except ValueError as e:
        print(f"[generate_story] Error loading profile: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Level    : {level}")
    print(f"Known words loaded: {len(known_words)}")
    print(f"Genre    : {args.genre}  |  Challenge: {args.challenge}  |  Target: ~{args.target_words} words")
    if args.topic:
        print(f"Topic    : {args.topic}")
    print(f"Model    : {args.filename}")
    print("Generating story …")

    # -- Build generator ------------------------------------------------------
    generator = LlamaCppGenerator(
        repo_id=args.repo_id,
        filename=args.filename,
        n_gpu_layers=args.n_gpu_layers,
        context_length=args.context_length,
    )

    # -- Generate -------------------------------------------------------------
    result = generator.generate_story(
        vocab_level=level,
        known_words=known_words,
        topic=args.topic,
        genre=args.genre,
        challenge=args.challenge,
        target_words=args.target_words,
        max_new_vocab=args.max_new_vocab,
    )

    time_elapsed = time.time() - time_start
    _print_story(result, time_elapsed)


if __name__ == "__main__":
    main()
