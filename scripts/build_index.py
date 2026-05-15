"""
Build a FAISS index from article JSONL data.

This version computes vocabulary metadata at the CHUNK level.

Why this matters:
    The article-level analyzer produces coverage for the whole article.
    But recommendations retrieve chunks/passages. A chunk can be much easier
    or harder than the full article. Therefore, each ArticleChunk should store
    its own coverage_ratio and new_words metadata.

Supported input schemas
-----------------------
1) Compact article rows:
    {"id": "...", "title": "...", "text": "...",
     "coverage_ratio": {"beginner": 0.72, "intermediate": 0.91, "advanced": 0.97},
     "new_words":      {"beginner": [...], "intermediate": [...], "advanced": [...]},
     "readability_score": 58.3}

2) analyze_articles rows:
    {"ID": "...", "Title": "...", "Text": "...", "Level": "Beginner",
     "Coverage_Ratio": 0.72, "New_Words": "word1, word2", ...}

Rows in schema (2) are normalized into one article object per ID before chunking.
After chunking, chunk-level vocabulary metadata is recomputed from the chunk text.

Usage
-----
Default:
    python scripts/build_index.py

1M index:
    python scripts/build_index.py \\
      --articles outputs/article_stats_1m.jsonl \\
      --index-dir data/faiss_index_1m_chunklevel \\
      --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.chunker import ArticleChunk, chunk_article
from src.embeddings.embedder import ArticleEmbedder
from src.embeddings.vector_store import FAISSVectorStore


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


# ── Text / vocab normalization ────────────────────────────────────────────────

def _normalize_word(word: str) -> str:
    """Normalize a word for vocabulary matching."""
    word = str(word).strip().lower()
    if word.endswith("'s"):
        word = word[:-2]
    return word.strip("'")


def _candidate_word_forms(word: str) -> set[str]:
    """
    Return simple word-family candidates.

    This avoids treating common inflections as unknown when the base word is known:
      planets  -> planet
      cheered  -> cheer
      clapped  -> clap
      danced   -> dance
      bringing -> bring
      making   -> make
    """
    word = _normalize_word(word)
    if not word:
        return set()

    forms = {word}

    # stories -> story
    if len(word) > 4 and word.endswith("ies"):
        forms.add(word[:-3] + "y")

    # planets -> planet
    if len(word) > 3 and word.endswith("s"):
        forms.add(word[:-1])

    # boxes -> box, buses -> bus
    if len(word) > 4 and word.endswith("es"):
        forms.add(word[:-2])

    # cheered -> cheer, clapped -> clap, danced -> dance
    if len(word) > 4 and word.endswith("ed"):
        base = word[:-2]
        forms.add(base)
        forms.add(base + "e")

        if len(base) > 2 and base[-1] == base[-2]:
            forms.add(base[:-1])

    # bringing -> bring, running -> run, making -> make
    if len(word) > 5 and word.endswith("ing"):
        base = word[:-3]
        forms.add(base)
        forms.add(base + "e")

        if len(base) > 2 and base[-1] == base[-2]:
            forms.add(base[:-1])

    # brighter -> bright, fastest -> fast
    if len(word) > 4 and word.endswith("er"):
        forms.add(word[:-2])
    if len(word) > 5 and word.endswith("est"):
        forms.add(word[:-3])

    # slowly -> slow
    if len(word) > 4 and word.endswith("ly"):
        forms.add(word[:-2])

    return {form for form in forms if form}


def _tokenize_words(text: str) -> list[str]:
    """Tokenize text into normalized word tokens."""
    return [
        token
        for raw in WORD_RE.findall(text or "")
        if (token := _normalize_word(raw))
    ]


def _is_known_word(token: str, known_set: set[str]) -> bool:
    """Return True if token or a simple word-family form is in known_set."""
    return any(form in known_set for form in _candidate_word_forms(token))


# ── Vocab file loading ────────────────────────────────────────────────────────

def _load_vocab_file(path: str | Path) -> set[str]:
    """
    Load vocabulary from either:
      - .txt file with one word per line
      - .csv file with a 'word' column, or else first column
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {p}")

    words: set[str] = set()

    if p.suffix.lower() == ".csv":
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                word_col = "word" if "word" in reader.fieldnames else reader.fieldnames[0]
                for row in reader:
                    word = _normalize_word(row.get(word_col, ""))
                    if word:
                        words.add(word)
    else:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Allows either "word" or "word extra metadata" style lines.
                word = _normalize_word(line.split()[0])
                if word:
                    words.add(word)

    if not words:
        raise ValueError(f"No vocabulary words loaded from: {p}")

    return words


def _build_cumulative_vocab_sets(
    beginner_path: str | Path,
    intermediate_path: str | Path,
    advanced_path: str | Path,
) -> dict[str, set[str]]:
    """
    Build cumulative known-word sets from exclusive vocab bands.

    beginner     = beginner band
    intermediate = beginner + intermediate bands
    advanced     = beginner + intermediate + advanced bands
    """
    beginner_band = _load_vocab_file(beginner_path)
    intermediate_band = _load_vocab_file(intermediate_path)
    advanced_band = _load_vocab_file(advanced_path)

    beginner_known = set(beginner_band)
    intermediate_known = beginner_known | intermediate_band
    advanced_known = intermediate_known | advanced_band

    print("[build_index] Vocabulary band sizes:")
    print(f"  Beginner band:     {len(beginner_band):,}")
    print(f"  Intermediate band: {len(intermediate_band):,}")
    print(f"  Advanced band:     {len(advanced_band):,}")
    print("[build_index] Cumulative known-word sizes:")
    print(f"  Beginner:          {len(beginner_known):,}")
    print(f"  Intermediate:      {len(intermediate_known):,}")
    print(f"  Advanced:          {len(advanced_known):,}")

    return {
        "beginner": beginner_known,
        "intermediate": intermediate_known,
        "advanced": advanced_known,
    }


# ── Article normalization ─────────────────────────────────────────────────────

def _parse_new_words(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [w.strip() for w in value.split(",") if w.strip()]
    return []


def normalize_articles(raw_rows: list[dict]) -> list[dict]:
    """Normalize supported JSONL layouts into the chunker's expected article schema."""
    if not raw_rows:
        return []

    # Already in expected schema.
    if "text" in raw_rows[0] and "id" in raw_rows[0]:
        return raw_rows

    # analyze_articles layout: one row per (article, level) with uppercase keys.
    if "ID" in raw_rows[0] and "Level" in raw_rows[0]:
        merged: "OrderedDict[str, dict]" = OrderedDict()

        for row in raw_rows:
            aid = str(row.get("ID", "")).strip()
            if not aid:
                continue

            article = merged.setdefault(
                aid,
                {
                    "id": aid,
                    "title": str(row.get("Title", "")),
                    "text": str(row.get("Text", "") or ""),
                    "coverage_ratio": {},
                    "new_words": {},
                    "readability_score": float(row.get("Flesch_Kincaid_Grade", 0.0) or 0.0),
                },
            )

            if not article["text"] and row.get("Text"):
                article["text"] = str(row["Text"])

            level = str(row.get("Level", "")).strip().lower()
            if level:
                cov = row.get("Coverage_Ratio")
                if isinstance(cov, (int, float)):
                    article["coverage_ratio"][level] = float(cov)
                article["new_words"][level] = _parse_new_words(row.get("New_Words"))

        return list(merged.values())

    raise SystemExit(
        "[build_index] Unsupported article JSONL schema. Expected either compact article rows "
        "with id/title/text or analyze_articles rows with ID/Title/Level/Text."
    )


# ── Chunk-level metadata ──────────────────────────────────────────────────────

def _coverage_and_new_words_for_level(
    tokens: list[str],
    known_set: set[str],
    *,
    max_new_words: int,
) -> tuple[float, list[str]]:
    """
    Compute token-level known-word coverage and a unique unknown-word list.

    Coverage counts repeated tokens.
    new_words stores unique unknown words in first-seen order.
    """
    if not tokens:
        return 0.0, []

    known_count = 0
    new_words: list[str] = []
    seen_new: set[str] = set()

    for token in tokens:
        if _is_known_word(token, known_set):
            known_count += 1
            continue

        if token not in seen_new:
            seen_new.add(token)
            if len(new_words) < max_new_words:
                new_words.append(token)

    coverage = known_count / len(tokens)
    return coverage, new_words


def recompute_chunk_vocab_metadata(
    chunk: ArticleChunk,
    cumulative_vocab: dict[str, set[str]],
    *,
    max_new_words: int,
) -> None:
    """
    Replace inherited article-level metadata with true chunk-level metadata.
    """
    tokens = _tokenize_words(chunk.text)

    coverage_ratio: Dict[str, float] = {}
    new_words: Dict[str, List[str]] = {}

    for level, known_set in cumulative_vocab.items():
        coverage, unknown_words = _coverage_and_new_words_for_level(
            tokens,
            known_set,
            max_new_words=max_new_words,
        )
        coverage_ratio[level] = coverage
        new_words[level] = unknown_words

    chunk.coverage_ratio = coverage_ratio
    chunk.new_words = new_words


def recompute_all_chunk_vocab_metadata(
    chunks: list[ArticleChunk],
    cumulative_vocab: dict[str, set[str]],
    *,
    max_new_words: int,
) -> None:
    """
    Compute chunk-level coverage/new_words for every chunk in-place.
    """
    for i, chunk in enumerate(chunks, start=1):
        recompute_chunk_vocab_metadata(
            chunk,
            cumulative_vocab,
            max_new_words=max_new_words,
        )

        if i % 100_000 == 0:
            print(f"  … chunk-level vocab computed for {i:,} / {len(chunks):,} chunks")


def _print_chunk_metadata_sanity(chunks: list[ArticleChunk]) -> None:
    if not chunks:
        return

    levels = ("beginner", "intermediate", "advanced")
    print("[build_index] Chunk-level coverage sanity check:")

    for level in levels:
        values = [
            float(c.coverage_ratio.get(level, 0.0))
            for c in chunks
            if c.coverage_ratio.get(level) is not None
        ]
        if not values:
            print(f"  {level}: no values")
            continue

        avg = sum(values) / len(values)
        print(
            f"  {level}: min={min(values):.4f} "
            f"avg={avg:.4f} max={max(values):.4f}"
        )

    first = chunks[0]
    print("[build_index] First chunk sample:")
    print(f"  title: {first.title}")
    print(f"  chunk_id: {first.chunk_id}")
    print(f"  coverage: {first.coverage_ratio}")
    print(
        "  intermediate new words sample: "
        f"{first.new_words.get('intermediate', [])[:15]}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Embed articles and build a FAISS index.")
    p.add_argument(
        "--articles",
        default="outputs/article_stats.jsonl",
        help="Path to .jsonl article file",
    )
    p.add_argument(
        "--index-dir",
        default="data/faiss_index",
        help="Directory where the FAISS index will be saved",
    )
    p.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Maximum words per chunk",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Word overlap between consecutive chunks",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Embedding batch size",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device: 'cuda' or 'cpu'",
    )
    p.add_argument(
        "--no-gpu-index",
        action="store_true",
        help="Disable GPU acceleration for FAISS index; embedder can still use GPU",
    )
    p.add_argument(
        "--beginner-vocab",
        default="data/vocab/beginner_1000.txt",
        help="Beginner exclusive vocabulary file",
    )
    p.add_argument(
        "--intermediate-vocab",
        default="data/vocab/intermediate_3000.txt",
        help="Intermediate exclusive vocabulary file",
    )
    p.add_argument(
        "--advanced-vocab",
        default="data/vocab/advanced_6000.txt",
        help="Advanced exclusive vocabulary file",
    )
    p.add_argument(
        "--max-new-words",
        type=int,
        default=50,
        help="Maximum unique unknown words stored per chunk per level",
    )
    args = p.parse_args()

    # ── Load vocabulary ───────────────────────────────────────────────────────
    cumulative_vocab = _build_cumulative_vocab_sets(
        args.beginner_vocab,
        args.intermediate_vocab,
        args.advanced_vocab,
    )

    # ── Load articles ─────────────────────────────────────────────────────────
    path = Path(args.articles)
    if not path.exists():
        sys.exit(f"[build_index] Article file not found: {path}")

    raw_rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    articles = normalize_articles(raw_rows)

    missing_text = sum(1 for a in articles if not str(a.get("text", "")).strip())

    print(
        f"[build_index] Loaded {len(raw_rows):,} row(s) and normalized to "
        f"{len(articles):,} article(s) from {path}"
    )
    if missing_text:
        print(f"[build_index] Warning: {missing_text:,} article(s) have empty text and will be skipped")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    all_chunks: list[ArticleChunk] = []
    for article in articles:
        all_chunks.extend(chunk_article(article, args.chunk_size, args.overlap))

    print(
        f"[build_index] Created {len(all_chunks):,} chunks "
        f"(size={args.chunk_size}, overlap={args.overlap})"
    )

    # ── Recompute chunk-level vocabulary metadata ─────────────────────────────
    print("[build_index] Computing chunk-level coverage/new_words metadata …")
    recompute_all_chunk_vocab_metadata(
        all_chunks,
        cumulative_vocab,
        max_new_words=args.max_new_words,
    )
    _print_chunk_metadata_sanity(all_chunks)

    # ── Embed ─────────────────────────────────────────────────────────────────
    embedder = ArticleEmbedder(model_name=args.model, device=args.device)
    print(f"[build_index] Embedding with '{args.model}' on {args.device} …")

    use_gpu = not args.no_gpu_index
    store = FAISSVectorStore(embedding_dim=embedder.embedding_dim, use_gpu=use_gpu)

    for i in range(0, len(all_chunks), args.batch_size):
        batch = all_chunks[i : i + args.batch_size]
        embs = embedder.embed_chunks(
            batch,
            batch_size=args.batch_size,
            show_progress=False,
        )
        store.add(batch, embs)

        if (i // args.batch_size) % 10 == 0 and i > 0:
            print(f"  … {i:,} / {len(all_chunks):,} chunks embedded")

    # ── Save ──────────────────────────────────────────────────────────────────
    store.save(args.index_dir)
    print(f"[build_index] Done. Index saved to '{args.index_dir}/'")


if __name__ == "__main__":
    main()