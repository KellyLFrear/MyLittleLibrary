"""
Build the FAISS index from the article_stats.jsonl produced by analyze_articles.

Input format — JSON lines (one article per line), as written by analyze_articles.py:
    {"id": "...", "title": "...", "text": "...",
     "coverage_ratio": {"beginner": 0.72, "intermediate": 0.91, "advanced": 0.97},
     "new_words":      {"beginner": [...], "intermediate": [...], "advanced": [...]},
     "readability_score": 58.3}

Usage
-----
Default (reads outputs/article_stats.jsonl, GPU):
    python scripts/build_index.py

Custom paths:
    python scripts/build_index.py --articles outputs/article_stats.jsonl --index-dir data/faiss_index

Larger batch size for high-VRAM GPUs:
    python scripts/build_index.py --batch-size 512
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.chunker import chunk_article
from src.embeddings.embedder import ArticleEmbedder
from src.embeddings.vector_store import FAISSVectorStore


def main() -> None:
    p = argparse.ArgumentParser(description="Embed articles and build a FAISS index.")
    p.add_argument("--articles",   default="outputs/article_stats.jsonl",
                   help="Path to .jsonl article file (default: outputs/article_stats.jsonl)")
    p.add_argument("--index-dir",  default="data/faiss_index",
                   help="Directory where the FAISS index will be saved")
    p.add_argument("--model",      default="all-MiniLM-L6-v2",
                   help="Sentence-transformer model name")
    p.add_argument("--chunk-size", type=int, default=400,
                   help="Maximum words per chunk")
    p.add_argument("--overlap",    type=int, default=50,
                   help="Word overlap between consecutive chunks")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Embedding batch size (increase for higher-VRAM GPUs)")
    p.add_argument("--device",     default="cuda",
                   help="Torch device: 'cuda' (default) or 'cpu'")
    p.add_argument("--no-gpu-index", action="store_true",
                   help="Disable GPU acceleration for the FAISS index (embedder still uses GPU)")
    args = p.parse_args()

    # ── Load articles ─────────────────────────────────────────────────────────
    path = Path(args.articles)
    if not path.exists():
        sys.exit(f"[build_index] Article file not found: {path}")

    articles = [json.loads(line) for line in path.open() if line.strip()]
    print(f"[build_index] Loaded {len(articles):,} articles from {path}")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    all_chunks = []
    for article in articles:
        all_chunks.extend(chunk_article(article, args.chunk_size, args.overlap))
    print(f"[build_index] Created {len(all_chunks):,} chunks "
          f"(size={args.chunk_size}, overlap={args.overlap})")

    # ── Embed ─────────────────────────────────────────────────────────────────
    embedder = ArticleEmbedder(model_name=args.model, device=args.device)
    print(f"[build_index] Embedding with '{args.model}' on {args.device} …")

    use_gpu = not args.no_gpu_index
    store = FAISSVectorStore(embedding_dim=embedder.embedding_dim, use_gpu=use_gpu)

    for i in range(0, len(all_chunks), args.batch_size):
        batch = all_chunks[i : i + args.batch_size]
        embs  = embedder.embed_chunks(batch, batch_size=args.batch_size, show_progress=False)
        store.add(batch, embs)
        if (i // args.batch_size) % 10 == 0 and i > 0:
            print(f"  … {i:,} / {len(all_chunks):,} chunks embedded")

    # ── Save ──────────────────────────────────────────────────────────────────
    store.save(args.index_dir)
    print(f"[build_index] Done. Index saved to '{args.index_dir}/'")


if __name__ == "__main__":
    main()
