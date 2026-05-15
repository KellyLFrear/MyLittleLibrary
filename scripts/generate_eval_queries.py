"""
Generate evaluation queries from the 1M-article FAISS index.

For each seed topic phrase, embeds the phrase with the same model used during
retrieval, searches the index, and records the top-N unique article IDs as the
`relevant_ids` ground truth.  Output format is identical to eval_queries.json.

Usage
-----
    python scripts/generate_eval_queries.py
    python scripts/generate_eval_queries.py \\
        --index-dir    data/faiss_index_1m_chunklevel \\
        --output       data/eval_queries.json \\
        --top-relevant 5 \\
        --device       cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.embedder import ArticleEmbedder
from src.embeddings.vector_store import FAISSVectorStore


# ── Seed queries ──────────────────────────────────────────────────────────────
# (query text, vocab level)
# Edit or extend this list to produce different evaluation sets.
SEED_QUERIES: List[tuple] = [
    # beginner
    ("animals and wildlife habitats",               "beginner"),
    ("weather and seasons around the world",        "beginner"),
    ("famous landmarks and tourist attractions",    "beginner"),
    ("popular sports and Olympic games",            "beginner"),
    ("food and cooking traditions",                 "beginner"),
    # intermediate
    ("space exploration and astronomy missions",    "intermediate"),
    ("political leaders and government systems",    "intermediate"),
    ("world war history and military battles",      "intermediate"),
    ("environmental issues and climate change",     "intermediate"),
    ("music bands and album releases",              "intermediate"),
    # advanced
    ("economics and financial market theory",       "advanced"),
    ("philosophy and ethics debates",               "advanced"),
    ("biochemistry and molecular biology",          "advanced"),
    ("constitutional law and court decisions",      "advanced"),
    ("linguistics and language evolution",          "advanced"),
]


def main() -> None:
    p = argparse.ArgumentParser(description="Generate eval queries from a FAISS index.")
    p.add_argument("--index-dir",    default="data/faiss_index_1m_chunklevel",
                   help="Path to the FAISS index directory")
    p.add_argument("--output",       default="data/eval_queries.json",
                   help="Output path for the generated queries file")
    p.add_argument("--top-relevant", type=int, default=5,
                   help="Number of unique article IDs to keep per query (default: 5)")
    p.add_argument("--broad-k",      type=int, default=50,
                   help="Number of chunks retrieved before de-duplicating (default: 50)")
    p.add_argument("--device",       default="cuda",
                   help="Device for the embedding model: 'cuda' or 'cpu'")
    args = p.parse_args()

    store    = FAISSVectorStore.load(args.index_dir)
    embedder = ArticleEmbedder(device=args.device)

    n_chunks   = len(store.chunks)
    n_articles = len({c.article_id for c in store.chunks})
    print(f"Loaded index: {n_chunks:,} chunks across {n_articles:,} articles.\n")

    queries = []
    for query_text, level in SEED_QUERIES:
        q_emb   = embedder.embed_query(query_text)
        results = store.search(q_emb, top_k=min(args.broad_k, n_chunks))

        # De-duplicate to unique articles in rank order
        seen:        Set[str]  = set()
        article_ids: List[str] = []
        titles:      List[str] = []
        for chunk, _score in results:
            if chunk.article_id not in seen:
                seen.add(chunk.article_id)
                article_ids.append(chunk.article_id)
                titles.append(chunk.title)
            if len(article_ids) >= args.top_relevant:
                break

        queries.append({
            "query":        query_text,
            "relevant_ids": article_ids,
            "level":        level,
            "note":         ", ".join(titles),
        })
        print(f"[{level:12s}] {query_text!r}")
        print(f"             → {article_ids}")
        print(f"             titles: {', '.join(titles)}\n")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(queries, indent=2))
    print(f"Saved {len(queries)} queries → {out_path}")


if __name__ == "__main__":
    main()
