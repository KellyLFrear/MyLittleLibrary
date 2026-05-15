"""
Chunking strategy:
  - Chunk size: 400 words  → fits within the 512-token limit of all-MiniLM-L6-v2
    while preserving semantic coherence across one or two paragraphs.
  - Overlap:    50 words (12.5%) → prevents retrieval misses at chunk boundaries
    without excessively growing index size.
  - Sentence-boundary aware: chunks only close at a sentence boundary, so
    semantic units are never split mid-thought.

Vocabulary metadata:
  - chunk_article() initially copies article-level metadata into each chunk.
  - scripts/build_index.py should recompute coverage_ratio and new_words at
    the chunk level before embedding/saving the FAISS index.
  - This makes recommendations score the selected passage, not the full article.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ArticleChunk:
    chunk_id: str            # "{article_id}_chunk_{idx}"
    article_id: str
    title: str
    text: str
    chunk_index: int
    # Inherited from the parent article (produced by step-1 pipeline)
    coverage_ratio: Dict[str, float] = field(default_factory=dict)
    new_words: Dict[str, List[str]] = field(default_factory=dict)
    readability_score: float = 0.0


def _sentences(text: str) -> List[str]:
    """Sentence splitter sufficient for Wikipedia prose."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\"])', text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_article(
    article: dict,
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[ArticleChunk]:
    """Split one article dict into overlapping word-level chunks."""
    text = article.get("text", "").strip()
    if not text:
        return []

    aid   = article["id"]
    title = article.get("title", "")
    cov   = article.get("coverage_ratio", {})
    nw    = article.get("new_words", {})
    read  = article.get("readability_score", 0.0)

    chunks: List[ArticleChunk] = []
    current: List[str] = []
    idx = 0

    for sent in _sentences(text):
        words = sent.split()
        if len(current) + len(words) > chunk_size and current:
            chunks.append(ArticleChunk(
                chunk_id=f"{aid}_chunk_{idx}",
                article_id=aid,
                title=title,
                text=" ".join(current),
                chunk_index=idx,
                coverage_ratio=cov,
                new_words=nw,
                readability_score=read,
            ))
            idx += 1
            current = current[-overlap:]  # carry trailing overlap into next chunk
        current.extend(words)

    if current:
        chunks.append(ArticleChunk(
            chunk_id=f"{aid}_chunk_{idx}",
            article_id=aid,
            title=title,
            text=" ".join(current),
            chunk_index=idx,
            coverage_ratio=cov,
            new_words=nw,
            readability_score=read,
        ))

    return chunks
