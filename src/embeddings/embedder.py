from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.embeddings.chunker import ArticleChunk


class ArticleEmbedder:
    """
    Sentence-transformer encoder producing L2-normalized 384-dim vectors.
    Normalization means inner product == cosine similarity, which pairs
    correctly with FAISS IndexFlatIP.

    Runs on GPU by default (device="cuda"). Accelerate handles multi-GPU
    distribution automatically when multiple CUDA devices are available.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cuda",
    ):
        self.model = SentenceTransformer(model_name, device=device)
        # get_embedding_dimension is the current API; fall back for older versions
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
                  getattr(self.model, "get_sentence_embedding_dimension")
        self.embedding_dim: int = get_dim()

    def embed_chunks(
        self,
        chunks: List[ArticleChunk],
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> np.ndarray:
        # Prefix title so short final-chunk fragments retain article context
        texts = [f"{c.title}: {c.text}" for c in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2-norm → IP == cosine similarity
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
