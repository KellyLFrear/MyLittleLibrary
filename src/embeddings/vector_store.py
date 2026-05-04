from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from src.embeddings.chunker import ArticleChunk


class FAISSVectorStore:
    """
    Exact cosine-similarity search via IndexFlatIP on L2-normalized vectors.

    Scale notes:
        50k articles × ~5 chunks = ~250k vectors @ 384-dim ≈ 370 MB RAM.
        Exact search queries in <100 ms at that scale — no approximation needed.

        Upgrade path if scaling beyond 1M vectors:
            Replace IndexFlatIP with IndexHNSWFlat:
            >>> self.index = faiss.IndexHNSWFlat(embedding_dim, 32)

        GPU acceleration (enabled by default):
            Uses the first available CUDA device via faiss-gpu.
            Falls back to CPU index if GPU resources are unavailable.
    """

    def __init__(self, embedding_dim: int, use_gpu: bool = True):
        self.embedding_dim = embedding_dim
        cpu_index = faiss.IndexFlatIP(embedding_dim)
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                self._gpu_res = res  # keep reference alive
            except Exception:
                self.index = cpu_index
        else:
            self.index = cpu_index
        self.chunks: List[ArticleChunk] = []

    def add(self, chunks: List[ArticleChunk], embeddings: np.ndarray) -> None:
        assert embeddings.shape == (len(chunks), self.embedding_dim), (
            f"Shape mismatch: {embeddings.shape} vs ({len(chunks)}, {self.embedding_dim})"
        )
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,   # shape (1, dim) or (dim,), L2-normalized
        top_k: int = 50,
    ) -> List[Tuple[ArticleChunk, float]]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis, :]
        scores, indices = self.index.search(query_embedding, top_k)
        return [
            (self.chunks[i], float(s))
            for s, i in zip(scores[0], indices[0])
            if i != -1
        ]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        # Always serialize to a CPU index for portability
        cpu_index = faiss.index_gpu_to_cpu(self.index) if hasattr(faiss, "index_gpu_to_cpu") else self.index
        faiss.write_index(cpu_index, str(d / "index.faiss"))
        with open(d / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(d / "meta.json", "w") as f:
            json.dump(
                {"embedding_dim": self.embedding_dim, "num_chunks": len(self.chunks)},
                f,
            )
        print(f"Saved {len(self.chunks):,} chunks to {d}/")

    @classmethod
    def load(cls, directory: str | Path, use_gpu: bool = True) -> "FAISSVectorStore":
        d = Path(directory)
        with open(d / "meta.json") as f:
            meta = json.load(f)
        store = cls(embedding_dim=meta["embedding_dim"], use_gpu=use_gpu)
        cpu_index = faiss.read_index(str(d / "index.faiss"))
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                store.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                store._gpu_res = res
            except Exception:
                store.index = cpu_index
        else:
            store.index = cpu_index
        with open(d / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        print(f"Loaded {len(store.chunks):,} chunks from {d}/")
        return store
