from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from src.embeddings.chunker import ArticleChunk
from src.embeddings.embedder import ArticleEmbedder
from src.embeddings.vector_store import FAISSVectorStore
from src.rag.reranker import VocabAwareReranker

if TYPE_CHECKING:
    from src.rag.student_profile import StudentProfile


class TwoStageRetriever:
    """
    Stage 1 — FAISS bi-encoder: broad top-`top_broad` recall, fast.
    Stage 2 — VocabAwareReranker: precise, vocabulary-filtered, cross-encoder scored.

    Profile-driven mode (query=None)
    ---------------------------------
    When ``query`` is ``None`` the FAISS semantic search is skipped entirely.
    Instead, *all* indexed chunks (up to ``top_broad``) are passed directly to
    the reranker, which then ranks purely by vocabulary-coverage score.  This
    allows the system to surface articles appropriate for a student's reading
    level and known-word list without requiring the student to name a topic.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: ArticleEmbedder,
        reranker: VocabAwareReranker,
        top_broad: int = 50,
        top_k: int = 10,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.top_broad = top_broad
        self.top_k = top_k

    def retrieve(
        self,
        query: Optional[str],
        vocab_level: str,
        coverage_window: Tuple[float, float] = (0.85, 0.97),
        student_profile: Optional["StudentProfile"] = None,
    ) -> List[Tuple[ArticleChunk, float]]:
        if query is not None:
            # Stage 1: broad semantic recall via FAISS
            q_emb = self.embedder.embed_query(query)
            # Cap top_broad to index size to avoid FAISS index errors
            top_broad = min(self.top_broad, len(self.vector_store.chunks))
            broad = self.vector_store.search(q_emb, top_k=top_broad)
        else:
            # Profile-driven mode: skip FAISS, feed the full chunk pool to the
            # reranker so vocab-coverage scoring drives all ranking.
            # Use a larger effective pool (up to 4× top_broad) to give the
            # vocab reranker enough diversity to work with.
            pool_size = min(self.top_broad * 4, len(self.vector_store.chunks))
            broad = [(chunk, 1.0) for chunk in self.vector_store.chunks[:pool_size]]

        # Stage 2: precise reranking with vocabulary filter (+ optional profile)
        reranked = self.reranker.rerank(
            query=query,
            candidates=broad,
            vocab_level=vocab_level,
            coverage_window=coverage_window,
            student_profile=student_profile,
        )
        return reranked[: self.top_k]
