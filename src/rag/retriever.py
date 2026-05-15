from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from src.embeddings.chunker import ArticleChunk
from src.embeddings.embedder import ArticleEmbedder
from src.embeddings.vector_store import FAISSVectorStore
from src.rag.reranker import VocabAwareReranker

if TYPE_CHECKING:
    from src.rag.student_profile import StudentProfile


class TwoStageRetriever:
    """
    Stage 1 — FAISS bi-encoder: broad top-`top_broad` recall, fast.
    Stage 2 — VocabAwareReranker: precise, vocabulary-aware, cross-encoder scored.

    Adaptive fallback mode
    ----------------------
    If a topic query returns semantically relevant but vocabulary-too-hard results,
    the retriever automatically searches easier educational query variants and
    reranks the combined candidate pool.

    Example:
        query = "space and planets"

    If top results are too hard, also try:
        - "space and planets for students"
        - "basic space and planets"
        - "space and planets simple explanation"
        - "basic astronomy planets solar system"
        - "solar system planets space for students"

    This helps the system recommend texts that are both:
        1. relevant to the user's interest
        2. closer to the student's reading level
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: ArticleEmbedder,
        reranker: VocabAwareReranker,
        top_broad: int = 50,
        top_k: int = 10,
        adaptive_min_coverage: float = 0.80,
        adaptive_fallback_top_broad: int = 100,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.top_broad = top_broad
        self.top_k = top_k
        self.adaptive_min_coverage = adaptive_min_coverage
        self.adaptive_fallback_top_broad = adaptive_fallback_top_broad

    def _semantic_search(
        self,
        query: str,
        *,
        top_broad: Optional[int] = None,
    ) -> List[Tuple[ArticleChunk, float]]:
        """Run FAISS semantic search for a query."""
        requested_top_broad = top_broad if top_broad is not None else self.top_broad
        safe_top_broad = min(requested_top_broad, len(self.vector_store.chunks))

        q_emb = self.embedder.embed_query(query)
        return self.vector_store.search(q_emb, top_k=safe_top_broad)

    def _profile_pool(self) -> List[Tuple[ArticleChunk, float]]:
        """
        Profile-driven mode: skip FAISS and let vocabulary coverage drive ranking.
        """
        pool_size = min(self.top_broad * 4, len(self.vector_store.chunks))
        return [(chunk, 1.0) for chunk in self.vector_store.chunks[:pool_size]]

    def _chunk_key(self, chunk: ArticleChunk) -> str:
        """
        Stable-ish key for de-duplicating candidate chunks.
        Prefer chunk_id, then article_id + text prefix.
        """
        chunk_id = getattr(chunk, "chunk_id", None)
        if chunk_id is not None:
            return f"chunk:{chunk_id}"

        article_id = getattr(chunk, "article_id", None)
        text_prefix = getattr(chunk, "text", "")[:80]
        return f"article:{article_id}:text:{text_prefix}"

    def _merge_candidates(
        self,
        candidate_groups: List[List[Tuple[ArticleChunk, float]]],
    ) -> List[Tuple[ArticleChunk, float]]:
        """
        Merge candidate lists while keeping the best broad-stage score per chunk.
        """
        merged: Dict[str, Tuple[ArticleChunk, float]] = {}

        for group in candidate_groups:
            for chunk, score in group:
                key = self._chunk_key(chunk)
                if key not in merged or score > merged[key][1]:
                    merged[key] = (chunk, score)

        return list(merged.values())

    def _effective_coverage(
        self,
        chunk: ArticleChunk,
        vocab_level: str,
        student_profile: Optional["StudentProfile"],
    ) -> float:
        """Return student-adjusted coverage if possible, else static chunk coverage."""
        try:
            if student_profile is not None:
                return float(student_profile.adjusted_coverage(chunk))
        except (TypeError, ValueError):
            pass

        try:
            return float(chunk.coverage_ratio.get(vocab_level, 0.0))
        except (TypeError, ValueError, AttributeError):
            return 0.0

    def _best_coverage(
        self,
        ranked: List[Tuple[ArticleChunk, float]],
        vocab_level: str,
        student_profile: Optional["StudentProfile"],
    ) -> float:
        if not ranked:
            return 0.0

        return max(
            self._effective_coverage(chunk, vocab_level, student_profile)
            for chunk, _ in ranked[: self.top_k]
        )

    def _average_coverage(
        self,
        ranked: List[Tuple[ArticleChunk, float]],
        vocab_level: str,
        student_profile: Optional["StudentProfile"],
    ) -> float:
        top = ranked[: self.top_k]
        if not top:
            return 0.0

        values = [
            self._effective_coverage(chunk, vocab_level, student_profile)
            for chunk, _ in top
        ]
        return sum(values) / len(values)

    def _needs_adaptive_fallback(
        self,
        ranked: List[Tuple[ArticleChunk, float]],
        vocab_level: str,
        student_profile: Optional["StudentProfile"],
        coverage_window: Tuple[float, float],
    ) -> bool:
        """
        Return True if the current top recommendations are too hard.

        We trigger fallback if none of the top results reach:
            max(adaptive_min_coverage, coverage_window_low - 0.05)

        For default 0.85–0.97, this means fallback triggers if best coverage < 0.80.
        """
        if not ranked:
            return True

        low, _ = coverage_window
        threshold = max(self.adaptive_min_coverage, float(low) - 0.05)

        best = self._best_coverage(ranked, vocab_level, student_profile)
        return best < threshold

    def _fallback_queries(self, query: str) -> List[str]:
        """
        Build easier educational query variants.

        These are intentionally generic and safe. They bias retrieval toward
        simpler explanatory articles without losing the user's topic.
        """
        cleaned = " ".join(query.strip().split())
        if not cleaned:
            return []

        fallbacks = [
            f"{cleaned} for students",
            f"basic {cleaned}",
            f"{cleaned} simple explanation",
            f"{cleaned} beginner friendly",
        ]

        lower = cleaned.lower()
        space_terms = ("space", "planet", "planets", "astronomy", "solar system")
        if any(term in lower for term in space_terms):
            fallbacks.extend(
                [
                    "solar system planets space for students",
                    "basic astronomy planets solar system",
                    "simple solar system planets",
                ]
            )

        return fallbacks

    def retrieve(
        self,
        query: Optional[str],
        vocab_level: str,
        coverage_window: Tuple[float, float] = (0.85, 0.97),
        student_profile: Optional["StudentProfile"] = None,
    ) -> List[Tuple[ArticleChunk, float]]:
        # Profile-driven mode: no topic query, so ranking is vocabulary-first.
        if query is None:
            broad = self._profile_pool()
            reranked = self.reranker.rerank(
                query=None,
                candidates=broad,
                vocab_level=vocab_level,
                coverage_window=coverage_window,
                student_profile=student_profile,
            )
            return reranked[: self.top_k]

        # Stage 1: primary semantic search.
        primary_broad = self._semantic_search(query, top_broad=self.top_broad)

        # Stage 2: primary reranking.
        primary_reranked = self.reranker.rerank(
            query=query,
            candidates=primary_broad,
            vocab_level=vocab_level,
            coverage_window=coverage_window,
            student_profile=student_profile,
        )

        # If the primary results are close enough to the student's level, stop here.
        if not self._needs_adaptive_fallback(
            primary_reranked,
            vocab_level,
            student_profile,
            coverage_window,
        ):
            return primary_reranked[: self.top_k]

        # Adaptive fallback: search easier versions of the same topic.
        fallback_groups: List[List[Tuple[ArticleChunk, float]]] = []

        fallback_top_broad = min(
            self.adaptive_fallback_top_broad,
            self.top_broad,
            len(self.vector_store.chunks),
        )

        for fallback_query in self._fallback_queries(query):
            fallback_groups.append(
                self._semantic_search(fallback_query, top_broad=fallback_top_broad)
            )

        if not fallback_groups:
            return primary_reranked[: self.top_k]

        # Merge primary + fallback candidates, then rerank against the original query.
        merged_candidates = self._merge_candidates([primary_broad, *fallback_groups])

        adaptive_reranked = self.reranker.rerank(
            query=query,
            candidates=merged_candidates,
            vocab_level=vocab_level,
            coverage_window=coverage_window,
            student_profile=student_profile,
        )

        # Keep adaptive results if they improve average coverage or best coverage.
        primary_avg = self._average_coverage(
            primary_reranked,
            vocab_level,
            student_profile,
        )
        adaptive_avg = self._average_coverage(
            adaptive_reranked,
            vocab_level,
            student_profile,
        )

        primary_best = self._best_coverage(
            primary_reranked,
            vocab_level,
            student_profile,
        )
        adaptive_best = self._best_coverage(
            adaptive_reranked,
            vocab_level,
            student_profile,
        )

        if adaptive_avg >= primary_avg or adaptive_best > primary_best:
            return adaptive_reranked[: self.top_k]

        return primary_reranked[: self.top_k]