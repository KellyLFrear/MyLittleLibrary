from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from src.embeddings.chunker import ArticleChunk

if TYPE_CHECKING:
    from src.rag.student_profile import StudentProfile


class VocabAwareReranker:
    """
    Two-component reranking score:

        combined = semantic_weight * cross_score_normalized
                 + vocab_weight   * coverage_score

    Coverage scoring:
        Within window  → [0.7, 1.0], peaks at the midpoint of the window.
        Outside window → out_of_window_penalty (large negative), so the
                         article sinks below all in-window results without
                         being hard-filtered (useful for ablation studies).

    Cross-encoder (semantic component):
        Uses cross-encoder/ms-marco-MiniLM-L-6-v2.
        Runs on GPU by default (device="cuda") for fast query-time scoring.
    """

    def __init__(
        self,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        semantic_weight: float = 0.6,
        vocab_weight: float = 0.4,
        out_of_window_penalty: float = -10.0,
        use_cross_encoder: bool = True,
        device: str = "cuda",
    ):
        self.semantic_weight = semantic_weight
        self.vocab_weight = vocab_weight
        self.out_of_window_penalty = out_of_window_penalty
        self.use_cross_encoder = use_cross_encoder
        self._cross_encoder = None

        if use_cross_encoder:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(cross_encoder_model, device=device)

    def _coverage_score_from_ratio(
        self,
        ratio: Optional[float],
        window: Tuple[float, float],
    ) -> float:
        if ratio is None:
            return 0.5   # no metadata → neutral score

        low, high = window
        if not (low <= ratio <= high):
            return self.out_of_window_penalty

        mid = (low + high) / 2
        half_width = (high - low) / 2
        distance = abs(ratio - mid) / half_width   # 0 at center, 1 at edges
        return 1.0 - 0.3 * distance                # [0.7, 1.0]

    def rerank(
        self,
        query: Optional[str],
        candidates: List[Tuple[ArticleChunk, float]],
        vocab_level: str,
        coverage_window: Tuple[float, float] = (0.85, 0.97),
        student_profile: Optional["StudentProfile"] = None,
    ) -> List[Tuple[ArticleChunk, float]]:
        if not candidates:
            return []

        chunks, bi_scores = zip(*candidates)

        # Profile-driven mode (query=None): skip semantic scoring entirely and
        # weight ranking purely by vocabulary-coverage score.
        if query is None:
            effective_semantic_weight = 0.0
            effective_vocab_weight = 1.0
            semantic_scores = [0.0] * len(chunks)
        elif self.use_cross_encoder and self._cross_encoder is not None:
            effective_semantic_weight = self.semantic_weight
            effective_vocab_weight = self.vocab_weight
            pairs = [(query, c.text[:512]) for c in chunks]
            raw = self._cross_encoder.predict(pairs, show_progress_bar=False)
            lo, hi = raw.min(), raw.max()
            semantic_scores = (raw - lo) / (hi - lo + 1e-8)
        else:
            # Fallback: normalize bi-encoder cosine scores as the semantic component
            effective_semantic_weight = self.semantic_weight
            effective_vocab_weight = self.vocab_weight
            bi = np.array(bi_scores, dtype=np.float32)
            lo, hi = bi.min(), bi.max()
            semantic_scores = (bi - lo) / (hi - lo + 1e-8)

        results = []
        for chunk, sem in zip(chunks, semantic_scores):
            # Dynamic coverage if a student profile is provided (vocab growth)
            if student_profile is not None:
                ratio = student_profile.adjusted_coverage(chunk)
            else:
                ratio = chunk.coverage_ratio.get(vocab_level)
            cov = self._coverage_score_from_ratio(ratio, coverage_window)
            if cov < 0:
                score = cov   # demoted below all in-window results
            else:
                score = effective_semantic_weight * float(sem) + effective_vocab_weight * cov
            results.append((chunk, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
