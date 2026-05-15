from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from src.embeddings.chunker import ArticleChunk

if TYPE_CHECKING:
    from src.rag.student_profile import StudentProfile


class VocabAwareReranker:
    """
    Vocabulary-aware reranker.

    Combined score:

        combined = semantic_weight * semantic_score
                 + vocab_weight   * coverage_score

    Key change:
    - Coverage outside the target window is no longer given a huge negative penalty.
    - Instead, it receives a soft distance-based score.
    - This lets the reranker prefer a 0.82 coverage article over a 0.57 coverage article,
      even if both are technically outside the target range.

    This is important for My Little Library because recommendations should be:
    1. relevant to the user's query
    2. close to the student's vocabulary level
    3. slightly challenging, not impossibly hard
    """

    def __init__(
        self,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        semantic_weight: float = 0.35,
        vocab_weight: float = 0.65,
        use_cross_encoder: bool = True,
        device: str = "cuda",
    ):
        if semantic_weight < 0 or vocab_weight < 0:
            raise ValueError("semantic_weight and vocab_weight must be non-negative")

        total = semantic_weight + vocab_weight
        if total <= 0:
            raise ValueError("semantic_weight + vocab_weight must be > 0")

        # Normalize weights in case the caller passes values that do not sum to 1.
        self.semantic_weight = semantic_weight / total
        self.vocab_weight = vocab_weight / total

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
        """
        Convert known-word coverage ratio into a smooth vocabulary-fit score.

        Score behavior:
        - In target window:
            high score, strongest near midpoint
        - Slightly below/above target:
            still usable, but penalized by distance
        - Very far from target:
            score approaches 0

        Example with window 0.85–0.97:
        - 0.91 -> excellent
        - 0.84 -> still good
        - 0.75 -> weak
        - 0.57 -> poor
        """
        if ratio is None:
            return 0.30

        try:
            ratio = float(ratio)
        except (TypeError, ValueError):
            return 0.30

        ratio = min(max(ratio, 0.0), 1.0)

        low, high = window
        low = float(low)
        high = float(high)

        if low > high:
            low, high = high, low

        mid = (low + high) / 2.0
        half_width = max((high - low) / 2.0, 1e-6)

        # Best case: inside target range.
        if low <= ratio <= high:
            distance_from_mid = abs(ratio - mid) / half_width
            return 1.0 - 0.20 * distance_from_mid
            # midpoint = 1.00, edges = 0.80

        # Below target: too hard.
        if ratio < low:
            distance = low - ratio

            # Soft penalty:
            # 0.84 when low is 0.85 should still be very usable.
            # 0.68 should be much weaker.
            return max(0.0, 0.78 - (distance * 2.8))

        # Above target: too easy.
        distance = ratio - high
        return max(0.0, 0.72 - (distance * 2.4))

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize an array to 0–1 safely."""
        if scores.size == 0:
            return scores

        lo = float(scores.min())
        hi = float(scores.max())

        if abs(hi - lo) < 1e-8:
            return np.ones_like(scores, dtype=np.float32) * 0.5

        return (scores - lo) / (hi - lo + 1e-8)

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

        # Profile-driven mode: if there is no query, rank by vocabulary fit only.
        if query is None:
            effective_semantic_weight = 0.0
            effective_vocab_weight = 1.0
            semantic_scores = np.zeros(len(chunks), dtype=np.float32)

        elif self.use_cross_encoder and self._cross_encoder is not None:
            effective_semantic_weight = self.semantic_weight
            effective_vocab_weight = self.vocab_weight

            pairs = [(query, c.text[:512]) for c in chunks]
            raw = self._cross_encoder.predict(pairs, show_progress_bar=False)
            raw = np.array(raw, dtype=np.float32)
            semantic_scores = self._normalize_scores(raw)

        else:
            effective_semantic_weight = self.semantic_weight
            effective_vocab_weight = self.vocab_weight

            bi = np.array(bi_scores, dtype=np.float32)
            semantic_scores = self._normalize_scores(bi)

        results: List[Tuple[ArticleChunk, float]] = []

        for chunk, sem in zip(chunks, semantic_scores):
            if student_profile is not None:
                ratio = student_profile.adjusted_coverage(chunk)
            else:
                ratio = chunk.coverage_ratio.get(vocab_level)

            cov = self._coverage_score_from_ratio(ratio, coverage_window)

            score = (
                effective_semantic_weight * float(sem)
                + effective_vocab_weight * float(cov)
            )

            results.append((chunk, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results