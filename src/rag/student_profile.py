"""
StudentProfile — tracks a student's known vocabulary and simulates growth.

Lifecycle:
    profile = StudentProfile(base_level="intermediate")
    results  = pipeline.recommend(query, student_profile=profile)
    # student reads the top article:
    newly_learned = profile.mark_as_read(results[0])
    # next recommend() call uses an updated coverage ratio automatically
"""
from __future__ import annotations

from typing import List, Set

from src.embeddings.chunker import ArticleChunk


class StudentProfile:
    def __init__(
        self,
        base_level: str,
        learned_words: Set[str] | None = None,
    ):
        assert base_level in ("beginner", "intermediate", "advanced"), (
            f"base_level must be one of beginner/intermediate/advanced, got '{base_level}'"
        )
        self.base_level = base_level
        # Words the student has learned on top of their base level vocabulary
        self.learned_words: Set[str] = {w.lower() for w in (learned_words or set())}

    # ── Coverage adjustment ───────────────────────────────────────────────────

    def adjusted_coverage(self, chunk: ArticleChunk) -> float:
        """
        Coverage ratio adjusted upward for words the student has already learned.
        Each previously-new word the student knows boosts coverage by 1/chunk_word_count.
        """
        base = chunk.coverage_ratio.get(self.base_level, 0.0)
        new_words = {w.lower() for w in chunk.new_words.get(self.base_level, [])}
        learned_overlap = new_words & self.learned_words
        if not learned_overlap:
            return base
        chunk_word_count = max(len(chunk.text.split()), 1)
        boost = len(learned_overlap) / chunk_word_count
        return min(1.0, base + boost)

    def get_remaining_new_words(self, chunk: ArticleChunk) -> List[str]:
        """Words in this chunk that are genuinely new to this student."""
        new_words = chunk.new_words.get(self.base_level, [])
        return [w for w in new_words if w.lower() not in self.learned_words]

    # ── Learning simulation ───────────────────────────────────────────────────

    def mark_as_read(self, chunk: ArticleChunk) -> List[str]:
        """
        Call after a student 'reads' an article.
        Adds all new words from the article to the student's known list.
        Returns the list of words newly learned (for display).
        """
        new_words = chunk.new_words.get(self.base_level, [])
        newly_learned = [w for w in new_words if w.lower() not in self.learned_words]
        self.learned_words.update(w.lower() for w in newly_learned)
        return newly_learned

    def summary(self) -> str:
        return (
            f"Level: {self.base_level} | "
            f"Extra vocab learned: {len(self.learned_words)} words"
        )
