"""
StudentProfile — tracks a student's known vocabulary and simulates growth.

Lifecycle:
    profile = StudentProfile(base_level="intermediate")
    results  = pipeline.recommend(query, student_profile=profile)
    # student reads the top article:
    newly_learned = profile.mark_as_read(results[0])
    # next recommend() call uses an updated coverage ratio automatically

Loading from DB:
    with get_db() as conn:
        profile = StudentProfile.from_db(user_id=42, conn=conn)
"""
from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, List, Optional, Set

from src.embeddings.chunker import ArticleChunk

if TYPE_CHECKING:
    pass  # avoid circular imports


class StudentProfile:
    # Maps DB-stored title-case levels to the lowercase keys used elsewhere
    _LEVEL_MAP: dict[str, str] = {
        "Beginner": "beginner",
        "Intermediate": "intermediate",
        "Advanced": "advanced",
    }

    def __init__(
        self,
        base_level: str,
        learned_words: Optional[Set[str]] = None,
    ):
        assert base_level in ("beginner", "intermediate", "advanced"), (
            f"base_level must be one of beginner/intermediate/advanced, got '{base_level}'"
        )
        self.base_level = base_level
        # Words the student has learned on top of their base level vocabulary
        self.learned_words: Set[str] = {w.lower() for w in (learned_words or set())}

    # ── DB loading ────────────────────────────────────────────────────────────

    @classmethod
    def from_db(cls, user_id: int, conn: sqlite3.Connection) -> "StudentProfile":
        """
        Build a StudentProfile from the user's current reading profile and
        known-word list stored in the database.

        Parameters
        ----------
        user_id:
            Primary key from the ``users`` table.
        conn:
            Open ``sqlite3.Connection`` (from ``get_db()``).  The connection
            must have ``row_factory = sqlite3.Row``.

        Raises
        ------
        ValueError
            If the user has no current reading profile in the DB.
        """
        # 1. Fetch the active reading profile to get the user's level
        profile_row = conn.execute(
            """
            SELECT estimated_level
            FROM user_reading_profiles
            WHERE user_id = ? AND is_current = 1
            """,
            (user_id,),
        ).fetchone()

        if profile_row is None:
            raise ValueError(
                f"No current reading profile found for user_id={user_id}. "
                "Create one with repositories.create_reading_profile() first."
            )

        base_level = cls._LEVEL_MAP.get(profile_row["estimated_level"], "beginner")

        # 2. Fetch all words the user is known/likely_known for
        rows = conn.execute(
            """
            SELECT vt.word
            FROM user_word_knowledge uwk
            JOIN vocabulary_terms vt ON uwk.term_id = vt.term_id
            WHERE uwk.user_id = ?
              AND uwk.knowledge_status IN ('known', 'likely_known')
            """,
            (user_id,),
        ).fetchall()

        learned_words: Set[str] = {r["word"].lower() for r in rows}

        return cls(base_level=base_level, learned_words=learned_words)

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

    def known_words(self) -> Set[str]:
        """Return all words known to the student (learned_words set)."""
        return set(self.learned_words)

    def summary(self) -> str:
        return (
            f"Level: {self.base_level} | "
            f"Extra vocab learned: {len(self.learned_words)} words"
        )
