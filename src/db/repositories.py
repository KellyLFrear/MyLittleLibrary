"""
Repository functions for every table defined in schema.sql.

All functions accept an open ``sqlite3.Connection`` (obtained from
``src.db.connection.get_db``) and return plain Python objects or None.

Conventions
-----------
* Functions that write data do NOT commit — the caller's ``with get_db()``
  context manager handles that.
* Passwords are NEVER handled here.  Pass in a pre-hashed value; this layer
  never sees plaintext.
* Row objects from sqlite3.Row are converted to dicts where a plain mapping
  is more convenient for callers.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional, Set


# ── Helpers ────────────────────────────────────────────────────────────────────

def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None


# ── Users ──────────────────────────────────────────────────────────────────────

def create_user(
    conn: sqlite3.Connection,
    *,
    username: str,
    password_hash: str,
    display_name: Optional[str] = None,
    age: Optional[int] = None,
    avg_reading_lvl: Optional[str] = None,
) -> int:
    """
    Insert a new user and return the new ``user_id``.

    Raises ``sqlite3.IntegrityError`` if the username already exists.

    Parameters
    ----------
    password_hash:
        Pre-computed hash (e.g. from bcrypt / argon2).  Never pass a
        plaintext password to this function.
    avg_reading_lvl:
        Optional reading level string, e.g. 'Beginner', 'Intermediate', 'Advanced'.
    """
    cur = conn.execute(
        """
        INSERT INTO users (username, password_hash, display_name, age, avg_reading_lvl)
        VALUES (?, ?, ?, ?, ?)
        """,
        (username, password_hash, display_name, age, avg_reading_lvl),
    )
    return cur.lastrowid  # type: ignore[return-value]


def get_user_by_id(
    conn: sqlite3.Connection,
    user_id: int,
) -> Optional[Dict[str, Any]]:
    """Return user row as a dict, or None if not found."""
    row = conn.execute(
        "SELECT * FROM users WHERE user_id = ?", (user_id,)
    ).fetchone()
    return _row_to_dict(row)


def get_user_by_username(
    conn: sqlite3.Connection,
    username: str,
) -> Optional[Dict[str, Any]]:
    """Return user row as a dict, or None if not found."""
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    return _row_to_dict(row)


# ── Vocabulary terms ───────────────────────────────────────────────────────────

def get_term_by_id(
    conn: sqlite3.Connection,
    term_id: int,
) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM vocabulary_terms WHERE term_id = ?", (term_id,)
    ).fetchone()
    return _row_to_dict(row)


def get_terms_by_level(
    conn: sqlite3.Connection,
    level: str,
) -> List[Dict[str, Any]]:
    """Return all vocabulary_terms rows for a given level."""
    rows = conn.execute(
        "SELECT * FROM vocabulary_terms WHERE level = ?", (level,)
    ).fetchall()
    return [dict(r) for r in rows]


def upsert_vocabulary_term(
    conn: sqlite3.Connection,
    *,
    word: str,
    level: str = "Unknown",
    source: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    """
    Insert or ignore a vocabulary term (unique on word+level).
    Returns the term_id (existing or newly created).
    """
    conn.execute(
        """
        INSERT OR IGNORE INTO vocabulary_terms (word, level, source, notes)
        VALUES (?, ?, ?, ?)
        """,
        (word, level, source, notes),
    )
    row = conn.execute(
        "SELECT term_id FROM vocabulary_terms WHERE word=? AND level=?",
        (word, level),
    ).fetchone()
    return row["term_id"]


# ── User word knowledge ────────────────────────────────────────────────────────

def get_known_words(
    conn: sqlite3.Connection,
    user_id: int,
    statuses: tuple[str, ...] = ("known", "likely_known"),
) -> Set[str]:
    """
    Return the set of *word* strings (lowercase) from vocabulary_terms that
    this user has achieved one of the given knowledge statuses for.

    Default statuses are ``('known', 'likely_known')``.
    """
    placeholders = ",".join("?" * len(statuses))
    rows = conn.execute(
        f"""
        SELECT vt.word
        FROM user_word_knowledge uwk
        JOIN vocabulary_terms vt ON uwk.term_id = vt.term_id
        WHERE uwk.user_id = ?
          AND uwk.knowledge_status IN ({placeholders})
        """,
        (user_id, *statuses),
    ).fetchall()
    return {r["word"].lower() for r in rows}


def upsert_word_knowledge(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    term_id: int,
    knowledge_status: str,
    confidence: float = 0.0,
    times_seen_delta: int = 0,
    times_used_delta: int = 0,
    times_correct_delta: int = 0,
    times_incorrect_delta: int = 0,
    run_id: Optional[int] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Insert a new user_word_knowledge row, or update an existing one.

    Counter columns are incremented by their ``*_delta`` values (default 0).
    """
    evidence_json = json.dumps(evidence) if evidence is not None else None
    conn.execute(
        """
        INSERT INTO user_word_knowledge
            (user_id, term_id, knowledge_status, confidence,
             times_seen, times_used, times_correct, times_incorrect,
             last_seen_at, last_updated_by_run_id, evidence_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
        ON CONFLICT(user_id, term_id) DO UPDATE SET
            knowledge_status        = excluded.knowledge_status,
            confidence              = excluded.confidence,
            times_seen              = times_seen  + excluded.times_seen,
            times_used              = times_used  + excluded.times_used,
            times_correct           = times_correct   + excluded.times_correct,
            times_incorrect         = times_incorrect + excluded.times_incorrect,
            last_seen_at            = CURRENT_TIMESTAMP,
            last_updated_by_run_id  = excluded.last_updated_by_run_id,
            evidence_json           = excluded.evidence_json
        """,
        (
            user_id, term_id, knowledge_status, confidence,
            times_seen_delta, times_used_delta,
            times_correct_delta, times_incorrect_delta,
            run_id, evidence_json,
        ),
    )


def get_word_knowledge(
    conn: sqlite3.Connection,
    user_id: int,
    term_id: int,
) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM user_word_knowledge WHERE user_id = ? AND term_id = ?",
        (user_id, term_id),
    ).fetchone()
    return _row_to_dict(row)


# ── User reading profiles ──────────────────────────────────────────────────────

def get_current_reading_profile(
    conn: sqlite3.Connection,
    user_id: int,
) -> Optional[Dict[str, Any]]:
    """Return the active (is_current=1) reading profile for a user."""
    row = conn.execute(
        "SELECT * FROM user_reading_profiles WHERE user_id = ? AND is_current = 1",
        (user_id,),
    ).fetchone()
    return _row_to_dict(row)


def create_reading_profile(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    estimated_level: str,
    estimated_grade_band: Optional[str] = None,
    known_beginner_words: int = 0,
    known_intermediate_words: int = 0,
    known_advanced_words: int = 0,
    known_unknown_level_words: int = 0,
    total_known_words: int = 0,
    academic_words_known: int = 0,
    beginner_mastery: float = 0.0,
    intermediate_mastery: float = 0.0,
    advanced_mastery: float = 0.0,
    confidence: float = 0.0,
    profile_json: Optional[Dict[str, Any]] = None,
    run_id: Optional[int] = None,
) -> int:
    """
    Retire the existing current profile (is_current → 0) and insert a new one.
    Returns the new ``profile_id``.
    """
    conn.execute(
        "UPDATE user_reading_profiles SET is_current = 0 WHERE user_id = ? AND is_current = 1",
        (user_id,),
    )
    cur = conn.execute(
        """
        INSERT INTO user_reading_profiles (
            user_id, run_id, estimated_level, estimated_grade_band,
            known_beginner_words, known_intermediate_words,
            known_advanced_words, known_unknown_level_words,
            total_known_words, academic_words_known,
            beginner_mastery, intermediate_mastery, advanced_mastery,
            confidence, profile_json, is_current
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """,
        (
            user_id, run_id, estimated_level, estimated_grade_band,
            known_beginner_words, known_intermediate_words,
            known_advanced_words, known_unknown_level_words,
            total_known_words, academic_words_known,
            beginner_mastery, intermediate_mastery, advanced_mastery,
            confidence,
            json.dumps(profile_json) if profile_json is not None else None,
        ),
    )
    return cur.lastrowid  # type: ignore[return-value]


def list_reading_profiles(
    conn: sqlite3.Connection,
    user_id: int,
) -> List[Dict[str, Any]]:
    """Return all profile snapshots for a user, newest first."""
    rows = conn.execute(
        "SELECT * FROM user_reading_profiles WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ── Analysis runs ──────────────────────────────────────────────────────────────

def start_analysis_run(
    conn: sqlite3.Connection,
    *,
    run_target_type: str,
    user_id: Optional[int] = None,
    book_id: Optional[int] = None,
    source_id: Optional[int] = None,
    story_id: Optional[int] = None,
    run_name: Optional[str] = None,
    script_name: Optional[str] = None,
    script_version: Optional[str] = None,
    model_name: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> int:
    """Insert a new analysis_run with status='started'. Returns ``run_id``."""
    cur = conn.execute(
        """
        INSERT INTO analysis_runs (
            run_target_type, user_id, book_id, source_id, story_id,
            run_name, script_name, script_version, model_name,
            tokenizer_name, chunk_size, parameters_json, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'started')
        """,
        (
            run_target_type, user_id, book_id, source_id, story_id,
            run_name, script_name, script_version, model_name,
            tokenizer_name, chunk_size,
            json.dumps(parameters) if parameters is not None else None,
        ),
    )
    return cur.lastrowid  # type: ignore[return-value]


def finish_analysis_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    status: str = "completed",
    notes: Optional[str] = None,
) -> None:
    """Mark an analysis run as completed or failed."""
    conn.execute(
        """
        UPDATE analysis_runs
        SET status = ?, completed_at = CURRENT_TIMESTAMP, notes = ?
        WHERE run_id = ?
        """,
        (status, notes, run_id),
    )


# ── Books ──────────────────────────────────────────────────────────────────────

def upsert_book(
    conn: sqlite3.Connection,
    *,
    title: str,
    author: Optional[str] = None,
    source_file: Optional[str] = None,
    cleaned_file: Optional[str] = None,
    source_hash: Optional[str] = None,
    language: str = "en",
) -> int:
    """
    Insert a book, or return the existing book_id if source_hash already exists.
    When source_hash is None a new row is always inserted.
    """
    if source_hash is not None:
        existing = conn.execute(
            "SELECT book_id FROM books WHERE source_hash = ?", (source_hash,)
        ).fetchone()
        if existing:
            return existing["book_id"]

    cur = conn.execute(
        """
        INSERT INTO books (title, author, source_file, cleaned_file, source_hash, language)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (title, author, source_file, cleaned_file, source_hash, language),
    )
    return cur.lastrowid  # type: ignore[return-value]


def get_book_by_id(
    conn: sqlite3.Connection,
    book_id: int,
) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM books WHERE book_id = ?", (book_id,)
    ).fetchone()
    return _row_to_dict(row)


# ── Book recommendations ───────────────────────────────────────────────────────

def save_book_recommendations(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    profile_id: Optional[int],
    recommendations: List[Dict[str, Any]],
) -> None:
    """
    Retire current recommendations for this user and insert fresh ones.

    Each item in ``recommendations`` must contain at least ``book_id`` and
    ``match_score``.  Optional keys: ``recommended_level``, ``known_word_ratio``,
    ``unknown_word_ratio``, ``advanced_word_ratio``, ``academic_word_ratio``,
    ``reason``.
    """
    conn.execute(
        "UPDATE book_recommendations SET is_current = 0 WHERE user_id = ? AND is_current = 1",
        (user_id,),
    )
    for rec in recommendations:
        conn.execute(
            """
            INSERT INTO book_recommendations (
                user_id, profile_id, book_id, recommended_level,
                match_score, known_word_ratio, unknown_word_ratio,
                advanced_word_ratio, academic_word_ratio, reason, is_current
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                user_id,
                profile_id,
                rec["book_id"],
                rec.get("recommended_level", "Unknown"),
                rec["match_score"],
                rec.get("known_word_ratio"),
                rec.get("unknown_word_ratio"),
                rec.get("advanced_word_ratio"),
                rec.get("academic_word_ratio"),
                rec.get("reason"),
            ),
        )


def get_current_recommendations(
    conn: sqlite3.Connection,
    user_id: int,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Return the current (is_current=1) recommendations ordered by match_score."""
    rows = conn.execute(
        """
        SELECT br.*, b.title, b.author
        FROM book_recommendations br
        JOIN books b ON br.book_id = b.book_id
        WHERE br.user_id = ? AND br.is_current = 1
        ORDER BY br.match_score DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ── Generated stories ──────────────────────────────────────────────────────────

def save_story(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    title: str,
    story_text: str,
    target_level: str,
    genre: Optional[str] = None,
    challenge: Optional[str] = None,
    new_vocab: Optional[List[Dict[str, str]]] = None,
    challenge_note: Optional[str] = None,
    model_name: Optional[str] = None,
    requested_word_count: Optional[int] = None,
) -> int:
    """
    Persist a generated story to the database after the user confirms they
    want to keep it.  Returns the new ``story_id``.

    Parameters
    ----------
    user_id:
        Owner of the story.
    title:
        Story title as returned by the model.
    story_text:
        Full body of the story.
    target_level:
        Must be one of ``'Beginner'``, ``'Intermediate'``, ``'Advanced'``.
        Pass the title-cased value — the DB CHECK constraint requires it.
    genre:
        e.g. ``'adventure'``, ``'mystery'``, etc.  Stored in ``prompt_used``
        alongside ``challenge`` as a JSON string.
    challenge:
        Difficulty setting: ``'low'``, ``'medium'``, or ``'high'``.
    new_vocab:
        List of ``{"word": ..., "definition": ...}`` dicts from the model.
        Stored in ``practice_words_json``.
    challenge_note:
        One-sentence challenge description from the model.  Stored in ``notes``
        of a companion ``analysis_runs`` row (not in ``generated_stories`` directly).
    model_name:
        Name/path of the model used, e.g.
        ``'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'``.
    requested_word_count:
        ``target_words`` value that was passed to the generator.
    """
    actual_word_count = len(story_text.split())

    # Normalise level to title-case for the DB CHECK constraint
    _level_map = {"beginner": "Beginner", "intermediate": "Intermediate", "advanced": "Advanced"}
    db_level = _level_map.get(target_level.lower(), target_level)

    prompt_meta = json.dumps({"genre": genre, "challenge": challenge})
    practice_words = json.dumps(new_vocab) if new_vocab is not None else None

    cur = conn.execute(
        """
        INSERT INTO generated_stories (
            user_id, title, story_text, target_level,
            model_name, prompt_used,
            requested_word_count, actual_word_count,
            practice_words_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id, title, story_text, db_level,
            model_name, prompt_meta,
            requested_word_count, actual_word_count,
            practice_words,
        ),
    )
    return cur.lastrowid  # type: ignore[return-value]


def get_stories_for_user(
    conn: sqlite3.Connection,
    user_id: int,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Return saved stories for a user, newest first."""
    rows = conn.execute(
        """
        SELECT story_id, title, target_level, actual_word_count,
               prompt_used, practice_words_json, created_at
        FROM generated_stories
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    results = []
    for r in rows:
        row = dict(r)
        # Deserialise JSON blobs for convenience
        if row["practice_words_json"]:
            row["new_vocab"] = json.loads(row["practice_words_json"])
        if row["prompt_used"]:
            row["prompt_meta"] = json.loads(row["prompt_used"])
        results.append(row)
    return results


def get_story_by_id(
    conn: sqlite3.Connection,
    story_id: int,
) -> Optional[Dict[str, Any]]:
    """Return a single saved story as a dict (with full story_text), or None."""
    row = conn.execute(
        "SELECT * FROM generated_stories WHERE story_id = ?", (story_id,)
    ).fetchone()
    if row is None:
        return None
    result = dict(row)
    if result.get("practice_words_json"):
        result["new_vocab"] = json.loads(result["practice_words_json"])
    if result.get("prompt_used"):
        result["prompt_meta"] = json.loads(result["prompt_used"])
    return result


# ── User library (user-submitted books / user_sources) ─────────────────────────

def add_user_book(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    title: str,
    body_text: str,
) -> int:
    """
    Save a user-submitted book (title + body) to user_sources.
    Returns the new ``source_id``.
    """
    cur = conn.execute(
        """
        INSERT INTO user_sources (user_id, title, source_type, raw_text)
        VALUES (?, ?, 'uploaded_book', ?)
        """,
        (user_id, title, body_text),
    )
    return cur.lastrowid  # type: ignore[return-value]


def get_user_library(
    conn: sqlite3.Connection,
    user_id: int,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Return user-submitted books from user_sources, newest first.
    """
    rows = conn.execute(
        """
        SELECT source_id, title, source_type, created_at
        FROM user_sources
        WHERE user_id = ? AND source_type = 'uploaded_book'
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ── Book search ────────────────────────────────────────────────────────────────

def search_books_by_title(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Case-insensitive substring search across both the ``books`` table (Wikipedia
    articles / system books) and the ``user_sources`` table (user-uploaded books).

    Each result dict always contains the keys:
      ``source`` ("system" | "user"), ``title``, ``author`` (may be None),
      and either ``book_id`` or ``source_id``.
    """
    pattern = f"%{query}%"

    system_rows = conn.execute(
        """
        SELECT book_id, title, author
        FROM books
        WHERE title LIKE ?
        ORDER BY title
        LIMIT ?
        """,
        (pattern, limit),
    ).fetchall()

    user_rows = conn.execute(
        """
        SELECT source_id, title
        FROM user_sources
        WHERE source_type = 'uploaded_book' AND title LIKE ?
        ORDER BY title
        LIMIT ?
        """,
        (pattern, limit),
    ).fetchall()

    results: List[Dict[str, Any]] = []
    for r in system_rows:
        results.append({"source": "system", "book_id": r["book_id"],
                         "title": r["title"], "author": r["author"]})
    for r in user_rows:
        results.append({"source": "user", "source_id": r["source_id"],
                         "title": r["title"], "author": None})
    return results[:limit]


# ── User account ───────────────────────────────────────────────────────────────

def update_user_display_name(
    conn: sqlite3.Connection,
    user_id: int,
    display_name: str,
) -> None:
    """Update a user's display name."""
    conn.execute(
        "UPDATE users SET display_name = ? WHERE user_id = ?",
        (display_name, user_id),
    )
