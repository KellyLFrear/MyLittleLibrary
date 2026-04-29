"""
Database connection management for MyLittleLibrary.

Usage
-----
    from src.db.connection import init_db, get_db

    # One-time setup (idempotent — safe to call on every startup)
    init_db()

    # Read / write inside a transaction
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", ("alice",)
        ).fetchone()
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
SCHEMA_PATH: Path = _HERE / "schema.sql"

# Default DB lives at  <repo_root>/data/library.db
DEFAULT_DB_PATH: Path = _HERE.parent.parent.parent / "data" / "library.db"


# ── Public API ─────────────────────────────────────────────────────────────────

def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """
    Create all tables and indexes from schema.sql if they do not already exist.

    Safe to call on every application startup — all DDL statements use
    IF NOT EXISTS, so repeated calls are idempotent.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    schema = SCHEMA_PATH.read_text(encoding="utf-8")

    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema)


@contextmanager
def get_db(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that yields an open ``sqlite3.Connection``.

    * Rows are returned as ``sqlite3.Row`` objects (accessible by column name).
    * Foreign-key enforcement is enabled for the connection lifetime.
    * Commits on clean exit; rolls back on exception.

    Example::

        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                ("alice", hashed_pw),
            )
            # commit happens automatically on __exit__
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
