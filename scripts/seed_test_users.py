"""
Seed three test users into the SQLite database, one per reading level.

    python scripts/seed_test_users.py [--db-path data/library.db]

What this does
--------------
1. Initialises the DB schema (idempotent).
2. Loads every word from the three vocab list files into vocabulary_terms.
3. Creates three users if they don't already exist:
       test_beginner       / password: test123
       test_intermediate   / password: test123
       test_advanced       / password: test123
4. Gives each user a current reading_profile at their level.
5. Marks all words at or below their level as 'known' in user_word_knowledge.
6. Seeds the top 5 highest-coverage articles per level as books.
7. Creates book_recommendations for each user matching their reading level.

Run it again safely — users and vocab terms are upserted, not duplicated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.connection import get_db, init_db
from src.db.repositories import (
    create_reading_profile,
    create_user,
    get_current_reading_profile,
    get_user_by_username,
    save_book_recommendations,
    upsert_book,
    upsert_vocabulary_term,
    upsert_word_knowledge,
)

# ── Constants ─────────────────────────────────────────────────────────────────

VOCAB_DIR = Path(__file__).resolve().parent.parent / "data" / "vocab"
ARTICLE_STATS_PATH = Path(__file__).resolve().parent.parent / "outputs" / "article_stats.jsonl"
BOOKS_PER_LEVEL = 5   # number of best-fit articles to seed per level

VOCAB_FILES: dict[str, Path] = {
    "Beginner":     VOCAB_DIR / "beginner_1000.txt",
    "Intermediate": VOCAB_DIR / "intermediate_3000.txt",
    "Advanced":     VOCAB_DIR / "advanced_6000.txt",
}

# Words a beginner knows, an intermediate knows, etc.
# (cumulative: an advanced user also knows beginner + intermediate words)
LEVEL_KNOWN_WORDS: dict[str, list[str]] = {
    "Beginner":     ["Beginner"],
    "Intermediate": ["Beginner", "Intermediate"],
    "Advanced":     ["Beginner", "Intermediate", "Advanced"],
}

TEST_USERS: list[dict] = [
    {
        "username":     "test_beginner",
        "display_name": "Test Beginner",
        "level":        "Beginner",
        "age":          10,
    },
    {
        "username":     "test_intermediate",
        "display_name": "Test Intermediate",
        "level":        "Intermediate",
        "age":          14,
    },
    {
        "username":     "test_advanced",
        "display_name": "Test Advanced",
        "level":        "Advanced",
        "age":          18,
    },
]

_PASSWORD_HASH = hashlib.sha256(b"test123").hexdigest()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_vocab_file(path: Path) -> list[str]:
    if not path.exists():
        print(f"  [warn] Vocab file not found: {path}", file=sys.stderr)
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _mastery_for_level(level: str) -> dict[str, float]:
    """Return realistic mastery ratios for a user at the given level."""
    tbl = {
        "Beginner":     {"beginner": 0.95, "intermediate": 0.0,  "advanced": 0.0},
        "Intermediate": {"beginner": 1.0,  "intermediate": 0.85, "advanced": 0.0},
        "Advanced":     {"beginner": 1.0,  "intermediate": 1.0,  "advanced": 0.80},
    }
    return tbl.get(level, {})


def _select_top_articles(n: int = BOOKS_PER_LEVEL) -> dict[str, list[dict]]:
    """
    Read article_stats.jsonl, merge per-level rows into one article object,
    and return the top ``n`` articles per level sorted by that level's
    coverage ratio (highest first).

    Returns a dict: level → list of article dicts with keys
    id, title, coverage_ratio (per-level dict), new_words (per-level dict).
    """
    if not ARTICLE_STATS_PATH.exists():
        print(f"  [warn] article_stats.jsonl not found at {ARTICLE_STATS_PATH}", file=sys.stderr)
        return {}

    merged: dict[str, dict] = {}
    with ARTICLE_STATS_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            aid = str(row.get("ID", "")).strip()
            if not aid:
                continue
            art = merged.setdefault(aid, {
                "id": aid,
                "title": row.get("Title", ""),
                "coverage_ratio": {},
                "new_words": {},
            })
            lvl = row.get("Level", "").strip()   # 'Beginner' / 'Intermediate' / 'Advanced'
            if lvl:
                art["coverage_ratio"][lvl] = float(row.get("Coverage_Ratio", 0) or 0)
                raw_nw = row.get("New_Words", "")
                art["new_words"][lvl] = (
                    [w.strip() for w in raw_nw.split(",") if w.strip()]
                    if isinstance(raw_nw, str) else []
                )

    result: dict[str, list[dict]] = {}
    for lvl in ("Beginner", "Intermediate", "Advanced"):
        ranked = sorted(
            merged.values(),
            key=lambda a: a["coverage_ratio"].get(lvl, 0),
            reverse=True,
        )
        result[lvl] = ranked[:n]

    return result

# ── Main ──────────────────────────────────────────────────────────────────────Allow

def main(db_path: str) -> None:
    print(f"Initialising DB at: {db_path}")
    init_db(db_path)

    # ── Step 1: load all vocab terms into DB ─────────────────────────────────
    print("\nLoading vocabulary terms …")
    term_ids: dict[str, dict[str, int]] = {}  # level → {word → term_id}

    with get_db(db_path) as conn:
        for level, path in VOCAB_FILES.items():
            words = _load_vocab_file(path)
            term_ids[level] = {}
            for word in words:
                tid = upsert_vocabulary_term(conn, word=word, level=level)
                term_ids[level][word] = tid
            print(f"  {level}: {len(words)} words loaded")

    # ── Step 2: create/update test users ─────────────────────────────────────
    print("\nCreating test users …")

    for user_spec in TEST_USERS:
        username = user_spec["username"]
        level    = user_spec["level"]
        mastery  = _mastery_for_level(level)

        with get_db(db_path) as conn:
            existing = get_user_by_username(conn, username)
            if existing:
                uid = existing["user_id"]
                print(f"  {username} already exists (user_id={uid}) — skipping user creation")
            else:
                uid = create_user(
                    conn,
                    username=username,
                    password_hash=_PASSWORD_HASH,
                    display_name=user_spec["display_name"],
                    age=user_spec["age"],
                    avg_reading_lvl=level,
                )
                print(f"  Created {username} (user_id={uid})")

            # Count totals for the profile
            known_levels = LEVEL_KNOWN_WORDS[level]
            total_known = sum(len(term_ids.get(lv, {})) for lv in known_levels)

            # Retire old profile + insert fresh one
            profile = get_current_reading_profile(conn, uid)
            if profile:
                print(f"    Replacing existing {level} profile …")

            create_reading_profile(
                conn,
                user_id=uid,
                estimated_level=level,
                known_beginner_words=len(term_ids.get("Beginner", {})),
                known_intermediate_words=(
                    len(term_ids.get("Intermediate", {})) if level in ("Intermediate", "Advanced") else 0
                ),
                known_advanced_words=(
                    len(term_ids.get("Advanced", {})) if level == "Advanced" else 0
                ),
                total_known_words=total_known,
                beginner_mastery=mastery.get("beginner", 0.0),
                intermediate_mastery=mastery.get("intermediate", 0.0),
                advanced_mastery=mastery.get("advanced", 0.0),
                confidence=0.9,
            )
            print(f"    Reading profile created: level={level}, known_words={total_known}")

        # Mark words as known (separate transaction per user to keep it manageable)
        with get_db(db_path) as conn:
            known_levels = LEVEL_KNOWN_WORDS[level]
            count = 0
            for lv in known_levels:
                for word, tid in term_ids.get(lv, {}).items():
                    upsert_word_knowledge(
                        conn,
                        user_id=uid,
                        term_id=tid,
                        knowledge_status="known",
                        confidence=0.95,
                    )
                    count += 1
            print(f"    Marked {count} words as 'known'")

    print("\nDone! Test users are ready.")
    print("  Username              Password   Level")
    print("  --------------------  ---------  ------------")
    for u in TEST_USERS:
        print(f"  {u['username']:<22}  test123    {u['level']}")

    # ── Step 3: seed books from article_stats.jsonl ───────────────────────────
    print(f"\nSeeding top {BOOKS_PER_LEVEL} articles per level as books …")
    top_articles = _select_top_articles(BOOKS_PER_LEVEL)

    # book_id lookup: article_id → db book_id
    book_id_map: dict[str, int] = {}
    # coverage lookup: article_id → coverage_ratio dict (per level)
    book_coverage: dict[str, dict[str, float]] = {}

    with get_db(db_path) as conn:
        for lvl, articles in top_articles.items():
            for art in articles:
                bid = upsert_book(
                    conn,
                    title=art["title"],
                    source_hash=art["id"],   # use Wikipedia article ID as stable hash
                )
                book_id_map[art["id"]] = bid
                book_coverage[art["id"]] = art["coverage_ratio"]
                print(f"  [{lvl}] book_id={bid}  {art['title'][:60]}")

    # ── Step 4: create book recommendations for each user ─────────────────────
    print("\nCreating book recommendations …")

    # Level → list of article dicts with the highest coverage for that level
    level_article_lists = {lvl: arts for lvl, arts in top_articles.items()}

    with get_db(db_path) as conn:
        for user_spec in TEST_USERS:
            username = user_spec["username"]
            level    = user_spec["level"]

            user_row = get_user_by_username(conn, username)
            if not user_row:
                print(f"  [warn] {username} not found — skipping recommendations")
                continue
            uid = user_row["user_id"]

            profile_row = conn.execute(
                "SELECT profile_id FROM user_reading_profiles WHERE user_id=? AND is_current=1",
                (uid,),
            ).fetchone()
            profile_id = profile_row["profile_id"] if profile_row else None

            articles = level_article_lists.get(level, [])
            if not articles:
                continue

            # Normalise match_score: highest coverage article → 1.0
            max_cov = max(a["coverage_ratio"].get(level, 0) for a in articles) or 1.0

            recs = []
            for art in articles:
                aid = art["id"]
                bid = book_id_map.get(aid)
                if bid is None:
                    continue

                cov = art["coverage_ratio"].get(level, 0)
                match_score = round(min(cov / max_cov, 1.0), 4)
                # unknown_word_ratio = fraction of level-specific new words vs total words
                new_word_count = len(art["new_words"].get(level, []))
                recs.append({
                    "book_id":            bid,
                    "recommended_level":  level,
                    "match_score":        match_score,
                    "known_word_ratio":   round(cov, 4),
                    "unknown_word_ratio": None,  # not available from article_stats
                    "reason": (
                        f"{art['title'][:60]} has the highest {level.lower()} "
                        f"vocabulary coverage ({cov:.1%}) among indexed articles."
                    ),
                })

            save_book_recommendations(conn, user_id=uid, profile_id=profile_id, recommendations=recs)
            print(f"  {username}: {len(recs)} recommendations saved (level={level})")

    print("\nAll done!")
    print("  Username              Password   Level")
    print("  --------------------  ---------  ------------")
    for u in TEST_USERS:
        print(f"  {u['username']:<22}  test123    {u['level']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Seed three test users into the library DB.")
    ap.add_argument("--db-path", default="data/library.db", help="Path to the SQLite DB")
    args = ap.parse_args()
    main(args.db_path)
