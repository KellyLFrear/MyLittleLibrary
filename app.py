"""
Flask web server for My Little Library.

Serves the static front-end and exposes a JSON REST API that the
front-end JavaScript calls to log in, fetch profile data, get book
recommendations, manage the user's library, and generate/save stories.

Run with:
    python app.py
or for production:
    flask --app app run --host 0.0.0.0 --port 5000

Endpoints
---------
POST  /api/auth/register        — create a new account
POST  /api/auth/login           — start a session
POST  /api/auth/logout          — end the session

GET   /api/profile              — current user's profile + reading history
GET   /api/recommendations      — current book recommendations (up to 3)
POST  /api/recommendations/generate — trigger new recommendations from RAG
GET   /api/library              — user-uploaded books
POST  /api/library              — add a book (title + body)
GET   /api/books/search?q=...   — search system + user books by title

POST  /api/story/generate       — generate a new story for the user
POST  /api/story/save           — save the most-recently-generated story
GET   /api/story/list           — list saved stories for the user
"""
from __future__ import annotations

import hashlib
import hmac
import os
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Dict
import time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Restrict CUDA libraries to GPU 0.
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    request,
    send_from_directory,
    session,
)
from werkzeug.security import check_password_hash, generate_password_hash

# ── Bootstrap Python path ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.db.connection import get_db, init_db
from src.db.repositories import (
    add_user_book,
    create_user,
    get_current_reading_profile,
    get_current_recommendations,
    get_stories_for_user,
    get_user_by_username,
    get_user_library,
    list_reading_profiles,
    save_story,
    search_books_by_title,
)

# ── App setup ─────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=str(_HERE / "front-end"), static_url_path="")

# Secret key for signing session cookies.
# In production, set the SECRET_KEY environment variable.
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


# ── DB init ───────────────────────────────────────────────────────────────────

DB_PATH = _HERE / "data" / "library.db"

with app.app_context():
    init_db(DB_PATH)

_RAG_PIPELINE = None
_LLAMA_GENERATOR = None


def get_llama_generator():
    """
    Lazily load and cache the shared LLaMA generator.

    This same generator is reused for:
    - RAG recommendation explanations
    - story generation

    Benefit:
    - avoids reloading/reinitializing LLaMA for every story request
    - avoids creating multiple CUDA llama.cpp contexts
    - makes story generation faster after the first load
    """
    global _LLAMA_GENERATOR

    if _LLAMA_GENERATOR is not None:
        return _LLAMA_GENERATOR

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    from src.rag.pipeline import LlamaCppGenerator

    _LLAMA_GENERATOR = LlamaCppGenerator(
        model_path=str(_HERE / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
        n_gpu_layers=-1,
        context_length=4096,
    )

    return _LLAMA_GENERATOR


def get_rag_pipeline():
    """
    Lazily load and cache the RAG pipeline.

    Uses GPU FAISS, CUDA embedder, CUDA CrossEncoder, and the shared cached
    LLaMA generator.
    """
    global _RAG_PIPELINE

    if _RAG_PIPELINE is not None:
        return _RAG_PIPELINE

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    from src.embeddings.embedder import ArticleEmbedder
    from src.embeddings.vector_store import FAISSVectorStore
    from src.rag.pipeline import RAGPipeline
    from src.rag.reranker import VocabAwareReranker
    from src.rag.retriever import TwoStageRetriever

    faiss_index_dir = _HERE / "data" / "faiss_index_1m_chunklevel"
    if not faiss_index_dir.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_index_dir}")

    vector_store = FAISSVectorStore.load(faiss_index_dir, use_gpu=True)

    embedder = ArticleEmbedder(device="cuda")

    reranker = VocabAwareReranker(
        use_cross_encoder=True,
        device="cuda",
    )

    retriever = TwoStageRetriever(
        vector_store,
        embedder,
        reranker,
        top_broad=100,
        top_k=3,
    )

    generator = get_llama_generator()

    _RAG_PIPELINE = RAGPipeline(retriever, generator)
    return _RAG_PIPELINE

# ── Helpers ───────────────────────────────────────────────────────────────────

def _json_error(message: str, status: int = 400) -> tuple[Response, int]:
    return jsonify({"error": message}), status


def _is_legacy_hash(stored_hash: str) -> bool:
    """Return True if the hash looks like a raw sha256 hex digest (64 hex chars, no colon prefix)."""
    return len(stored_hash) == 64 and stored_hash.isalnum() and ":" not in stored_hash


def _verify_password(plaintext: str, stored_hash: str) -> bool:
    """
    Verify a password against its stored hash.

    * New accounts use werkzeug's pbkdf2/scrypt hashing.
    * Accounts seeded by seed_test_users.py have a legacy raw-sha256 hash.
      These are accepted for backwards compatibility, and the caller should
      immediately upgrade the hash in the database (see the login endpoint).
    """
    if not _is_legacy_hash(stored_hash):
        return check_password_hash(stored_hash, plaintext)
    # Legacy check — constant-time comparison to resist timing attacks
    expected = hashlib.sha256(plaintext.encode()).hexdigest()
    return hmac.compare_digest(expected, stored_hash)


def login_required(f):
    """Decorator: return 401 JSON if user is not logged in."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return _json_error("Not authenticated", 401)
        return f(*args, **kwargs)
    return decorated


# ── Static frontend ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/main_page")
@app.route("/main_page.html")
def main_page():
    if "user_id" not in session:
        return redirect("/")
    return send_from_directory(app.static_folder, "main_page.html")


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.route("/api/auth/register", methods=["POST"])
def register():
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    display_name = (data.get("display_name") or "").strip() or None

    if not username or not password:
        return _json_error("username and password are required")
    if len(username) < 2:
        return _json_error("username must be at least 2 characters")
    if len(password) < 4:
        return _json_error("password must be at least 4 characters")

    pw_hash = generate_password_hash(password)
    try:
        with get_db(DB_PATH) as conn:
            user_id = create_user(
                conn,
                username=username,
                password_hash=pw_hash,
                display_name=display_name,
            )
    except Exception as exc:
        if "UNIQUE" in str(exc).upper():
            return _json_error("Username already taken", 409)
        raise

    session["user_id"] = user_id
    session["username"] = username
    return jsonify({"message": "Account created", "user_id": user_id}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    if not username or not password:
        return _json_error("username and password are required")

    with get_db(DB_PATH) as conn:
        user = get_user_by_username(conn, username)

    if user is None or not _verify_password(password, user["password_hash"]):
        return _json_error("Invalid username or password", 401)

    # Upgrade legacy sha256 hash to werkzeug pbkdf2 on successful login
    if _is_legacy_hash(user["password_hash"]):
        new_hash = generate_password_hash(password)
        with get_db(DB_PATH) as conn:
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE user_id = ?",
                (new_hash, user["user_id"]),
            )

    session["user_id"] = user["user_id"]
    session["username"] = user["username"]
    return jsonify({"message": "Logged in", "user_id": user["user_id"]})


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"})


# ── Profile ───────────────────────────────────────────────────────────────────

@app.route("/api/profile")
@login_required
def profile():
    user_id: int = session["user_id"]

    with get_db(DB_PATH) as conn:
        user = conn.execute(
            "SELECT user_id, username, display_name, age, avg_reading_lvl "
            "FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if user is None:
            session.clear()
            return _json_error("User not found", 404)

        current_profile = get_current_reading_profile(conn, user_id)
        history = list_reading_profiles(conn, user_id)

    # Build a trimmed history list for the reading-progress chart
    chart_data = [
        {
            "created_at": p["created_at"],
            "estimated_level": p["estimated_level"],
            "beginner_mastery": p["beginner_mastery"],
            "intermediate_mastery": p["intermediate_mastery"],
            "advanced_mastery": p["advanced_mastery"],
        }
        for p in history
    ]

    return jsonify(
        {
            "user_id": dict(user)["user_id"],
            "username": dict(user)["username"],
            "display_name": dict(user)["display_name"],
            "age": dict(user)["age"],
            "avg_reading_lvl": dict(user)["avg_reading_lvl"],
            "current_profile": current_profile,
            "reading_history": chart_data,
        }
    )


# ── Book recommendations ──────────────────────────────────────────────────────

@app.route("/api/recommendations")
@login_required
def recommendations():
    """Return up to 3 current DB-stored book recommendations for this user."""
    user_id: int = session["user_id"]
    with get_db(DB_PATH) as conn:
        recs = get_current_recommendations(conn, user_id, limit=3)
    return jsonify({"recommendations": recs})


@app.route("/api/recommendations/generate", methods=["POST"])
@login_required
def generate_recommendations():
    """
    Attempt to run the full RAG pipeline and store new recommendations.
    Falls back to existing DB recommendations if the FAISS index or
    llama-cpp-python are not available.
    """
    user_id: int = session["user_id"]
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    query: str | None = (data.get("query") or "").strip() or None

    faiss_index_dir = _HERE / "data" / "faiss_index_1m_chunklevel"
    if not faiss_index_dir.exists():
        # Fall back to whatever is already in the DB
        with get_db(DB_PATH) as conn:
            recs = get_current_recommendations(conn, user_id, limit=3)
        return jsonify({
            "recommendations": recs,
            "note": "FAISS index not found — showing existing recommendations.",
        })

    try:
        from src.rag.student_profile import StudentProfile
        from src.db.repositories import save_book_recommendations, upsert_book

        with get_db(DB_PATH) as conn:
            student = StudentProfile.from_db(user_id=user_id, conn=conn)

        pipeline = get_rag_pipeline()

        outputs = pipeline.recommend(
            query=query,
            vocab_level=student.base_level,
            top_k=3,
            student_profile=student,
        )

        app.logger.info(
            "Generated %d recommendations for user_id=%s level=%s query=%r",
            len(outputs),
            user_id,
            student.base_level,
            query,
        )

        for index, o in enumerate(outputs, start=1):
            app.logger.info(
                "Recommendation %d: title=%r coverage=%.4f difficulty=%.2f rationale=%r",
                index,
                o.title,
                float(o.coverage_ratio),
                float(o.difficulty_rating),
                (o.rationale or "")[:120],
            )

        with get_db(DB_PATH) as conn:
            profile_row = get_current_reading_profile(conn, user_id)
            profile_id = profile_row["profile_id"] if profile_row else None
            recs_to_save = []
            for o in outputs:
                bid = upsert_book(conn, title=o.title)

                coverage = min(max(float(o.coverage_ratio), 0.0), 1.0)

                reason = (o.rationale or "").strip()
                if coverage < 0.80:
                    reason = (
                        f"This article matches the topic, but it may be challenging for a "
                        f"{student.base_level.capitalize()} reader because its known-word coverage "
                        f"is only {coverage:.1%}. "
                        f"{reason}"
                    ).strip()
                if not reason:
                    if coverage >= 0.85:
                        reason = (
                            f"Recommended for a {student.base_level.capitalize()} reader because "
                            f"its known-word coverage is {coverage:.1%}, making it a good vocabulary match."
                        )
                    else:
                        reason = (
                            f"Recommended mainly because it matches the topic, but it may be challenging: "
                            f"known-word coverage is {coverage:.1%}."
                        )

                recs_to_save.append({
                    "book_id": bid,
                    "recommended_level": student.base_level.capitalize(),
                    "match_score": coverage,
                    "known_word_ratio": coverage,
                    "unknown_word_ratio": 1.0 - coverage,
                    "reason": reason,
                })
            save_book_recommendations(
                conn, user_id=user_id,
                profile_id=profile_id,
                recommendations=recs_to_save,
            )
            recs = get_current_recommendations(conn, user_id, limit=3)

        return jsonify({"recommendations": recs})

    except ImportError as exc:
        app.logger.exception("RAG imports failed")
        with get_db(DB_PATH) as conn:
            recs = get_current_recommendations(conn, user_id, limit=3)
        return jsonify({
            "recommendations": recs,
            "note": "RAG pipeline imports failed — showing existing recommendations.",
            "error": str(exc),
        }), 500

    except Exception as exc:
        app.logger.exception("RAG recommendation generation failed")
        with get_db(DB_PATH) as conn:
            recs = get_current_recommendations(conn, user_id, limit=3)
        return jsonify({
            "recommendations": recs,
            "note": "RAG recommendation generation failed — showing existing recommendations.",
            "error": str(exc),
        }), 500


# ── User library ──────────────────────────────────────────────────────────────

@app.route("/api/library", methods=["GET"])

@login_required
def get_library():
    user_id: int = session["user_id"]
    with get_db(DB_PATH) as conn:
        books = get_user_library(conn, user_id)
    return jsonify({"books": books})


@app.route("/api/library", methods=["POST"])
@login_required
def add_book():
    user_id: int = session["user_id"]
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    body = (data.get("body") or "").strip()

    if not title:
        return _json_error("title is required")
    if not body:
        return _json_error("body is required")

    with get_db(DB_PATH) as conn:
        source_id = add_user_book(conn, user_id=user_id, title=title, body_text=body)

    return jsonify({"message": "Book added", "source_id": source_id}), 201


# ── Book search ───────────────────────────────────────────────────────────────

@app.route("/api/books/search")
@login_required
def search_books():
    q = (request.args.get("q") or "").strip()
    if not q:
        return _json_error("q parameter is required")
    with get_db(DB_PATH) as conn:
        results = search_books_by_title(conn, q, limit=20)
    return jsonify({"results": results})


# ── Story generation ──────────────────────────────────────────────────────────

# Transient store: holds the most-recently-generated story per user_id
# (only lives for the lifetime of this process — save explicitly to persist)
_pending_stories: Dict[int, Dict[str, Any]] = {}


@app.route("/api/story/generate", methods=["POST"])
@login_required
def story_generate():
    """
    Generate a short story for the current user.

    Body JSON keys (all optional):
      topic    — subject / theme hint
      genre    — adventure | mystery | fantasy | sci-fi | slice-of-life
      challenge — low | medium | high
    """
    time_start = time.time()
    user_id: int = session["user_id"]
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    topic: str | None = (data.get("topic") or "").strip() or None
    genre: str = (data.get("genre") or "adventure").strip()
    challenge: str = (data.get("challenge") or "high").strip()

    faiss_index_dir = _HERE / "data" / "faiss_index_1m_chunklevel"

    # ── Try real generation ──────────────────────────────────────────────────
    try:
        from src.rag.student_profile import StudentProfile

        with get_db(DB_PATH) as conn:
            student = StudentProfile.from_db(user_id=user_id, conn=conn)

        generator = get_llama_generator()

        result = generator.generate_story(
            vocab_level=student.base_level,
            known_words=student.known_words(),
            topic=topic,
            genre=genre,
            challenge=challenge,
        )
        story_data = {
            "title": result.title,
            "story": result.story,
            "new_vocab": result.new_vocab,
            "challenge_note": result.challenge_note,
            "vocab_level": result.vocab_level,
            "was_revised": result.was_revised,
            "genre": result.genre,
            "challenge": result.challenge,
            "known_word_ratio": result.known_word_ratio,
            "unknown_word_ratio": result.unknown_word_ratio,
            "new_word_count": result.new_word_count,
            "total_word_count": result.total_word_count,
            "target_known_range": list(result.target_known_range),
            "within_target_range": result.within_target_range,
            "actual_new_words": result.actual_new_words,
        }

    except Exception as exc:
        app.logger.exception("Story generation failed")
        time_elapsed = time.time() - time_start
        # ── Stub response when model is not available ────────────────────────
        level_label = "beginner"
        with get_db(DB_PATH) as conn:
            prof = get_current_reading_profile(conn, user_id)
        if prof:
            level_label = prof.get("estimated_level", "Unknown").lower()

        topic_text = f"about {topic}" if topic else "on an interesting adventure"
        story_data = {
            "title": f"A {genre.capitalize()} Story",
            "story": (
                f"Once upon a time, a curious reader set off on a journey {topic_text}. "
                f"This story is crafted for a {level_label} reader. "
                "The full story will appear here once the language model is set up. "
                "Run the setup instructions in the README to enable story generation."
            ),
            "new_vocab": [],
            "challenge_note": "Story generation model not yet configured.",
            "vocab_level": level_label,
            "genre": genre,
            "challenge": challenge,
            "was_revised": False,
        }

    _pending_stories[user_id] = story_data
    return jsonify(story_data)


@app.route("/api/story/save", methods=["POST"])
@login_required
def story_save():
    """Save the most-recently-generated story to the database."""
    user_id: int = session["user_id"]
    story_data = _pending_stories.get(user_id)
    if not story_data:
        return _json_error("No story to save — generate one first", 400)

    with get_db(DB_PATH) as conn:
        story_id = save_story(
            conn,
            user_id=user_id,
            title=story_data["title"],
            story_text=story_data["story"],
            target_level=story_data.get("vocab_level", "Unknown").capitalize(),
            genre=story_data.get("genre"),
            challenge=story_data.get("challenge"),
            new_vocab=story_data.get("new_vocab"),
            challenge_note=story_data.get("challenge_note"),
        )

    _pending_stories.pop(user_id, None)
    return jsonify({"message": "Story saved", "story_id": story_id}), 201


@app.route("/api/story/list")
@login_required
def story_list():
    user_id: int = session["user_id"]
    with get_db(DB_PATH) as conn:
        stories = get_stories_for_user(conn, user_id, limit=20)
    return jsonify({"stories": stories})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=False, use_reloader=False)
