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
import os
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Dict

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _json_error(message: str, status: int = 400) -> tuple[Response, int]:
    return jsonify({"error": message}), status


def _check_legacy_hash(plaintext: str, stored_hash: str) -> bool:
    """
    Accept passwords that were hashed with the legacy sha256 scheme used by
    seed_test_users.py  (hashlib.sha256(b"password").hexdigest()).
    """
    return hashlib.sha256(plaintext.encode()).hexdigest() == stored_hash


def _verify_password(plaintext: str, stored_hash: str) -> bool:
    """Try werkzeug pbkdf2 first, then fall back to legacy sha256."""
    if stored_hash.startswith("pbkdf2:") or stored_hash.startswith("scrypt:"):
        return check_password_hash(stored_hash, plaintext)
    return _check_legacy_hash(plaintext, stored_hash)


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

    faiss_index_dir = _HERE / "data" / "faiss_index"
    if not faiss_index_dir.exists():
        # Fall back to whatever is already in the DB
        with get_db(DB_PATH) as conn:
            recs = get_current_recommendations(conn, user_id, limit=3)
        return jsonify({
            "recommendations": recs,
            "note": "FAISS index not found — showing existing recommendations.",
        })

    try:
        from src.embeddings.vector_store import FAISSVectorStore
        from src.rag.pipeline import LlamaCppGenerator, RAGPipeline
        from src.rag.retriever import TwoStageRetriever
        from src.rag.student_profile import StudentProfile
        from src.db.repositories import save_book_recommendations, upsert_book

        with get_db(DB_PATH) as conn:
            student = StudentProfile.from_db(user_id=user_id, conn=conn)

        vector_store = FAISSVectorStore.load(faiss_index_dir)
        retriever = TwoStageRetriever(vector_store)
        generator = LlamaCppGenerator(
            repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        )
        pipeline = RAGPipeline(retriever, generator)

        outputs = pipeline.recommend(
            query=query,
            vocab_level=student.base_level,
            top_k=3,
            student_profile=student,
        )

        with get_db(DB_PATH) as conn:
            profile_row = get_current_reading_profile(conn, user_id)
            profile_id = profile_row["profile_id"] if profile_row else None
            recs_to_save = []
            for o in outputs:
                bid = upsert_book(conn, title=o.title)
                recs_to_save.append({
                    "book_id": bid,
                    "match_score": min(o.coverage_ratio, 1.0),
                    "reason": o.rationale,
                })
            save_book_recommendations(
                conn, user_id=user_id,
                profile_id=profile_id,
                recommendations=recs_to_save,
            )
            recs = get_current_recommendations(conn, user_id, limit=3)

        return jsonify({"recommendations": recs})

    except ImportError:
        with get_db(DB_PATH) as conn:
            recs = get_current_recommendations(conn, user_id, limit=3)
        return jsonify({
            "recommendations": recs,
            "note": "RAG pipeline not available — showing existing recommendations.",
        })


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
    user_id: int = session["user_id"]
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    topic: str | None = (data.get("topic") or "").strip() or None
    genre: str = (data.get("genre") or "adventure").strip()
    challenge: str = (data.get("challenge") or "medium").strip()

    faiss_index_dir = _HERE / "data" / "faiss_index"

    # ── Try real generation ──────────────────────────────────────────────────
    try:
        from llama_cpp import Llama  # noqa: F401 — just probe availability
        from src.rag.pipeline import LlamaCppGenerator
        from src.rag.student_profile import StudentProfile

        with get_db(DB_PATH) as conn:
            student = StudentProfile.from_db(user_id=user_id, conn=conn)

        generator = LlamaCppGenerator(
            repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        )
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
            "genre": result.genre,
            "challenge": result.challenge,
        }

    except (ImportError, Exception):
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
    app.run(host="0.0.0.0", port=port, debug=debug)
