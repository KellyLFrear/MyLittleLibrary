"""
Query the RAG pipeline and display article recommendations.

Usage
-----
Topic query mode (original):
    CUDA_VISIBLE_DEVICES=0,2 python scripts/run_rag.py \\
        --query "history of ancient civilizations" --level beginner \\
        --repo-id bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \\
        --filename Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \\
        --tensor-split 0.5 0.5

Profile-driven mode (load user from DB — no query required):
    python scripts/run_rag.py --user-id 42 --db-path data/library.db

    The script reads the user's current reading level and known-word list from
    the database and recommends articles purely based on vocabulary coverage,
    without needing a topic query.

Vocabulary growth simulation:
    python scripts/run_rag.py --query "space and physics" --level intermediate --simulate-growth
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.embedder import ArticleEmbedder
from src.embeddings.vector_store import FAISSVectorStore
from src.rag.pipeline import LlamaCppGenerator, RAGOutput, RAGPipeline
from src.rag.reranker import VocabAwareReranker
from src.rag.retriever import TwoStageRetriever
from src.rag.student_profile import StudentProfile


def print_result(i: int, r: RAGOutput) -> None:
    print(f"\n[{i}] {r.title}")
    print(f"    Coverage  : {r.coverage_ratio:.1%}  |  Difficulty: {r.difficulty_rating}/5")
    print(f"    Summary   : {r.summary[:300]}")
    if r.new_vocab:
        print(f"    New words ({len(r.new_vocab)}):")
        for v in r.new_vocab[:6]:
            word = v.get("word", "")
            defn = v.get("definition", "")
            print(f"      * {word}: {defn}")
        if len(r.new_vocab) > 6:
            print(f"      ... and {len(r.new_vocab) - 6} more")
    print(f"    Rationale : {r.rationale[:250]}")
    print(f"    Chunks    : {', '.join(r.source_chunk_ids)}")


def _load_profile_from_db(user_id: int, db_path: str) -> StudentProfile:
    """Load a StudentProfile from the database for the given user_id."""
    from src.db.connection import get_db
    with get_db(db_path) as conn:
        return StudentProfile.from_db(user_id, conn)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Query the vocabulary-aware RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--index-dir",   default="data/faiss_index")

    # ── Input mode: topic query OR user profile ──────────────────────────────
    input_group = p.add_mutually_exclusive_group()
    input_group.add_argument(
        "--query",
        default=None,
        help="Student topic of interest (topic-query mode).",
    )
    input_group.add_argument(
        "--user-id",
        type=int,
        default=None,
        metavar="USER_ID",
        help=(
            "Load the student's reading level and known-word list from the DB "
            "and recommend articles by vocabulary coverage (profile-driven mode)."
        ),
    )

    p.add_argument(
        "--db-path",
        default="data/library.db",
        help="Path to the SQLite database (used with --user-id). Default: data/library.db",
    )
    p.add_argument("--level",           choices=["beginner", "intermediate", "advanced"],
                   default="intermediate",
                   help="Vocabulary level — ignored when --user-id is given.")
    p.add_argument("--top-k",           type=int, default=3)
    p.add_argument("--top-broad",       type=int, default=25)
    p.add_argument("--repo-id",         default="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                   help="Hugging Face repo ID for the GGUF model")
    p.add_argument("--filename",        default="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                   help="GGUF filename within the repository")
    p.add_argument("--tensor-split", type=float, nargs="+", default=None,
               help="Fraction of model layers per GPU, e.g. --tensor-split 0.5 0.5")
    p.add_argument("--cuda-visible-devices", default=None,
                   help="Optional GPU visibility override, e.g. '0,2'")
    p.add_argument("--n-gpu-layers",    type=int, default=-1,
                   help="GPU layers to offload in llama.cpp (-1 = all)")
    p.add_argument("--context-length",  type=int, default=4096,
                   help="LLM context window size in tokens (default: 4096)")
    p.add_argument("--no-cross-encoder", action="store_true",
                   help="Disable cross-encoder reranker (uses bi-encoder scores only)")
    p.add_argument("--simulate-growth", action="store_true",
                   help="Mark top result as read and re-query to show vocab growth effect")
    args = p.parse_args()

    # Require at least one of --query or --user-id
    if args.query is None and args.user_id is None:
        p.error("Provide either --query <topic> or --user-id <id>.")

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # -- Resolve input mode ---------------------------------------------------
    query: Optional[str]
    if args.user_id is not None:
        db_path = Path(args.db_path)
        if not db_path.exists():
            sys.exit(
                f"[run_rag] Database not found at '{db_path}'.\n"
                f"          Run init_db() first or check --db-path."
            )
        profile = _load_profile_from_db(args.user_id, str(db_path))
        query = args.query  # May still be provided alongside --user-id for topic filtering
        level = profile.base_level
        mode_label = f"profile-driven (user_id={args.user_id})"
    else:
        profile = StudentProfile(base_level=args.level)
        query = args.query
        level = args.level
        mode_label = "topic-query"

    # -- Load index -----------------------------------------------------------
    if not Path(args.index_dir).exists():
        sys.exit(
            f"[run_rag] Index not found at '{args.index_dir}'.\n"
            f"          Run: python scripts/build_index.py"
        )

    store     = FAISSVectorStore.load(args.index_dir)
    embedder  = ArticleEmbedder(device="cuda")
    reranker  = VocabAwareReranker(
        use_cross_encoder=not args.no_cross_encoder,
        device="cuda",
    )
    retriever = TwoStageRetriever(
        store, embedder, reranker,
        top_broad=args.top_broad,
        top_k=args.top_k,
    )

    # -- Generator ------------------------------------------------------------
    generator = LlamaCppGenerator(
        repo_id=args.repo_id,
        filename=args.filename,
        n_gpu_layers=args.n_gpu_layers,
        tensor_split=args.tensor_split,
        context_length=args.context_length,
    )

    if args.n_gpu_layers == -1 and not args.no_cross_encoder and args.top_broad > 25:
        print(
            "[run_rag] Warning: high VRAM pressure configuration detected "
            "(full offload + cross-encoder + high top_broad). "
            "If CUDA aborts, try --top-broad 10 and/or --no-cross-encoder.",
            file=sys.stderr,
        )

    pipeline = RAGPipeline(retriever, generator)

    # -- Initial query --------------------------------------------------------
    print(f"\nMode     : {mode_label}")
    print(f"Query    : {query or '(none — profile-driven)'}")
    print(f"Level    : {level}  |  {profile.summary()}")
    print(f"Model    : {args.repo_id} / {args.filename}")
    print("=" * 60)

    results = pipeline.recommend(
        query, level,
        top_k=args.top_k,
        student_profile=profile,
    )

    if not results:
        print("\nNo articles found within the vocabulary coverage window.")
        return

    for i, r in enumerate(results, 1):
        print_result(i, r)

    # -- Vocabulary growth simulation -----------------------------------------
    if args.simulate_growth and results:
        top_chunk_id = results[0].source_chunk_ids[0]
        chunk_obj = next((c for c in store.chunks if c.chunk_id == top_chunk_id), None)

        if chunk_obj:
            newly_learned = profile.mark_as_read(chunk_obj)
            print(f"\n{'-'*60}")
            print(f"[VOCAB GROWTH] Student read: '{results[0].title}'")
            if newly_learned:
                print(f"  Newly learned : {', '.join(newly_learned)}")
            else:
                print("  No new words (all already known).")
            print(f"  {profile.summary()}")

            print(f"\n[RE-QUERY] Same input with updated vocabulary:")
            print("=" * 60)

            updated = pipeline.recommend(
                query, level,
                top_k=args.top_k,
                student_profile=profile,
            )
            if updated:
                for i, r in enumerate(updated, 1):
                    print_result(i, r)
            else:
                print("No results after vocabulary update.")


if __name__ == "__main__":
    main()
