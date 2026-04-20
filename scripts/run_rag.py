"""
Query the RAG pipeline and display article recommendations.

Usage
-----
Template generator (no model needed, useful for testing retrieval):
    python scripts/run_rag.py --query "space exploration and black holes" --level intermediate --generator template

LLaMA GGUF model via llama.cpp (GPU):
    python scripts/run_rag.py --query "history of ancient civilizations" --level beginner \
        --model-path models/llama-3.1-8b-instruct.Q4_K_M.gguf

Vocabulary growth simulation:
    python scripts/run_rag.py --query "space and physics" --level intermediate \
        --simulate-growth --generator template
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.embedder import ArticleEmbedder
from src.embeddings.vector_store import FAISSVectorStore
from src.rag.pipeline import LlamaCppGenerator, RAGOutput, RAGPipeline, TemplateGenerator
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


def main() -> None:
    p = argparse.ArgumentParser(description="Query the vocabulary-aware RAG pipeline.")
    p.add_argument("--index-dir",       default="data/faiss_index")
    p.add_argument("--query",           required=True, help="Student topic of interest")
    p.add_argument("--level",           choices=["beginner", "intermediate", "advanced"],
                   default="intermediate")
    p.add_argument("--top-k",           type=int, default=3)
    p.add_argument("--top-broad",       type=int, default=50)
    p.add_argument("--device",          default="cuda",
                   help="Torch device for embedder/reranker: 'cuda' (default) or 'cpu'")
    p.add_argument("--generator",       choices=["llama", "template"], default="llama")
    p.add_argument("--model-path",      default="models/llama-3.1-8b-instruct.Q4_K_M.gguf",
                   help="Path to GGUF model file (used when --generator llama)")
    p.add_argument("--n-gpu-layers",    type=int, default=-1,
                   help="GPU layers to offload in llama.cpp (-1 = all)")
    p.add_argument("--no-cross-encoder", action="store_true",
                   help="Disable cross-encoder reranker (uses bi-encoder scores only)")
    p.add_argument("--no-gpu-index",    action="store_true",
                   help="Disable GPU acceleration for the FAISS index")
    p.add_argument("--simulate-growth", action="store_true",
                   help="Mark top result as read and re-query to show vocab growth effect")
    args = p.parse_args()

    # -- Load index ------------------------------------------------------------
    if not Path(args.index_dir).exists():
        sys.exit(
            f"[run_rag] Index not found at '{args.index_dir}'.\n"
            f"          Run: python scripts/build_index.py"
        )

    use_gpu_index = not args.no_gpu_index
    store     = FAISSVectorStore.load(args.index_dir, use_gpu=use_gpu_index)
    embedder  = ArticleEmbedder(device=args.device)
    reranker  = VocabAwareReranker(
        use_cross_encoder=not args.no_cross_encoder,
        device=args.device,
    )
    retriever = TwoStageRetriever(
        store, embedder, reranker,
        top_broad=args.top_broad,
        top_k=args.top_k,
    )

    # -- Generator ------------------------------------------------------------
    if args.generator == "llama":
        generator = LlamaCppGenerator(
            model_path=args.model_path,
            n_gpu_layers=args.n_gpu_layers,
        )
    else:
        generator = TemplateGenerator()

    pipeline = RAGPipeline(retriever, generator)
    profile  = StudentProfile(base_level=args.level)

    # -- Initial query --------------------------------------------------------
    print(f"\nQuery    : {args.query}")
    print(f"Level    : {args.level}  |  {profile.summary()}")
    print(f"Generator: {args.generator}")
    print("=" * 60)

    results = pipeline.recommend(
        args.query, args.level,
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

            print(f"\n[RE-QUERY] Same query with updated vocabulary:")
            print("=" * 60)

            updated = pipeline.recommend(
                args.query, args.level,
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
