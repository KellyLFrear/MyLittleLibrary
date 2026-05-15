"""
RAG Evaluation Script
=====================
Computes:
  - Precision@k, Recall@k, MRR for k in {5, 10, 20}
  - Retrieval ablation modes:
      * semantic_only: FAISS semantic ranking only
      * coverage_filter: FAISS semantic ranking followed by a hard coverage filter
      * weighted_rerank: final vocabulary-aware weighted reranker
  - ROUGE-1/2/L between generated summaries and article reference text
  - BERTScore (optional, requires bert-score package)
  - Vocabulary-awareness %: results falling within coverage window

Saves full results to data/eval_results.json.

Examples
--------
    # Final weighted reranker evaluation
    python scripts/evaluate_rag.py \
        --mode weighted_rerank \
        --queries data/eval_queries.json \
        --output data/eval_results_weighted.json

    # Ablation runs for the report table
    python scripts/evaluate_rag.py --mode semantic_only --skip-generation \
        --queries data/eval_queries.json --output data/eval_semantic_only.json
    python scripts/evaluate_rag.py --mode coverage_filter --skip-generation \
        --queries data/eval_queries.json --output data/eval_coverage_filter.json
    python scripts/evaluate_rag.py --mode weighted_rerank --skip-generation \
        --queries data/eval_queries.json --output data/eval_weighted_rerank.json

    # Full generation metrics, including BERTScore
    python scripts/evaluate_rag.py --mode weighted_rerank --with-bertscore
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.chunker import ArticleChunk
from src.embeddings.embedder import ArticleEmbedder
from src.embeddings.vector_store import FAISSVectorStore
from src.rag.pipeline import LlamaCppGenerator
from src.rag.reranker import VocabAwareReranker


# ── IR metric helpers ─────────────────────────────────────────────────────────

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of top-k retrieved articles that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for r in top_k if r in relevant) / len(top_k)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of relevant articles found in top-k."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for r in top_k if r in relevant) / len(relevant)


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """1 / rank of the first relevant result; 0 if none are found."""
    for rank, rid in enumerate(retrieved, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def mean_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    """Average metric dictionaries across queries."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: round(sum(q.get(k, 0.0) for q in per_query) / len(per_query), 4) for k in keys}


# ── Text metric helpers ───────────────────────────────────────────────────────

def compute_rouge(hypothesis: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("[evaluate_rag] rouge-score not installed. Run: pip install rouge-score")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def article_reference(chunk_text: str, max_words: int = 100) -> str:
    """Use the first max_words of the retrieved chunk as the reference text."""
    return " ".join((chunk_text or "").split()[:max_words])


# ── Ranking helpers ───────────────────────────────────────────────────────────

def chunk_coverage(chunk: ArticleChunk, vocab_level: str) -> float:
    """Safely read a chunk's known-word coverage ratio for a vocabulary level."""
    try:
        return float(chunk.coverage_ratio.get(vocab_level, 0.0))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def in_coverage_window(chunk: ArticleChunk, vocab_level: str, window: Tuple[float, float]) -> bool:
    ratio = chunk_coverage(chunk, vocab_level)
    return window[0] <= ratio <= window[1]


def dedupe_article_ids(ranked: List[Tuple[ArticleChunk, float]]) -> List[str]:
    """Return article IDs in rank order, keeping only the first chunk per article."""
    seen_ids: List[str] = []
    seen_set: Set[str] = set()
    for chunk, _score in ranked:
        if chunk.article_id not in seen_set:
            seen_ids.append(chunk.article_id)
            seen_set.add(chunk.article_id)
    return seen_ids


def top_unique_chunks(ranked: List[Tuple[ArticleChunk, float]], top_k: int) -> List[ArticleChunk]:
    """Return top chunks, deduplicated to one chunk per article."""
    chunks: List[ArticleChunk] = []
    seen_set: Set[str] = set()
    for chunk, _score in ranked:
        if chunk.article_id in seen_set:
            continue
        chunks.append(chunk)
        seen_set.add(chunk.article_id)
        if len(chunks) >= top_k:
            break
    return chunks


def rank_candidates(
    *,
    mode: str,
    query: str,
    broad: List[Tuple[ArticleChunk, float]],
    reranker: Optional[VocabAwareReranker],
    vocab_level: str,
    coverage_window: Tuple[float, float],
) -> List[Tuple[ArticleChunk, float]]:
    """
    Apply one of the report ablation modes.

    semantic_only:
        Use the original FAISS semantic scores only.
    coverage_filter:
        Keep only chunks inside the coverage window, preserving FAISS semantic rank.
    weighted_rerank:
        Use the final vocabulary-aware weighted reranker.
    """
    if mode == "semantic_only":
        return list(broad)

    if mode == "coverage_filter":
        return [
            (chunk, score)
            for chunk, score in broad
            if in_coverage_window(chunk, vocab_level, coverage_window)
        ]

    if mode == "weighted_rerank":
        if reranker is None:
            raise ValueError("weighted_rerank mode requires a VocabAwareReranker")
        return reranker.rerank(
            query=query,
            candidates=broad,
            vocab_level=vocab_level,
            coverage_window=coverage_window,
        )

    raise ValueError(f"Unsupported evaluation mode: {mode}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the RAG pipeline on held-out queries.")
    p.add_argument("--index-dir", default="data/faiss_index_1m_chunklevel",)
    p.add_argument("--queries", default="data/eval_queries.json")
    p.add_argument("--output", default="data/eval_results.json")
    p.add_argument(
        "--mode",
        choices=["semantic_only", "coverage_filter", "weighted_rerank"],
        default="weighted_rerank",
        help=(
            "Ablation mode. semantic_only uses FAISS scores only; coverage_filter "
            "hard-filters to the coverage window; weighted_rerank uses the final "
            "vocabulary-aware reranker."
        ),
    )
    p.add_argument("--repo-id", default="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                   help="Hugging Face repo ID for the GGUF model")
    p.add_argument("--filename", default="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                   help="GGUF filename or local path")
    p.add_argument("--model-path", default=None,
                   help="Optional explicit local GGUF path. Overrides --filename path lookup.")
    p.add_argument("--n-gpu-layers", type=int, default=-1)
    p.add_argument("--context-length", type=int, default=4096,
                   help="LLM context window size in tokens")
    p.add_argument("--top-broad", type=int, default=100,
                   help="Number of candidates retrieved from FAISS before filtering/reranking")
    p.add_argument("--generation-top-k", type=int, default=3,
                   help="Number of top recommendations to generate summaries for")
    p.add_argument("--skip-generation", action="store_true",
                   help="Only compute retrieval/vocabulary metrics; do not load LLaMA or compute ROUGE/BERTScore")
    p.add_argument("--no-cross-encoder", action="store_true",
                   help="Disable cross-encoder reranker in weighted_rerank mode")
    p.add_argument("--semantic-weight", type=float, default=0.35,
                   help="Semantic weight for weighted_rerank mode")
    p.add_argument("--vocab-weight", type=float, default=0.65,
                   help="Vocabulary-fit weight for weighted_rerank mode")
    p.add_argument("--with-bertscore", action="store_true",
                   help="Also compute BERTScore over generated summary/reference pairs")
    p.add_argument("--tensor-split", type=float, nargs="+", default=None,
                   help="Fraction of model layers per visible GPU, e.g. --tensor-split 0.5 0.5")
    p.add_argument("--cuda-visible-devices", default=None,
                   help="Optional GPU visibility override, e.g. '0,2'")
    p.add_argument("--coverage-low", type=float, default=0.85)
    p.add_argument("--coverage-high", type=float, default=0.97)
    args = p.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    coverage_window = (args.coverage_low, args.coverage_high)
    ks = [5, 10, 20]

    # -- Load retrieval system ------------------------------------------------
    store = FAISSVectorStore.load(args.index_dir)
    embedder = ArticleEmbedder(device="cuda")

    reranker: Optional[VocabAwareReranker] = None
    if args.mode == "weighted_rerank":
        reranker = VocabAwareReranker(
            semantic_weight=args.semantic_weight,
            vocab_weight=args.vocab_weight,
            use_cross_encoder=not args.no_cross_encoder,
            device="cuda",
        )

    # FAISS GPU indexes support top-k up to 2048 per query.
    top_broad = min(len(store.chunks), 2048, max(1, args.top_broad))

    generator: Optional[LlamaCppGenerator] = None
    if not args.skip_generation:
        generator = LlamaCppGenerator(
            repo_id=args.repo_id,
            filename=args.filename,
            model_path=args.model_path,
            n_gpu_layers=args.n_gpu_layers,
            tensor_split=args.tensor_split,
            context_length=args.context_length,
        )

    # -- Load ground truth ----------------------------------------------------
    eval_queries = json.loads(Path(args.queries).read_text())
    print(f"Evaluation mode: {args.mode}")
    print(f"Loaded {len(eval_queries)} evaluation queries.")
    print(f"Index has {len(store.chunks)} chunks across "
          f"{len(set(c.article_id for c in store.chunks))} articles.")
    print(f"Coverage window: [{coverage_window[0]:.0%}, {coverage_window[1]:.0%}]\n")

    effective_ks = [min(k, len(store.chunks)) for k in ks]

    per_query_ir: List[Dict[str, float]] = []
    per_query_rouge: List[Dict[str, float]] = []
    generated_pairs: List[Dict[str, str]] = []
    all_coverage_ratios: List[float] = []
    in_window_count = 0
    total_results = 0
    structured_attempts = 0
    structured_successes = 0
    query_details = []

    # -- Evaluation loop ------------------------------------------------------
    for item in eval_queries:
        query = item["query"]
        relevant = {str(x) for x in item.get("relevant_ids", [])}
        level = item.get("level", "intermediate")

        # Stage 1: broad semantic retrieval.
        q_emb = embedder.embed_query(query)
        broad = store.search(q_emb, top_k=top_broad)

        # Stage 2: selected ablation mode.
        ranked = rank_candidates(
            mode=args.mode,
            query=query,
            broad=broad,
            reranker=reranker,
            vocab_level=level,
            coverage_window=coverage_window,
        )

        # Article-level rank list for IR metrics.
        seen_ids = dedupe_article_ids(ranked)

        ir_row: Dict[str, float] = {}
        for k, ek in zip(ks, effective_ks):
            ir_row[f"precision@{k}"] = precision_at_k(seen_ids, relevant, ek)
            ir_row[f"recall@{k}"] = recall_at_k(seen_ids, relevant, ek)
        ir_row["mrr"] = reciprocal_rank(seen_ids, relevant)
        per_query_ir.append(ir_row)

        # Vocabulary-awareness over top-10 chunks for this configuration.
        top_chunks = [chunk for chunk, _score in ranked[:10]]
        for chunk in top_chunks:
            ratio = chunk_coverage(chunk, level)
            all_coverage_ratios.append(ratio)
            total_results += 1
            if coverage_window[0] <= ratio <= coverage_window[1]:
                in_window_count += 1

        generation_rows = []
        rouge_rows = []

        if generator is not None:
            for chunk in top_unique_chunks(ranked, args.generation_top_k):
                structured_attempts += 1
                reference = article_reference(chunk.text)
                try:
                    output = generator.generate(
                        query=query,
                        context_chunks=[chunk],
                        new_words=chunk.new_words.get(level, []),
                        vocab_level=level,
                    )
                    structured_successes += 1

                    rouge_row = compute_rouge(output.summary, reference)
                    rouge_rows.append(rouge_row)
                    per_query_rouge.append(rouge_row)

                    generated_pairs.append({
                        "hypothesis": output.summary,
                        "reference": reference,
                        "query": query,
                        "level": level,
                        "article_id": chunk.article_id,
                        "chunk_id": chunk.chunk_id,
                    })

                    generation_rows.append({
                        "article_id": chunk.article_id,
                        "chunk_id": chunk.chunk_id,
                        "title": output.title,
                        "summary": output.summary,
                        "coverage_ratio": output.coverage_ratio,
                        "rouge": rouge_row,
                        "structured_valid": True,
                    })

                except Exception as exc:  # keep evaluation running and record failure
                    generation_rows.append({
                        "article_id": chunk.article_id,
                        "chunk_id": chunk.chunk_id,
                        "title": chunk.title,
                        "summary": "",
                        "coverage_ratio": chunk_coverage(chunk, level),
                        "rouge": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
                        "structured_valid": False,
                        "error": str(exc),
                    })

        query_details.append({
            "query": query,
            "level": level,
            "relevant": list(relevant),
            "retrieved": seen_ids,
            "ir_metrics": ir_row,
            "rouge": rouge_rows,
            "generated": generation_rows,
            "coverage_ratios_top10": [chunk_coverage(c, level) for c in top_chunks],
        })

        print(f"  Query: '{query[:50]}...' | "
              f"P@5={ir_row['precision@5']:.2f} R@5={ir_row['recall@5']:.2f} "
              f"MRR={ir_row['mrr']:.2f} | top chunks={len(top_chunks)}")

    # -- Aggregate metrics ----------------------------------------------------
    mean_ir = mean_metrics(per_query_ir)
    mean_rouge = mean_metrics(per_query_rouge) if per_query_rouge else {}
    vocab_awareness_pct = (in_window_count / total_results * 100) if total_results else 0.0
    structured_output_valid_rate = (
        structured_successes / structured_attempts * 100 if structured_attempts else None
    )

    print("\n" + "=" * 60)
    print("RETRIEVAL METRICS (mean across queries)")
    print("=" * 60)
    print(f"  {'Metric':<20} {'k=5':>8} {'k=10':>8} {'k=20':>8}")
    print(f"  {'-' * 44}")
    for metric in ["precision", "recall"]:
        row = "  " + f"{metric:<20}"
        for k in ks:
            row += f"  {mean_ir.get(f'{metric}@{k}', 0.0):>6.4f}"
        print(row)
    print(f"  {'MRR':<20}  {mean_ir.get('mrr', 0.0):>6.4f}")

    if not args.skip_generation:
        print("\nGENERATION QUALITY (mean ROUGE vs article reference text)")
        print("=" * 60)
        for m in ["rouge1", "rouge2", "rougeL"]:
            print(f"  {m:<10}: {mean_rouge.get(m, 0.0):.4f}")
        print(f"  structured output validity: {structured_output_valid_rate or 0.0:.1f}%")

    print("\nVOCABULARY AWARENESS")
    print("=" * 60)
    print(f"  Results within coverage window [{coverage_window[0]:.0%}, {coverage_window[1]:.0%}]: "
          f"{in_window_count}/{total_results} = {vocab_awareness_pct:.1f}%")

    # -- BERTScore ------------------------------------------------------------
    bertscore_result: Dict[str, float] = {}
    if args.with_bertscore:
        if args.skip_generation:
            print("\n[evaluate_rag] --with-bertscore ignored because --skip-generation was set.")
        elif not generated_pairs:
            print("\n[evaluate_rag] No generated summary/reference pairs available for BERTScore.")
        else:
            try:
                from bert_score import score as bs_score

                hypotheses = [pair["hypothesis"] for pair in generated_pairs]
                references = [pair["reference"] for pair in generated_pairs]
                P, R, F = bs_score(hypotheses, references, lang="en", verbose=False)
                bertscore_result = {
                    "precision": round(P.mean().item(), 4),
                    "recall": round(R.mean().item(), 4),
                    "f1": round(F.mean().item(), 4),
                    "n_pairs": len(generated_pairs),
                }
                print(f"\nBERTScore: P={bertscore_result['precision']:.4f} "
                      f"R={bertscore_result['recall']:.4f} "
                      f"F1={bertscore_result['f1']:.4f} "
                      f"n={bertscore_result['n_pairs']}")
            except ImportError:
                print("\n[evaluate_rag] bert-score not installed. Run: pip install bert-score")

    # -- Save results ---------------------------------------------------------
    output = {
        "mode": args.mode,
        "mean_ir_metrics": mean_ir,
        "mean_rouge": mean_rouge,
        "bertscore": bertscore_result,
        "vocab_awareness_pct": round(vocab_awareness_pct, 2),
        "in_window_count": in_window_count,
        "total_results": total_results,
        "structured_output_valid_rate": (
            round(structured_output_valid_rate, 2)
            if structured_output_valid_rate is not None
            else None
        ),
        "structured_successes": structured_successes,
        "structured_attempts": structured_attempts,
        "coverage_window": list(coverage_window),
        "all_coverage_ratios": all_coverage_ratios,
        "ks": ks,
        "top_broad": top_broad,
        "generation_top_k": args.generation_top_k,
        "semantic_weight": args.semantic_weight,
        "vocab_weight": args.vocab_weight,
        "used_cross_encoder": args.mode == "weighted_rerank" and not args.no_cross_encoder,
        "generated_pairs": generated_pairs,
        "query_details": query_details,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()