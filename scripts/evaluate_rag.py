"""
RAG Evaluation Script
======================
Computes:
  - Precision@k, Recall@k, MRR for k in {5, 10, 20}
  - ROUGE-1/2/L between generated summaries and article text
  - BERTScore (optional, requires bert-score package)
  - Vocabulary-awareness %: results falling within coverage window

Saves full results to data/eval_results.json.

Usage
-----
    python scripts/evaluate_rag.py                         # template generator, no cross-encoder
    python scripts/evaluate_rag.py --generator llama       # llama.cpp GPU
        --model-path models/llama-3.1-8b-instruct.Q4_K_M.gguf
    python scripts/evaluate_rag.py --with-bertscore        # slow but complete
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.embedder import ArticleEmbedder
from src.embeddings.vector_store import FAISSVectorStore
from src.rag.pipeline import LlamaCppGenerator, RAGPipeline, TemplateGenerator
from src.rag.reranker import VocabAwareReranker
from src.rag.retriever import TwoStageRetriever


# ── IR metric helpers ─────────────────────────────────────────────────────────

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of top-k retrieved that are relevant."""
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
    """1 / rank of the first relevant result (0 if none found)."""
    for rank, rid in enumerate(retrieved, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def mean_metrics(per_query: List[Dict]) -> Dict:
    """Average metric dicts across queries."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: round(sum(q[k] for q in per_query) / len(per_query), 4) for k in keys}


# ── ROUGE helper ──────────────────────────────────────────────────────────────

def compute_rouge(hypothesis: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
    except ImportError:
        print("[evaluate_rag] rouge-score not installed. Run: pip install rouge-score")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def article_reference(chunk_text: str, max_words: int = 100) -> str:
    """Use first max_words of the article as the evaluation reference."""
    return " ".join(chunk_text.split()[:max_words])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the RAG pipeline on held-out queries.")
    p.add_argument("--index-dir",       default="data/faiss_index")
    p.add_argument("--queries",         default="data/eval_queries.json")
    p.add_argument("--output",          default="data/eval_results.json")
    p.add_argument("--device",          default="cuda",
                   help="Torch device: 'cuda' (default) or 'cpu'")
    p.add_argument("--generator",       choices=["template", "llama"], default="template")
    p.add_argument("--model-path",      default="models/llama-3.1-8b-instruct.Q4_K_M.gguf",
                   help="Path to GGUF model file (used when --generator llama)")
    p.add_argument("--n-gpu-layers",    type=int, default=-1)
    p.add_argument("--no-cross-encoder", action="store_true",
                   help="Disable cross-encoder reranker (uses bi-encoder scores only; faster)")
    p.add_argument("--no-gpu-index",    action="store_true")
    p.add_argument("--with-bertscore",  action="store_true",
                   help="Also compute BERTScore (slow, requires bert-score package)")
    p.add_argument("--coverage-low",    type=float, default=0.85)
    p.add_argument("--coverage-high",   type=float, default=0.97)
    args = p.parse_args()

    coverage_window = (args.coverage_low, args.coverage_high)
    ks = [5, 10, 20]

    # -- Load system ----------------------------------------------------------
    use_gpu_index = not args.no_gpu_index
    store     = FAISSVectorStore.load(args.index_dir, use_gpu=use_gpu_index)
    embedder  = ArticleEmbedder(device=args.device)
    reranker  = VocabAwareReranker(
        use_cross_encoder=not args.no_cross_encoder,
        device=args.device,
    )
    # Use all chunks for broad retrieval (no cap larger than index)
    top_broad = len(store.chunks)
    retriever = TwoStageRetriever(store, embedder, reranker, top_broad=top_broad, top_k=top_broad)

    if args.generator == "llama":
        generator = LlamaCppGenerator(
            model_path=args.model_path,
            n_gpu_layers=args.n_gpu_layers,
        )
    else:
        generator = TemplateGenerator()

    pipeline = RAGPipeline(retriever, generator, coverage_window=coverage_window)

    # -- Load ground truth ----------------------------------------------------
    eval_queries = json.loads(Path(args.queries).read_text())
    print(f"Loaded {len(eval_queries)} evaluation queries.")
    print(f"Index has {len(store.chunks)} chunks across "
          f"{len(set(c.article_id for c in store.chunks))} articles.\n")

    effective_ks = [min(k, len(store.chunks)) for k in ks]

    per_query_ir:    List[Dict] = []
    per_query_rouge: List[Dict] = []
    all_coverage_ratios: List[float] = []
    in_window_count = 0
    total_results   = 0
    query_details   = []

    # -- Evaluation loop ------------------------------------------------------
    for item in eval_queries:
        query    = item["query"]
        relevant = set(item["relevant_ids"])
        level    = item.get("level", "intermediate")

        # Retrieve and rerank (get all chunks ranked)
        q_emb = embedder.embed_query(query)
        broad = store.search(q_emb, top_k=len(store.chunks))
        reranked = reranker.rerank(
            query=query,
            candidates=broad,
            vocab_level=level,
            coverage_window=coverage_window,
        )

        # Retrieved article IDs in rank order (de-duplicated to one per article)
        seen_ids: List[str] = []
        seen_set: Set[str] = set()
        for chunk, score in reranked:
            if chunk.article_id not in seen_set:
                seen_ids.append(chunk.article_id)
                seen_set.add(chunk.article_id)

        # IR metrics
        ir_row: Dict = {}
        for k, ek in zip(ks, effective_ks):
            ir_row[f"precision@{k}"] = precision_at_k(seen_ids, relevant, ek)
            ir_row[f"recall@{k}"]    = recall_at_k(seen_ids, relevant, ek)
        ir_row["mrr"] = reciprocal_rank(seen_ids, relevant)
        per_query_ir.append(ir_row)

        # Vocabulary-awareness: collect coverage ratios of top-10 results
        top_chunks = [chunk for chunk, score in reranked[:10]]
        for chunk in top_chunks:
            ratio = chunk.coverage_ratio.get(level, 0.0)
            all_coverage_ratios.append(ratio)
            total_results += 1
            if coverage_window[0] <= ratio <= coverage_window[1]:
                in_window_count += 1

        # Generation + ROUGE (top-3 results only to keep eval fast)
        top3_results = pipeline.recommend(query, level, top_k=3)
        rouge_rows = []
        for result in top3_results:
            ref_chunk = next(
                (c for c in store.chunks if c.chunk_id in result.source_chunk_ids), None
            )
            if ref_chunk:
                reference = article_reference(ref_chunk.text)
                rouge_row = compute_rouge(result.summary, reference)
                rouge_rows.append(rouge_row)
                per_query_rouge.append(rouge_row)

        query_details.append({
            "query":      query,
            "level":      level,
            "relevant":   list(relevant),
            "retrieved":  seen_ids,
            "ir_metrics": ir_row,
            "rouge":      rouge_rows,
            "coverage_ratios_top10": [
                c.coverage_ratio.get(level, 0.0) for c in top_chunks
            ],
        })

        print(f"  Query: '{query[:50]}...' | "
              f"P@5={ir_row['precision@5']:.2f} R@5={ir_row['recall@5']:.2f} "
              f"MRR={ir_row['mrr']:.2f}")

    # -- Aggregate metrics ----------------------------------------------------
    mean_ir    = mean_metrics(per_query_ir)
    mean_rouge = mean_metrics(per_query_rouge) if per_query_rouge else {}
    vocab_awareness_pct = (in_window_count / total_results * 100) if total_results else 0.0

    print("\n" + "=" * 60)
    print("RETRIEVAL METRICS (mean across queries)")
    print("=" * 60)
    print(f"  {'Metric':<20} {'k=5':>8} {'k=10':>8} {'k=20':>8}")
    print(f"  {'-'*44}")
    for metric in ["precision", "recall"]:
        row = "  " + f"{metric:<20}"
        for k in ks:
            row += f"  {mean_ir.get(f'{metric}@{k}', 0.0):>6.4f}"
        print(row)
    print(f"  {'MRR':<20}  {mean_ir.get('mrr', 0.0):>6.4f}")

    print("\nGENERATION QUALITY (mean ROUGE vs article reference text)")
    print("=" * 60)
    for m in ["rouge1", "rouge2", "rougeL"]:
        print(f"  {m:<10}: {mean_rouge.get(m, 0.0):.4f}")

    print(f"\nVOCABULARY AWARENESS")
    print("=" * 60)
    print(f"  Results within coverage window [{coverage_window[0]:.0%}, {coverage_window[1]:.0%}]: "
          f"{in_window_count}/{total_results} = {vocab_awareness_pct:.1f}%")

    # BERTScore (optional — slow)
    bertscore_result = {}
    if args.with_bertscore:
        try:
            from bert_score import score as bs_score
            hypotheses = [
                d["summary"]
                for qd in query_details
                for d in qd.get("rouge", [])
                if isinstance(d, dict) and "summary" in d
            ]
            refs = []
            for qd in query_details:
                for result in pipeline.recommend(qd["query"], qd["level"], top_k=3):
                    c = next(
                        (ch for ch in store.chunks if ch.chunk_id in result.source_chunk_ids),
                        None,
                    )
                    if c:
                        refs.append(article_reference(c.text))
            if hypotheses and len(hypotheses) == len(refs):
                P, R, F = bs_score(hypotheses, refs, lang="en", verbose=False)
                bertscore_result = {
                    "precision": round(P.mean().item(), 4),
                    "recall":    round(R.mean().item(), 4),
                    "f1":        round(F.mean().item(), 4),
                }
                print(f"\nBERTScore: P={bertscore_result['precision']:.4f} "
                      f"R={bertscore_result['recall']:.4f} "
                      f"F1={bertscore_result['f1']:.4f}")
        except ImportError:
            print("\n[evaluate_rag] bert-score not installed. Run: pip install bert-score")

    # -- Save results ---------------------------------------------------------
    output = {
        "mean_ir_metrics":     mean_ir,
        "mean_rouge":          mean_rouge,
        "bertscore":           bertscore_result,
        "vocab_awareness_pct": round(vocab_awareness_pct, 2),
        "in_window_count":     in_window_count,
        "total_results":       total_results,
        "coverage_window":     list(coverage_window),
        "all_coverage_ratios": all_coverage_ratios,
        "ks":                  ks,
        "query_details":       query_details,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
