"""
Retrieval Benchmark Evaluation Script for Day 7.
Evaluates BM25, Dense, and Hybrid (RRF) on Dev & Test splits.
Computes Hit@1, Evidence Hit@3, and Context Evidence Rate.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

if "." not in sys.path:
    sys.path.insert(0, ".")

from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


def is_evidence_in_chunk(evidence_quote: str, page: int, chunk: Dict[str, Any]) -> bool:
    """Check if the chunk matches gold page and contains the key evidence string/quote."""
    if not evidence_quote:
        return False
    # Check page match if available
    chunk_page = chunk.get("page")
    if page is not None and chunk_page is not None and chunk_page != page:
        return False
    
    # Check text match (clean whitespace)
    chunk_text = " ".join(chunk.get("text", "").split()).lower()
    quote = " ".join(evidence_quote.split()).lower()
    
    # Check full or partial quote match (at least 25 chars or full quote)
    if quote in chunk_text:
        return True
    
    # If quote is long, check if a 40-character substring matches
    if len(quote) > 40:
        for offset in range(0, len(quote) - 35, 20):
            sub = quote[offset:offset+35]
            if sub in chunk_text:
                return True

    return False


def evaluate_retriever(
    retriever_name: str,
    retriever,
    questions: List[Dict[str, Any]],
    top_k: int = 3,
    context_budget: int = 1200
) -> Dict[str, Any]:
    hit_at_1 = 0
    hit_at_3 = 0
    context_evidence_hits = 0
    total_answerable = 0
    total_time = 0.0

    traces = []

    for q in questions:
        if not q.get("answerable", True):
            continue
        
        total_answerable += 1
        q_id = q["id"]
        q_text = q["question"]
        gold_page = q.get("gold_page")
        evidence = q.get("evidence_quote")
        filter_meta = q.get("filter")

        t0 = time.time()
        results = retriever.search(q_text, top_k=top_k, filter_meta=filter_meta)
        elapsed = time.time() - t0
        total_time += elapsed

        # Evaluate Hit@1 and Hit@3
        h1 = False
        h3 = False
        for rank, res in enumerate(results[:top_k], start=1):
            if is_evidence_in_chunk(evidence, gold_page, res):
                if rank == 1:
                    h1 = True
                h3 = True
                break

        if h1:
            hit_at_1 += 1
        if h3:
            hit_at_3 += 1

        # Simulate context selection within budget
        context_chunks = []
        current_len = 0
        context_has_evidence = False

        for res in results[:top_k]:
            chunk_len = len(res.get("text", "")) + len(str(res.get("source", ""))) + 50
            if current_len + chunk_len <= context_budget:
                context_chunks.append(res)
                current_len += chunk_len
                if is_evidence_in_chunk(evidence, gold_page, res):
                    context_has_evidence = True

        if context_has_evidence:
            context_evidence_hits += 1

        traces.append({
            "id": q_id,
            "hit_at_1": h1,
            "hit_at_3": h3,
            "context_hit": context_has_evidence,
            "results": [{"chunk_id": r["chunk_id"], "page": r["page"], "score": r["score"]} for r in results]
        })

    rate_h1 = (hit_at_1 / total_answerable) if total_answerable > 0 else 0.0
    rate_h3 = (hit_at_3 / total_answerable) if total_answerable > 0 else 0.0
    rate_ctx = (context_evidence_hits / total_answerable) if total_answerable > 0 else 0.0

    return {
        "retriever": retriever_name,
        "total_answerable": total_answerable,
        "hit_at_1": f"{hit_at_1}/{total_answerable} ({rate_h1:.1%})",
        "hit_at_3": f"{hit_at_3}/{total_answerable} ({rate_h3:.1%})",
        "context_evidence_rate": f"{context_evidence_hits}/{total_answerable} ({rate_ctx:.1%})",
        "avg_latency_ms": f"{(total_time / max(1, total_answerable)) * 1000:.1f}ms",
        "total_time_s": f"{total_time:.2f}s",
        "traces": traces
    }


def main():
    with open("data/chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    with open("data/benchmark.json", encoding="utf-8") as f:
        benchmark = json.load(f)

    dev_questions = [q for q in benchmark if q["split"] == "dev"]

    print("Initializing Retrievers...")
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chroma_path="data/chroma_db", collection_name="strategy_b")
    hybrid = HybridRetriever(dense, bm25, rrf_k=60)

    print("\n--- Evaluating on DEV (15 answerable questions) ---")
    res_bm25 = evaluate_retriever("BM25 (Lexical)", bm25, dev_questions, top_k=3)
    res_dense = evaluate_retriever("Dense (EmbeddingGemma-2)", dense, dev_questions, top_k=3)
    res_hybrid = evaluate_retriever("Hybrid (RRF)", hybrid, dev_questions, top_k=3)

    summary = [res_bm25, res_dense, res_hybrid]
    
    os.makedirs("results", exist_ok=True)
    with open("results/retrieval_dev_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "="*80)
    print(f"{'Method':<25} | {'Hit@1':<16} | {'Evidence Hit@3':<16} | {'Context Rate':<16} | {'Avg Latency'}")
    print("="*80)
    for r in summary:
        print(f"{r['retriever']:<25} | {r['hit_at_1']:<16} | {r['hit_at_3']:<16} | {r['context_evidence_rate']:<16} | {r['avg_latency_ms']}")
    print("="*80)


if __name__ == "__main__":
    main()
