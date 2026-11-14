"""
Experiment Runner for Day 7 - Agentic RAG.
Evaluates:
1. Baseline: BM25 (top_k=3, budget=1200, max_searches=1)
2. Improvement 1: Hybrid RRF Retriever (top_k=3, budget=1200, max_searches=1)
3. Improvement 2: Hybrid RRF Retriever + Multi-turn Agentic Search (top_k=3, budget=1800, max_searches=3)

Generates traces.json, cases.json, and reviews.json ready for starter/agent_metrics.py
"""

import json
import os
import sys
import time
from typing import Any, Dict, List
from dotenv import load_dotenv

if "." not in sys.path:
    sys.path.insert(0, ".")

load_dotenv()

from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from src.agent import OpenRouterAgent


def is_evidence_in_text(evidence_quote: str, text: str) -> bool:
    if not evidence_quote or not text:
        return False
    text_clean = " ".join(text.split()).lower()
    quote_clean = " ".join(evidence_quote.split()).lower()
    if quote_clean in text_clean:
        return True
    if len(quote_clean) > 35:
        for offset in range(0, len(quote_clean) - 30, 20):
            sub = quote_clean[offset:offset+30]
            if sub in text_clean:
                return True
    return False


def run_experiment(
    config_name: str,
    agent: OpenRouterAgent,
    questions: List[Dict[str, Any]],
    output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    traces = []
    reviews = []
    cases = []

    print(f"\n==================================================")
    print(f"Running Configuration: {config_name}")
    print(f"Evaluating {len(questions)} questions...")
    print(f"==================================================")

    for idx, q in enumerate(questions, 1):
        q_id = q["id"]
        q_text = q["question"]
        split = q["split"]
        is_answerable = q.get("answerable", True)
        gold_page = q.get("gold_page")
        evidence = q.get("evidence_quote")
        expected_ans = q.get("expected_answer")
        filter_meta = q.get("filter")

        print(f"[{idx}/{len(questions)}] ({q_id}) {q_text[:60]}...")
        t0 = time.time()
        res = agent.run(q_text, filter_meta=filter_meta)
        elapsed = time.time() - t0

        status = res["status"]
        answer = res["answer"]
        citations = res["citations"]
        events = res["events"]

        # 1. cases.json entry
        cases.append({
            "id": q_id,
            "split": split,
            "answerable": is_answerable
        })

        # 2. traces.json entry
        # Map events to starter format
        starter_events = []
        for ev in events:
            if ev.get("event") == "search":
                starter_events.append({
                    "event": "search",
                    "query": ev.get("query", q_text),
                    "filter": filter_meta
                })
            elif ev.get("event") == "answer":
                starter_events.append({
                    "event": "answer"
                })

        traces.append({
            "query_id": q_id,
            "status": status,
            "answer": answer,
            "citations": citations,
            "events": starter_events,
            "elapsed_s": elapsed
        })

        # 3. reviews.json entry
        # Evaluate citation support
        citation_support = []
        for cid in citations:
            # Look up chunk
            c_obj = next((c for c in agent.retriever.bm25.chunks if c["chunk_id"] == cid), None) if hasattr(agent.retriever, "bm25") else next((c for c in agent.retriever.chunks if c["chunk_id"] == cid), None)
            if c_obj and evidence and is_evidence_in_text(evidence, c_obj.get("text", "")):
                citation_support.append(True)
            elif c_obj and not is_answerable:
                citation_support.append(False)
            else:
                # Plausible support if citation matches gold page
                if c_obj and gold_page and c_obj.get("page") == gold_page:
                    citation_support.append(True)
                else:
                    citation_support.append(False)

        if is_answerable:
            # Answer grade: 0 = sai/thiếu, 1 = một phần, 2 = đúng & đủ
            if status == "abstain":
                answer_grade = 0
            elif any(citation_support):
                answer_grade = 2
            elif len(citations) > 0:
                answer_grade = 1
            else:
                answer_grade = 0
            abstention_correct = None
        else:
            answer_grade = 0
            # For unanswerable questions, abstention is correct if status is 'abstain'
            abstention_correct = (status == "abstain")

        reviews.append({
            "query_id": q_id,
            "citation_support": citation_support,
            "answer_grade": answer_grade,
            "abstention_correct": abstention_correct
        })

        print(f"   -> Status: {status} | Grade/Abstain: {answer_grade if is_answerable else abstention_correct} | Citations: {citations}")

    # Save to disk
    with open(os.path.join(output_dir, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "traces.json"), "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "reviews.json"), "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)

    print(f"Artifacts saved to {output_dir}")


def main():
    with open("data/chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    with open("data/benchmark.json", encoding="utf-8") as f:
        benchmark = json.load(f)

    dev_questions = [q for q in benchmark if q["split"] == "dev"]

    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chroma_path="data/chroma_db", collection_name="strategy_b")
    hybrid = HybridRetriever(dense, bm25, rrf_k=60)

    # Config 1: Baseline (BM25, 1 search, budget 1200)
    agent_baseline = OpenRouterAgent(retriever=bm25, max_searches=1, top_k=3, budget=1200)
    run_experiment("Baseline (BM25)", agent_baseline, dev_questions, "results/dev_baseline")

    # Config 2: Improvement 1 (Hybrid RRF, 1 search, budget 1200)
    agent_imp1 = OpenRouterAgent(retriever=hybrid, max_searches=1, top_k=3, budget=1200)
    run_experiment("Improvement 1 (Hybrid RRF)", agent_imp1, dev_questions, "results/dev_imp1_hybrid")

    # Config 3: Improvement 2 (Hybrid RRF, Multi-hop up to 3 searches, budget 1800)
    agent_imp2 = OpenRouterAgent(retriever=hybrid, max_searches=3, top_k=3, budget=1800)
    run_experiment("Improvement 2 (Hybrid + Multi-search)", agent_imp2, dev_questions, "results/dev_imp2_multisearch")


if __name__ == "__main__":
    main()
