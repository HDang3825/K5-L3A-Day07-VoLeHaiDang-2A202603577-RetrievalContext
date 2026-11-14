"""Evaluation module for computing Hit@1, Evidence Hit@3 and saving traces."""
import json
import os
from typing import Any


def is_chunk_hit(retrieved_chunk: dict, gold_page: int, evidence_quote: str) -> bool:
    """
    Check if a retrieved chunk matches gold evidence:
    1. Must match gold_page
    2. Must contain key words / evidence_quote from source
    """
    chunk_meta = retrieved_chunk.get("metadata", {})
    chunk_page = chunk_meta.get("page")
    chunk_text = retrieved_chunk.get("text", "")

    if chunk_page != gold_page:
        return False

    # Check evidence substring (clean whitespace)
    norm_quote = " ".join(evidence_quote.split())
    norm_chunk = " ".join(chunk_text.split())

    if norm_quote in norm_chunk:
        return True

    # If quote was slightly longer than single chunk window, check high keyword coverage
    quote_words = [w.lower() for w in norm_quote.split() if len(w) > 3]
    if not quote_words:
        return True
    matched_words = sum(1 for w in quote_words if w in norm_chunk.lower())
    return (matched_words / len(quote_words)) >= 0.75


def evaluate_strategy_results(
    benchmark_questions: list[dict],
    strategy_name: str,
    retrieval_results: dict[str, list[dict]]
) -> tuple[dict[str, Any], list[dict]]:
    """
    Compute Hit@1 and Hit@3 metrics and record detailed traces.
    Returns:
        (metrics_summary, traces_list)
    """
    traces = []
    dev_hits_top1 = 0
    dev_hits_top3 = 0
    dev_total = 0

    test_hits_top1 = 0
    test_hits_top3 = 0
    test_total = 0

    for q in benchmark_questions:
        q_id = q["id"]
        split = q["split"]
        gold_page = q["gold_page"]
        evidence_quote = q["evidence_quote"]

        top_chunks = retrieval_results.get(q_id, [])

        top1_hit = False
        top3_hit = False

        if len(top_chunks) > 0:
            top1_hit = is_chunk_hit(top_chunks[0], gold_page, evidence_quote)
            top3_hit = any(is_chunk_hit(c, gold_page, evidence_quote) for c in top_chunks[:3])

        if split == "dev":
            dev_total += 1
            if top1_hit:
                dev_hits_top1 += 1
            if top3_hit:
                dev_hits_top3 += 1
        else:
            test_total += 1
            if top1_hit:
                test_hits_top1 += 1
            if top3_hit:
                test_hits_top3 += 1

        traces.append({
            "question_id": q_id,
            "split": split,
            "strategy": strategy_name,
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold_page": gold_page,
            "evidence_quote": evidence_quote,
            "applied_filter": q.get("filter"),
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
            "retrieved_top3": [
                {
                    "rank": idx + 1,
                    "chunk_id": c["chunk_id"],
                    "page": c["metadata"].get("page"),
                    "section": c["metadata"].get("section"),
                    "char_count": len(c["text"]),
                    "text_preview": c["text"][:150] + "...",
                    "is_hit": is_chunk_hit(c, gold_page, evidence_quote)
                }
                for idx, c in enumerate(top_chunks[:3])
            ]
        })

    metrics = {
        "strategy": strategy_name,
        "dev": {
            "total": dev_total,
            "hit@1_count": dev_hits_top1,
            "hit@1_rate": dev_hits_top1 / dev_total if dev_total > 0 else 0,
            "hit@3_count": dev_hits_top3,
            "hit@3_rate": dev_hits_top3 / dev_total if dev_total > 0 else 0
        },
        "test": {
            "total": test_total,
            "hit@1_count": test_hits_top1,
            "hit@1_rate": test_hits_top1 / test_total if test_total > 0 else 0,
            "hit@3_count": test_hits_top3,
            "hit@3_rate": test_hits_top3 / test_total if test_total > 0 else 0
        },
        "overall": {
            "total": dev_total + test_total,
            "hit@1_count": dev_hits_top1 + test_hits_top1,
            "hit@1_rate": (dev_hits_top1 + test_hits_top1) / (dev_total + test_total),
            "hit@3_count": dev_hits_top3 + test_hits_top3,
            "hit@3_rate": (dev_hits_top3 + test_hits_top3) / (dev_total + test_total)
        }
    }

    return metrics, traces
