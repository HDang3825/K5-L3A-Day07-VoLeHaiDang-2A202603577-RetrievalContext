"""End-to-end pipeline runner for indexing and benchmarking Strategy A, B, and C."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from src.ingestion import load_pdf
from src.chunking import fixed_size_chunking, paragraph_aware_chunking
from src.embeddings import EmbeddingModel
from src.store import ChromaVectorStore
from src.evaluation import evaluate_strategy_results

PDF_PATH = "data/raw/20260319_VNM_Bao_cao_thuong_nien_2025_eae619d5f6.pdf"
BENCHMARK_PATH = "benchmark/benchmark.json"
RESULTS_DIR = "results"


def run_pipeline():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("STEP 1: Ingestion & Text Normalization")
    pages, excluded = load_pdf(PDF_PATH)
    print(f"Loaded {len(pages)} useful pages, excluded {len(excluded)} pages.")

    print("STEP 2: Generating Chunks for Strategy A & B")
    chunks_a = fixed_size_chunking(pages, chunk_size=500, overlap=50, strategy_name="strategy_a")
    chunks_b = paragraph_aware_chunking(pages, target_max_size=800, strategy_name="strategy_b")
    print(f"Strategy A generated: {len(chunks_a)} chunks")
    print(f"Strategy B & C generated: {len(chunks_b)} chunks")

    print("STEP 3: Loading Embedding Model (google/embeddinggemma-2)")
    embedding_model = EmbeddingModel(model_name="google/embeddinggemma-2")

    print("STEP 4: Computing Embeddings & Indexing into ChromaDB")
    vector_store = ChromaVectorStore(persist_directory="data/chroma_db")

    # Strategy A
    start_time_a = time.time()
    texts_a = [c.text for c in chunks_a]
    embeddings_a = embedding_model.embed_texts(texts_a, batch_size=16)
    vector_store.index_chunks("strategy_a", chunks_a, embeddings_a)
    time_a = time.time() - start_time_a
    print(f"Strategy A indexed successfully in {time_a:.2f}s!")

    # Strategy B (also used for Strategy C)
    start_time_b = time.time()
    texts_b = [c.text for c in chunks_b]
    embeddings_b = embedding_model.embed_texts(texts_b, batch_size=16)
    vector_store.index_chunks("strategy_b", chunks_b, embeddings_b)
    time_b = time.time() - start_time_b
    print(f"Strategy B indexed successfully in {time_b:.2f}s!")

    print("STEP 5: Running Benchmark Retrieval (A, B, C)")
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark_questions = json.load(f)

    # Pre-embed queries
    query_embeddings = {}
    for q in benchmark_questions:
        query_embeddings[q["id"]] = embedding_model.embed_query(q["question"])

    # Strategy A Retrieval (No filter)
    results_a = {}
    for q in benchmark_questions:
        q_id = q["id"]
        retrieved = vector_store.query("strategy_a", query_embeddings[q_id], top_k=3, metadata_filter=None)
        results_a[q_id] = retrieved

    # Strategy B Retrieval (No filter)
    results_b = {}
    for q in benchmark_questions:
        q_id = q["id"]
        retrieved = vector_store.query("strategy_b", query_embeddings[q_id], top_k=3, metadata_filter=None)
        results_b[q_id] = retrieved

    # Strategy C Retrieval (Metadata Filter applied when available)
    results_c = {}
    for q in benchmark_questions:
        q_id = q["id"]
        meta_filter = q.get("filter")
        retrieved = vector_store.query("strategy_b", query_embeddings[q_id], top_k=3, metadata_filter=meta_filter)
        results_c[q_id] = retrieved

    print("STEP 6: Evaluating Metrics & Saving Traces")
    metrics_a, traces_a = evaluate_strategy_results(benchmark_questions, "Strategy A (Fixed-500)", results_a)
    metrics_b, traces_b = evaluate_strategy_results(benchmark_questions, "Strategy B (Paragraph-800)", results_b)
    metrics_c, traces_c = evaluate_strategy_results(benchmark_questions, "Strategy C (Metadata-Filtered)", results_c)

    all_metrics = {
        "strategy_a": metrics_a,
        "strategy_b": metrics_b,
        "strategy_c": metrics_c,
        "timing": {
            "index_time_seconds_a": round(time_a, 2),
            "index_time_seconds_b": round(time_b, 2)
        }
    }

    with open(os.path.join(RESULTS_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    with open(os.path.join(RESULTS_DIR, "traces_strategy_a.json"), "w", encoding="utf-8") as f:
        json.dump(traces_a, f, ensure_ascii=False, indent=2)

    with open(os.path.join(RESULTS_DIR, "traces_strategy_b.json"), "w", encoding="utf-8") as f:
        json.dump(traces_b, f, ensure_ascii=False, indent=2)

    with open(os.path.join(RESULTS_DIR, "traces_strategy_c.json"), "w", encoding="utf-8") as f:
        json.dump(traces_c, f, ensure_ascii=False, indent=2)

    print("\nSummary of Benchmark Results:")
    print(f"{'Strategy':<30} | {'Dev Hit@1':<12} | {'Dev Hit@3':<12} | {'Test Hit@1':<12} | {'Test Hit@3':<12} | {'All Hit@3':<12}")
    for name, m in [("Strategy A (Fixed)", metrics_a), ("Strategy B (Paragraph)", metrics_b), ("Strategy C (Filtered)", metrics_c)]:
        dev_h1 = f"{m['dev']['hit@1_count']}/{m['dev']['total']} ({m['dev']['hit@1_rate']*100:.1f}%)"
        dev_h3 = f"{m['dev']['hit@3_count']}/{m['dev']['total']} ({m['dev']['hit@3_rate']*100:.1f}%)"
        test_h1 = f"{m['test']['hit@1_count']}/{m['test']['total']} ({m['test']['hit@1_rate']*100:.1f}%)"
        test_h3 = f"{m['test']['hit@3_count']}/{m['test']['total']} ({m['test']['hit@3_rate']*100:.1f}%)"
        all_h3 = f"{m['overall']['hit@3_count']}/{m['overall']['total']} ({m['overall']['hit@3_rate']*100:.1f}%)"
        print(f"{name:<30} | {dev_h1:<12} | {dev_h3:<12} | {test_h1:<12} | {test_h3:<12} | {all_h3:<12}")
    print(f"All traces and metrics successfully exported to {RESULTS_DIR}/")


if __name__ == "__main__":
    run_pipeline()
