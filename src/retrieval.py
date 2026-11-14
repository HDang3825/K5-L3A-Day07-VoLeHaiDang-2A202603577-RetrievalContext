"""
Retrieval Module for Day 7 - Agentic RAG
Implements 3 retrieval methods:
1. Dense Retrieval (ChromaDB + EmbeddingGemma-2)
2. Lexical Retrieval (BM25 with Vietnamese tokenization)
3. Hybrid Retrieval (Reciprocal Rank Fusion - RRF of Dense + BM25)
"""

import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure current directory is in sys.path
if "." not in sys.path:
    sys.path.insert(0, ".")

try:
    from src.embeddings import EmbeddingModel
    from src.store import ChromaVectorStore
except Exception as e:
    print(f"Warning importing embeddings/store: {e}")
    EmbeddingModel = None
    ChromaVectorStore = None


def simple_vietnamese_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    text = text.lower()
    tokens = re.findall(r"[\w]+", text, flags=re.UNICODE)
    return tokens


class BM25Retriever:
    """In-memory BM25 lexical retriever for Vietnamese text chunks."""

    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.corpus_size = len(chunks)
        self.doc_lens = []
        self.doc_freqs: Dict[str, int] = {}
        self.doc_tokens: List[List[str]] = []
        
        self._build_index()

    def _build_index(self):
        total_len = 0
        for chunk in self.chunks:
            tokens = simple_vietnamese_tokenize(chunk.get("text", ""))
            self.doc_tokens.append(tokens)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len

            seen = set()
            for token in tokens:
                if token not in seen:
                    self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                    seen.add(token)

        self.avg_doc_len = total_len / max(1, self.corpus_size)

        # Precompute IDF (Robertson-Spärck Jones IDF with smoothing)
        self.idf: Dict[str, float] = {}
        for token, df in self.doc_freqs.items():
            self.idf[token] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5, filter_meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query_tokens = simple_vietnamese_tokenize(query)
        if not query_tokens:
            return []

        scores = [0.0] * self.corpus_size

        for i, tokens in enumerate(self.doc_tokens):
            chunk = self.chunks[i]

            # Apply filter if specified
            if filter_meta:
                match = True
                meta = chunk.get("metadata", {})
                for k, v in filter_meta.items():
                    if meta.get(k) != v:
                        match = False
                        break
                if not match:
                    scores[i] = -1.0
                    continue

            # Calculate BM25 score
            doc_len = self.doc_lens[i]
            token_counts: Dict[str, int] = {}
            for t in tokens:
                token_counts[t] = token_counts.get(t, 0) + 1

            doc_score = 0.0
            for q_token in query_tokens:
                if q_token not in self.idf:
                    continue
                tf = token_counts.get(q_token, 0)
                if tf > 0:
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1e-6, self.avg_doc_len)))
                    doc_score += self.idf[q_token] * (numerator / denominator)

            scores[i] = doc_score

        # Get top-k indices
        ranked_indices = sorted(range(self.corpus_size), key=lambda idx: scores[idx], reverse=True)
        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            if scores[idx] <= 0:
                break
            chunk = self.chunks[idx].copy()
            results.append({
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "text": chunk.get("text"),
                "metadata": chunk.get("metadata", {}),
                "score": float(scores[idx]),
                "rank": rank,
                "retriever": "bm25"
            })
        return results


class DenseRetriever:
    """Dense retriever using ChromaDB collection and EmbeddingGemma-2."""

    def __init__(self, chroma_path: str = "./data/chroma_db", collection_name: str = "strategy_b"):
        import chromadb
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(collection_name)
        self.embedding_model = EmbeddingModel()

    def search(self, query: str, top_k: int = 5, filter_meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query_vector = self.embedding_model.embed_query(query)

        where_clause = None
        if filter_meta:
            # Chroma where format
            if len(filter_meta) == 1:
                where_clause = filter_meta
            elif len(filter_meta) > 1:
                where_clause = {"$and": [{k: v} for k, v in filter_meta.items()]}

        res = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_clause
        )

        results = []
        if res and res["ids"] and len(res["ids"][0]) > 0:
            ids = res["ids"][0]
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            distances = res["distances"][0] if "distances" in res and res["distances"] else [0.0] * len(ids)

            for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances), start=1):
                # Chroma cosine distance = 1 - cosine_similarity
                score = 1.0 - dist if dist is not None else 1.0
                results.append({
                    "chunk_id": cid,
                    "document_id": meta.get("document_id", "VNM_AR_2025"),
                    "source": meta.get("source"),
                    "page": meta.get("page"),
                    "text": doc,
                    "metadata": meta,
                    "score": float(score),
                    "rank": rank,
                    "retriever": "dense"
                })
        return results


class HybridRetriever:
    """Hybrid Retriever combining Dense and BM25 using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever, rrf_k: int = 60):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5, filter_meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # Fetch candidate pools from both
        candidate_k = max(top_k * 3, 20)
        dense_results = self.dense.search(query, top_k=candidate_k, filter_meta=filter_meta)
        bm25_results = self.bm25.search(query, top_k=candidate_k, filter_meta=filter_meta)

        # Compute RRF score: score(d) = sum(1 / (k + rank_i(d)))
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            if cid not in chunk_map:
                chunk_map[cid] = item

        for rank, item in enumerate(bm25_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            if cid not in chunk_map:
                chunk_map[cid] = item

        # Sort by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        results = []
        for rank, cid in enumerate(sorted_cids[:top_k], start=1):
            item = chunk_map[cid].copy()
            item["score"] = rrf_scores[cid]
            item["rank"] = rank
            item["retriever"] = "hybrid_rrf"
            results.append(item)

        return results
