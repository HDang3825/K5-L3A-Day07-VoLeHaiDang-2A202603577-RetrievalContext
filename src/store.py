"""ChromaDB Vector Store wrapper for indexing and retrieval."""
import os
from typing import Optional, Any
import chromadb
from chromadb.config import Settings
from src.models import Chunk


class ChromaVectorStore:
    def __init__(self, persist_directory: str = "data/chroma_db"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

    def get_or_create_collection(self, name: str):
        """Create or get a collection with cosine similarity metric."""
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )

    def index_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        batch_size: int = 100
    ):
        """Insert or update chunks into the specified collection in batches."""
        collection = self.get_or_create_collection(collection_name)
        
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            ids = [c.chunk_id for c in batch_chunks]
            documents = [c.text for c in batch_chunks]
            metadatas = [
                {
                    "document_id": str(c.document_id),
                    "source": str(c.source),
                    "page": int(c.page),
                    "section": str(c.metadata.get("section", "")),
                    "year": int(c.metadata.get("year", 2025)),
                    "strategy": str(c.strategy),
                    "char_count": int(c.char_count)
                }
                for c in batch_chunks
            ]

            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=batch_embeddings,
                metadatas=metadatas
            )

    def query(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 3,
        metadata_filter: Optional[dict[str, Any]] = None
    ) -> list[dict]:
        """
        Query top-k nearest chunks.
        Optionally applies metadata_filter (e.g. {"section": "quan_tri_cong_ty"}).
        """
        collection = self.get_or_create_collection(collection_name)
        
        where_clause = metadata_filter if metadata_filter else None
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause
        )

        retrieved: list[dict] = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for idx in range(len(results["ids"][0])):
                retrieved.append({
                    "chunk_id": results["ids"][0][idx],
                    "text": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "distance": results["distances"][0][idx] if results.get("distances") else None
                })

        return retrieved
