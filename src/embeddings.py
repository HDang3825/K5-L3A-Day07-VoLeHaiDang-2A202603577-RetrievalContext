"""Embedding module using google/embeddinggemma-2 via SentenceTransformer."""
import os
from typing import Optional
import torch

# Prevent SentenceTransformer / HuggingFace from attempting network requests when cached locally
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str = "google/embeddinggemma-2", device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        print(f"Loading EmbeddingGemma 2 model '{model_name}' on {self.device.upper()}...")
        try:
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                model_kwargs={"torch_dtype": self.dtype}
            )
        except Exception as e:
            # Fallback to CPU if CUDA OOM occurs during model loading
            if self.device == "cuda":
                print(f"CUDA loading failed ({e}). Falling back to CPU...")
                self.device = "cpu"
                self.dtype = torch.float32
                self.model = SentenceTransformer(
                    model_name,
                    device="cpu",
                    model_kwargs={"torch_dtype": torch.float32}
                )
            else:
                raise e

        print(f"Embedding model loaded successfully on {self.device.upper()}!")

    def embed_texts(self, texts: list[str], batch_size: int = 16, show_progress_bar: bool = True) -> list[list[float]]:
        """Encode a list of text chunks into embedding vectors."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Encode a single search query."""
        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist()
