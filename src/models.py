"""Shared data models for ingestion pipeline."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentPage:
    document_id: str
    source: str
    page: int
    text: str
    char_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    source: str
    page: int
    text: str
    strategy: str
    char_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
