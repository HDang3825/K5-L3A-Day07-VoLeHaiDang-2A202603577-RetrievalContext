"""Implementation of chunking strategies for Data Ingestion pipeline."""
import re
from typing import Any
from src.models import Chunk, DocumentPage


def fixed_size_chunking(
    pages: list[DocumentPage],
    chunk_size: int = 500,
    overlap: int = 50,
    strategy_name: str = "strategy_a"
) -> list[Chunk]:
    """
    Strategy A: Fixed-size character window with constant overlap.
    Retains page, document_id, source and section metadata.
    """
    chunks: list[Chunk] = []
    chunk_idx = 1

    for page in pages:
        text = page.text
        if not text:
            continue

        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"{strategy_name}_p{page.page}_{chunk_idx:04d}"
                meta = dict(page.metadata)
                meta["strategy"] = strategy_name
                meta["start_char"] = start
                meta["end_char"] = end

                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=chunk_text,
                    strategy=strategy_name,
                    char_count=len(chunk_text),
                    metadata=meta
                ))
                chunk_idx += 1

            if end >= text_len:
                break
            start += chunk_size - overlap

    return chunks


def _split_long_paragraph(text: str, max_size: int = 800) -> list[str]:
    """Helper to break oversized paragraphs into sentence-boundary slices."""
    if len(text) <= max_size:
        return [text]

    # Try splitting by sentence endings (. or ; or newline)
    sentences = re.split(r"(?<=[.!?;\n])\s+", text)
    sub_chunks = []
    current_sub = []
    current_len = 0

    for s in sentences:
        s_len = len(s)
        if current_len + s_len > max_size and current_sub:
            sub_chunks.append(" ".join(current_sub).strip())
            current_sub = [s]
            current_len = s_len
        else:
            current_sub.append(s)
            current_len += s_len + 1

    if current_sub:
        sub_chunks.append(" ".join(current_sub).strip())

    # Fallback if any single sentence is still larger than max_size
    final_chunks = []
    for sc in sub_chunks:
        if len(sc) > max_size:
            for i in range(0, len(sc), max_size - 100):
                final_chunks.append(sc[i:i + max_size].strip())
        else:
            final_chunks.append(sc)

    return [fc for fc in final_chunks if fc]


def paragraph_aware_chunking(
    pages: list[DocumentPage],
    target_max_size: int = 800,
    strategy_name: str = "strategy_b"
) -> list[Chunk]:
    """
    Strategy B: Structure / Paragraph-aware chunking.
    Respects paragraph boundaries (double newlines, bullet points, headers),
    merging small paragraphs up to target size and splitting oversized paragraphs cleanly.
    """
    chunks: list[Chunk] = []
    chunk_idx = 1

    for page in pages:
        text = page.text
        if not text:
            continue

        # Split into logical units (paragraphs or bullet points)
        raw_paras = [p.strip() for p in re.split(r"\n{2,}|(?=\n[•▪\-\d]+\.?\s)", text) if p.strip()]
        
        # Flatten by splitting any paragraph longer than target_max_size
        paras: list[str] = []
        for p in raw_paras:
            paras.extend(_split_long_paragraph(p, max_size=target_max_size))

        current_chunk_paras: list[str] = []
        current_len = 0

        for para in paras:
            para_len = len(para)

            if current_len + para_len > target_max_size and current_chunk_paras:
                chunk_text = "\n\n".join(current_chunk_paras).strip()
                if chunk_text:
                    chunk_id = f"{strategy_name}_p{page.page}_{chunk_idx:04d}"
                    meta = dict(page.metadata)
                    meta["strategy"] = strategy_name

                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        document_id=page.document_id,
                        source=page.source,
                        page=page.page,
                        text=chunk_text,
                        strategy=strategy_name,
                        char_count=len(chunk_text),
                        metadata=meta
                    ))
                    chunk_idx += 1
                current_chunk_paras = [para]
                current_len = para_len
            else:
                current_chunk_paras.append(para)
                current_len += para_len + 2

        if current_chunk_paras:
            chunk_text = "\n\n".join(current_chunk_paras).strip()
            if chunk_text:
                chunk_id = f"{strategy_name}_p{page.page}_{chunk_idx:04d}"
                meta = dict(page.metadata)
                meta["strategy"] = strategy_name

                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=chunk_text,
                    strategy=strategy_name,
                    char_count=len(chunk_text),
                    metadata=meta
                ))
                chunk_idx += 1

    return chunks
