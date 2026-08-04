"""Shared indexing: summary + transcript chunks → embed → Chroma."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.chroma_store import upsert_media_documents
from app.chunking import chunk_text
from app.config import Settings, get_settings
from app.embeddings import embed_texts
from app.summarize import summarize_transcript


def index_transcript(
    media_id: int,
    transcript: str,
    title: str = "",
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Summarize, chunk, embed summary + chunks, upsert under media_id."""
    settings = settings or get_settings()
    title = title or ""
    transcript = (transcript or "").strip()
    if not transcript:
        raise ValueError("Empty transcript")

    summary, summary_source = summarize_transcript(transcript, settings=settings)
    chunks = chunk_text(
        transcript,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise RuntimeError("No transcript chunks to index")

    documents: List[Dict[str, Any]] = [
        {
            "text": summary,
            "kind": "summary",
            "chunk_index": -1,
        }
    ]
    for i, chunk in enumerate(chunks):
        documents.append(
            {
                "text": chunk,
                "kind": "chunk",
                "chunk_index": i,
            }
        )

    texts = [d["text"] for d in documents]
    vectors = embed_texts(texts, settings=settings)
    count = upsert_media_documents(
        media_id=media_id,
        documents=documents,
        embeddings=vectors,
        title=title,
        settings=settings,
    )

    return {
        "media_id": media_id,
        "title": title,
        "summary": summary,
        "summary_source": summary_source,
        "summary_indexed": 1,
        "chunks_indexed": len(chunks),
        "documents_indexed": count,
        "embedding_model": settings.embedding_model,
    }
