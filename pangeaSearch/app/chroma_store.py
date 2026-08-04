"""ChromaDB persistence for media summary + transcript chunks."""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_client = None
_collection = None


def get_collection(settings: Optional[Settings] = None):
    global _client, _collection
    settings = settings or get_settings()
    with _lock:
        if _collection is not None:
            return _collection
        path = Path(settings.chroma_path)
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Opening Chroma at %s", path)
        _client = chromadb.PersistentClient(path=str(path))
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        return _collection


def reset_collection_cache() -> None:
    global _client, _collection
    with _lock:
        _client = None
        _collection = None


def delete_media_chunks(media_id: int, settings: Optional[Settings] = None) -> int:
    """Remove existing documents for a media_id before re-index."""
    collection = get_collection(settings)
    existing = collection.get(where={"media_id": media_id})
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def upsert_media_documents(
    media_id: int,
    documents: List[Dict[str, Any]],
    embeddings: List[List[float]],
    title: str = "",
    settings: Optional[Settings] = None,
) -> int:
    """Store summary + chunks. Each document: text, kind ('summary'|'chunk'), chunk_index."""
    if len(documents) != len(embeddings):
        raise ValueError("documents and embeddings length mismatch")
    if not documents:
        return 0

    settings = settings or get_settings()
    delete_media_chunks(media_id, settings=settings)
    collection = get_collection(settings)

    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for doc in documents:
        kind = doc.get("kind") or "chunk"
        chunk_index = int(doc.get("chunk_index", 0))
        if kind == "summary":
            doc_id = "%s_summary" % media_id
        else:
            doc_id = "%s_chunk_%s" % (media_id, chunk_index)
        ids.append(doc_id)
        texts.append(doc["text"])
        metadatas.append(
            {
                "media_id": media_id,
                "kind": kind,
                "chunk_index": chunk_index,
                "title": title or "",
            }
        )

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(documents)


def upsert_chunks(
    media_id: int,
    chunks: List[str],
    embeddings: List[List[float]],
    title: str = "",
    settings: Optional[Settings] = None,
) -> int:
    """Backward-compatible helper: index chunks only (no summary)."""
    documents = [
        {"text": c, "kind": "chunk", "chunk_index": i} for i, c in enumerate(chunks)
    ]
    return upsert_media_documents(
        media_id=media_id,
        documents=documents,
        embeddings=embeddings,
        title=title,
        settings=settings,
    )


def query_similar(
    query_embedding: List[float],
    top_k: int = 5,
    settings: Optional[Settings] = None,
) -> List[Dict[str, Any]]:
    collection = get_collection(settings)
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )

    hits: List[Dict[str, Any]] = []
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]

    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else None
        score = None if dist is None else round(1.0 - float(dist), 4)
        hits.append(
            {
                "id": ids[i] if i < len(ids) else None,
                "media_id": meta.get("media_id"),
                "kind": meta.get("kind") or "chunk",
                "chunk_index": meta.get("chunk_index"),
                "title": meta.get("title") or "",
                "text": doc,
                "score": score,
                "distance": dist,
            }
        )
    return hits


def collection_stats(settings: Optional[Settings] = None) -> Dict[str, Any]:
    settings = settings or get_settings()
    collection = get_collection(settings)
    return {
        "collection": settings.chroma_collection,
        "path": settings.chroma_path,
        "count": collection.count(),
    }
