"""Lazy-loaded sentence-transformers embedder (EmbeddingGemma by default)."""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model = None
_model_name: Optional[str] = None


def _load_model(name: str):
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s (first call may download weights)", name)
    return SentenceTransformer(name)


def get_embedder(settings: Optional[Settings] = None):
    global _model, _model_name
    settings = settings or get_settings()
    name = settings.embedding_model
    with _lock:
        if _model is None or _model_name != name:
            _model = _load_model(name)
            _model_name = name
        return _model


def embed_texts(texts: List[str], settings: Optional[Settings] = None) -> List[List[float]]:
    if not texts:
        return []
    model = get_embedder(settings)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(text: str, settings: Optional[Settings] = None) -> List[float]:
    return embed_texts([text], settings=settings)[0]


def reset_embedder() -> None:
    """Test helper — drop cached model."""
    global _model, _model_name
    with _lock:
        _model = None
        _model_name = None
