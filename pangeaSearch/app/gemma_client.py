"""Shared Ollama/Gemma generate helper."""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def generate_with_gemma(
    prompt: str,
    *,
    temperature: float = 0.3,
    num_predict: int = 280,
    settings: Optional[Settings] = None,
) -> str:
    """Call Ollama /api/generate. Raises on failure or empty response."""
    settings = settings or get_settings()
    url = settings.gemma_api_url.rstrip("/") + "/api/generate"
    payload = {
        "model": settings.gemma_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    with httpx.Client(timeout=settings.gemma_timeout_seconds) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("response") or "").strip()
        if not text:
            raise RuntimeError("Gemma returned empty response")
        return text
