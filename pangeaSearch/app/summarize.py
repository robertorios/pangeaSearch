"""Generate a short media summary (Gemma via Ollama when available)."""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

from app.config import Settings, get_settings
from app.gemma_client import generate_with_gemma

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = (
    "Summarize the following testimony transcript in 2 to 4 clear sentences. "
    "Focus on the person's situation, feelings, and what helped. "
    "Do not invent facts. Output only the summary.\n\nTranscript:\n%s"
)


def _fallback_summary(transcript: str, max_chars: int = 500) -> str:
    """Used when Gemma/Ollama is unavailable — first sentences / truncated text."""
    cleaned = " ".join((transcript or "").split())
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    summary = " ".join(parts[:3]).strip() if parts else cleaned
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return summary


def summarize_transcript(
    transcript: str,
    settings: Optional[Settings] = None,
) -> Tuple[str, str]:
    """Return (summary_text, source) where source is 'gemma' or 'fallback'."""
    settings = settings or get_settings()
    text = (transcript or "").strip()
    if not text:
        raise ValueError("Empty transcript; cannot summarize")

    if not settings.summarize_enabled:
        return _fallback_summary(text), "fallback_disabled"

    try:
        summary = generate_with_gemma(
            _SUMMARY_PROMPT % text[:12000],
            temperature=0.2,
            num_predict=180,
            settings=settings,
        )
        logger.info("Summary generated via Gemma model=%s", settings.gemma_model)
        return summary, "gemma"
    except Exception as exc:
        logger.warning(
            "Gemma summarize failed (%s); using fallback summary",
            exc,
        )
        return _fallback_summary(text), "fallback"
