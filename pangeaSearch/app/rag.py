"""RAG: turn Chroma hits + member situation into a human answer via Gemma."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.config import Settings, get_settings
from app.gemma_client import generate_with_gemma

logger = logging.getLogger(__name__)

_ANSWER_PROMPT = (
    "You help people find testimonies that match their situation.\n"
    "Write a warm, clear answer in 3 to 6 sentences.\n"
    "Use only the testimony excerpts below. Do not invent facts or quotes.\n"
    "If excerpts are weak or unrelated, say so gently and still point to what might help.\n"
    "Do not mention embeddings, databases, or that you are an AI.\n\n"
    "Member situation:\n%s\n\n"
    "Testimony excerpts:\n%s\n\n"
    "Answer:"
)


def _format_excerpts(hits: List[Dict[str, Any]], max_chars: int = 3500) -> str:
    parts: List[str] = []
    used = 0
    for i, hit in enumerate(hits, start=1):
        title = (hit.get("title") or "Untitled").strip()
        kind = hit.get("kind") or "chunk"
        media_id = hit.get("media_id")
        text = " ".join((hit.get("text") or "").split())
        if not text:
            continue
        block = "[%s] media_id=%s title=%r kind=%s\n%s" % (
            i,
            media_id,
            title,
            kind,
            text[:900],
        )
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts) if parts else "(no excerpts)"


def _fallback_answer(query: str, hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return (
            "I could not find matching testimonies for that situation yet. "
            "Try a few more details about what you are going through, or browse the gallery."
        )
    titles = []
    seen = set()
    for hit in hits:
        mid = hit.get("media_id")
        title = (hit.get("title") or "").strip() or ("Media #%s" % mid)
        key = mid if mid is not None else title
        if key in seen:
            continue
        seen.add(key)
        titles.append(title)
        if len(titles) >= 3:
            break
    listed = "; ".join(titles)
    return (
        "Based on testimonies in our library, these stories may relate to your situation "
        "(%s): %s. Open one that feels closest and watch or start a conversation from there."
        % (query.strip()[:120], listed)
    )


def answer_from_hits(
    query: str,
    hits: List[Dict[str, Any]],
    settings: Optional[Settings] = None,
) -> Tuple[str, str]:
    """Return (answer_text, source) where source is gemma | fallback | fallback_disabled | none."""
    settings = settings or get_settings()
    q = (query or "").strip()
    if not q:
        return "Please describe your situation in a sentence or two.", "none"

    if not settings.rag_enabled:
        return _fallback_answer(q, hits), "fallback_disabled"

    excerpts = _format_excerpts(hits)
    prompt = _ANSWER_PROMPT % (q, excerpts)

    try:
        answer = generate_with_gemma(
            prompt,
            temperature=0.35,
            num_predict=320,
            settings=settings,
        )
        logger.info("RAG answer via Gemma model=%s", settings.gemma_model)
        return answer, "gemma"
    except Exception as exc:
        logger.warning("Gemma RAG failed (%s); using fallback answer", exc)
        return _fallback_answer(q, hits), "fallback"


def suggest_media(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dedupe hits by media_id, keeping best score and title."""
    best: Dict[Any, Dict[str, Any]] = {}
    for hit in hits:
        mid = hit.get("media_id")
        if mid is None:
            continue
        score = hit.get("score")
        existing = best.get(mid)
        if existing is None or (score is not None and (existing.get("best_score") or -1) < score):
            best[mid] = {
                "media_id": mid,
                "title": hit.get("title") or "",
                "best_score": score,
            }
    return sorted(
        best.values(),
        key=lambda m: m.get("best_score") if m.get("best_score") is not None else -1,
        reverse=True,
    )
