"""RQ job: obtain transcript → summarize → chunk → embed summary+chunks → Chroma."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

from app.config import get_settings
from app.index_pipeline import index_transcript
from app.transcribe import transcribe_file

logger = logging.getLogger(__name__)


def _download_media(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    name = Path(parsed.path).name or "media.bin"
    dest = dest_dir / name
    logger.info("Downloading media from %s", url)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes():
                fh.write(chunk)
    return dest


def process_media(
    media_id: int,
    title: str = "",
    source_url: Optional[str] = None,
    local_path: Optional[str] = None,
    transcript_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Background job: transcript → Gemma summary → embed summary + chunks.

    Provide one of:
    - transcript_text: skip Whisper (pipeline test)
    - local_path: Whisper on a local file
    - source_url: download then Whisper
    """
    settings = get_settings()
    title = title or ""

    if transcript_text and transcript_text.strip():
        text = transcript_text.strip()
        source = "provided_text"
    elif local_path:
        text = transcribe_file(local_path, settings=settings)
        source = "local_path"
    elif source_url:
        download_dir = Path(settings.media_download_dir) / str(media_id)
        if download_dir.exists():
            shutil.rmtree(download_dir, ignore_errors=True)
        path = _download_media(source_url, download_dir)
        try:
            text = transcribe_file(str(path), settings=settings)
        finally:
            shutil.rmtree(download_dir, ignore_errors=True)
        source = "source_url"
    else:
        raise ValueError(
            "Provide transcript_text, local_path, or source_url to process media"
        )

    indexed = index_transcript(
        media_id=media_id,
        transcript=text,
        title=title,
        settings=settings,
    )

    result = {
        "media_id": media_id,
        "title": title,
        "source": source,
        "transcript_chars": len(text),
        "summary": indexed.get("summary"),
        "summary_source": indexed.get("summary_source"),
        "summary_indexed": indexed.get("summary_indexed"),
        "chunks_indexed": indexed.get("chunks_indexed"),
        "documents_indexed": indexed.get("documents_indexed"),
        "embedding_model": indexed.get("embedding_model"),
        "whisper_model": settings.whisper_model
        if source != "provided_text"
        else None,
    }
    logger.info("process_media done: %s", result)
    return result
