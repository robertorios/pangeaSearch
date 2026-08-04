"""Whisper transcription (lazy-loaded). Requires ffmpeg on PATH for media files."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model = None
_model_name: Optional[str] = None


def _load_whisper(name: str):
    import whisper

    logger.info("Loading Whisper model: %s", name)
    return whisper.load_model(name)


def get_whisper(settings: Optional[Settings] = None):
    global _model, _model_name
    settings = settings or get_settings()
    name = settings.whisper_model
    with _lock:
        if _model is None or _model_name != name:
            _model = _load_whisper(name)
            _model_name = name
        return _model


def transcribe_file(path: str, settings: Optional[Settings] = None) -> str:
    media_path = Path(path)
    if not media_path.is_file():
        raise FileNotFoundError("Media file not found: %s" % path)

    model = get_whisper(settings)
    logger.info("Transcribing %s", media_path)
    result = model.transcribe(str(media_path))
    text = (result.get("text") or "").strip()
    if not text:
        raise RuntimeError("Whisper returned empty transcript")
    return text
