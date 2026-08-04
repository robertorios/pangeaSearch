from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    jwt_secret_key: str = ""
    internal_service_token: str = ""
    app_env: str = "development"
    log_level: str = "info"

    # EmbeddingGemma (default). For a tiny smoke test on a slow machine you can
    # temporarily set EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
    embedding_model: str = "google/embeddinggemma-300m"
    chroma_path: str = str(Path(__file__).resolve().parent.parent / "data" / "chroma")
    chroma_collection: str = "media_chunks"
    chunk_size: int = 800
    chunk_overlap: int = 120
    search_top_k: int = 5
    # Drop weak Chroma hits (cosine similarity). 0.35 filters noise; raise if too strict.
    search_min_score: float = 0.35

    # Part 3 — RQ + Whisper
    redis_url: str = "redis://127.0.0.1:6379/1"
    rq_queue_name: str = "pangea_search"
    whisper_model: str = "tiny"
    media_download_dir: str = str(
        Path(__file__).resolve().parent.parent / "data" / "downloads"
    )

    # Extended Part 3 / Part 4 — Gemma via Ollama (fallback if Ollama is down)
    summarize_enabled: bool = True
    rag_enabled: bool = True
    gemma_api_url: str = "http://127.0.0.1:11434"
    gemma_model: str = "gemma2:2b"
    gemma_timeout_seconds: float = 120.0

    @property
    def jwt_ready(self) -> bool:
        return bool(self.jwt_secret_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
