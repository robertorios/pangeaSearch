"""Redis + RQ queue for pangeaSearch background jobs."""

from typing import Optional

from redis import Redis
from rq import Queue

from app.config import Settings, get_settings

_redis: Optional[Redis] = None


def get_redis(settings: Optional[Settings] = None) -> Redis:
    global _redis
    settings = settings or get_settings()
    if _redis is None:
        _redis = Redis.from_url(settings.redis_url)
    return _redis


def get_queue(settings: Optional[Settings] = None) -> Queue:
    settings = settings or get_settings()
    return Queue(
        name=settings.rq_queue_name,
        connection=get_redis(settings),
        default_timeout=60 * 60,  # long videos / whisper
    )


def reset_queue_clients() -> None:
    global _redis
    _redis = None
