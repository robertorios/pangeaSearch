from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from rq.job import Job

from app.auth import decode_access_token
from app.config import Settings, get_settings
from app.jobs import process_media
from app.queue import get_queue, get_redis

router = APIRouter(prefix="/api/v1", tags=["process"])
_bearer = HTTPBearer(auto_error=False)


class ProcessRequest(BaseModel):
    media_id: int = Field(..., gt=0)
    title: Optional[str] = ""
    source_url: Optional[str] = None
    local_path: Optional[str] = None
    transcript_text: Optional[str] = Field(
        None,
        description="If set, skip Whisper and index this text (pipeline test)",
    )


class ProcessEnqueueResponse(BaseModel):
    job_id: str
    media_id: int
    queue: str
    status: str = "queued"


def require_user_or_internal(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    settings: Settings = Depends(get_settings),
    x_internal_token: Optional[str] = Header(None, alias="X-Internal-Token"),
) -> str:
    """Allow access JWT or internal service token."""
    expected = (settings.internal_service_token or "").strip()
    provided = (x_internal_token or "").strip()
    if expected and provided and provided == expected:
        return "internal"

    if not settings.jwt_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT_SECRET_KEY is not configured",
        )
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthenticated",
        )
    try:
        payload = decode_access_token(credentials.credentials, settings.jwt_secret_key)
        user_id = int(payload["user_id"])
        if user_id <= 0:
            raise ValueError("invalid user_id")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthenticated",
        ) from None
    return "user:%s" % user_id


@router.post("/process", response_model=ProcessEnqueueResponse)
def enqueue_process(
    body: ProcessRequest,
    settings: Settings = Depends(get_settings),
    actor: str = Depends(require_user_or_internal),
) -> ProcessEnqueueResponse:
    """Enqueue RQ background indexing (Whisper → embed → Chroma).

    Prefer running this **before** a live demo, not during it.
    """
    _ = actor

    if not (body.transcript_text or body.local_path or body.source_url):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide transcript_text, local_path, or source_url",
        )

    try:
        queue = get_queue(settings)
        job = queue.enqueue(
            process_media,
            media_id=body.media_id,
            title=body.title or "",
            source_url=body.source_url,
            local_path=body.local_path,
            transcript_text=body.transcript_text,
            job_timeout=60 * 60,
            result_ttl=60 * 60 * 24,
            failure_ttl=60 * 60 * 24,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not enqueue job (is Redis running?): %s" % exc,
        ) from exc

    return ProcessEnqueueResponse(
        job_id=job.id,
        media_id=body.media_id,
        queue=settings.rq_queue_name,
        status="queued",
    )


@router.get("/process/{job_id}")
def process_status(
    job_id: str,
    settings: Settings = Depends(get_settings),
    actor: str = Depends(require_user_or_internal),
) -> Dict[str, Any]:
    _ = actor
    try:
        job = Job.fetch(job_id, connection=get_redis(settings))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        ) from None

    payload: Dict[str, Any] = {
        "job_id": job.id,
        "status": job.get_status(),
        "enqueued_at": str(job.enqueued_at) if job.enqueued_at else None,
        "started_at": str(job.started_at) if job.started_at else None,
        "ended_at": str(job.ended_at) if job.ended_at else None,
    }
    if job.is_finished:
        payload["result"] = job.result
    if job.is_failed:
        payload["error"] = str(job.exc_info) if job.exc_info else "failed"
    return payload
