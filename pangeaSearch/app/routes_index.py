from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import AuthUser, get_current_user
from app.chroma_store import collection_stats
from app.config import Settings, get_settings
from app.index_pipeline import index_transcript

router = APIRouter(prefix="/api/v1", tags=["index"])


class IndexRequest(BaseModel):
    media_id: int = Field(..., gt=0)
    text: str = Field(..., min_length=1, description="Transcript or pasted text to index")
    title: Optional[str] = ""


class IndexResponse(BaseModel):
    media_id: int
    summary: str
    summary_source: str
    summary_indexed: int
    chunks_indexed: int
    documents_indexed: int
    embedding_model: str
    collection_count: int


@router.post("/index", response_model=IndexResponse)
def index_text(
    body: IndexRequest,
    user: AuthUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> IndexResponse:
    """Summarize + embed summary and transcript chunks into Chroma."""
    _ = user
    try:
        indexed = index_transcript(
            media_id=body.media_id,
            transcript=body.text,
            title=body.title or "",
            settings=settings,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Indexing failed: %s" % exc,
        ) from exc

    stats = collection_stats(settings)
    return IndexResponse(
        media_id=indexed["media_id"],
        summary=indexed["summary"],
        summary_source=indexed["summary_source"],
        summary_indexed=indexed["summary_indexed"],
        chunks_indexed=indexed["chunks_indexed"],
        documents_indexed=indexed["documents_indexed"],
        embedding_model=indexed["embedding_model"],
        collection_count=stats["count"],
    )


@router.get("/index/stats")
def index_stats(
    user: AuthUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> dict:
    stats = collection_stats(settings)
    stats["embedding_model"] = settings.embedding_model
    stats["requested_by"] = user.user_id
    return stats
