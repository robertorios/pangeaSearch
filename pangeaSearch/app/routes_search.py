from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import AuthUser, get_current_user
from app.chroma_store import query_similar
from app.config import Settings, get_settings
from app.embeddings import embed_query
from app.rag import answer_from_hits, suggest_media

router = APIRouter(prefix="/api/v1", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Member situation text")
    top_k: Optional[int] = Field(None, ge=1, le=20)
    include_answer: bool = Field(
        True,
        description="If true, run Gemma RAG over retrieved hits (Part 4)",
    )


class SearchHit(BaseModel):
    media_id: Optional[int]
    kind: str = "chunk"
    chunk_index: Optional[int]
    title: str = ""
    text: str
    score: Optional[float]


class SuggestedMedia(BaseModel):
    media_id: int
    title: str = ""
    best_score: Optional[float]


class SearchResponse(BaseModel):
    query: str
    embedding_model: str
    answer: Optional[str] = None
    answer_source: Optional[str] = None
    hits: List[SearchHit]
    suggested_media: List[SuggestedMedia]


@router.post("/search", response_model=SearchResponse)
def search(
    body: SearchRequest,
    user: AuthUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> SearchResponse:
    """Embed query → Chroma hits → optional Gemma human answer (RAG)."""
    _ = user
    top_k = body.top_k or settings.search_top_k
    try:
        vector = embed_query(body.query.strip(), settings=settings)
        # Fetch extra candidates, then apply min-score so weak leftover matches drop out.
        raw_hits = query_similar(
            vector,
            top_k=max(top_k * 3, top_k),
            settings=settings,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed: %s" % exc,
        ) from exc

    min_score = settings.search_min_score
    filtered = [
        h
        for h in raw_hits
        if h.get("score") is None or float(h["score"]) >= min_score
    ][:top_k]

    hits = [
        SearchHit(
            media_id=h.get("media_id"),
            kind=h.get("kind") or "chunk",
            chunk_index=h.get("chunk_index"),
            title=h.get("title") or "",
            text=h.get("text") or "",
            score=h.get("score"),
        )
        for h in filtered
    ]
    suggested = [
        SuggestedMedia(
            media_id=int(m["media_id"]),
            title=m.get("title") or "",
            best_score=m.get("best_score"),
        )
        for m in suggest_media(filtered)
    ]

    answer = None
    answer_source = None
    if body.include_answer:
        answer, answer_source = answer_from_hits(
            body.query.strip(),
            filtered,
            settings=settings,
        )

    return SearchResponse(
        query=body.query.strip(),
        embedding_model=settings.embedding_model,
        answer=answer,
        answer_source=answer_source,
        hits=hits,
        suggested_media=suggested,
    )
