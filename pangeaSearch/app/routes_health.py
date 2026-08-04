from fastapi import APIRouter, Depends

from app.auth import AuthUser, get_current_user
from app.config import Settings, get_settings

router = APIRouter(tags=["health"])


@router.get("/up")
def up(settings: Settings = Depends(get_settings)) -> dict:
    """Load balancer / smoke check — no auth."""
    return {
        "status": "ok",
        "service": "pangeaSearch",
        "jwt_configured": settings.jwt_ready,
        "env": settings.app_env,
        "embedding_model": settings.embedding_model,
        "chroma_collection": settings.chroma_collection,
    }


@router.get("/me")
def me(user: AuthUser = Depends(get_current_user)) -> dict:
    """Smoke-test JWT: returns the verified user_id from the access token."""
    return {"user_id": user.user_id}
