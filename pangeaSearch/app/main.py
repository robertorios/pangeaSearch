from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes_health import router as health_router
from app.routes_index import router as index_router
from app.routes_process import router as process_router
from app.routes_search import router as search_router

settings = get_settings()

app = FastAPI(
    title="pangeaSearch",
    description="Semantic media search and RAG answers for Testimony",
    version="0.5.0",
)

# Dev defaults; tighten via CORS_ALLOWED_ORIGINS in a later part.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(index_router)
app.include_router(search_router)
app.include_router(process_router)


@app.on_event("startup")
def _startup() -> None:
    if not settings.jwt_ready and settings.app_env == "production":
        raise RuntimeError("JWT_SECRET_KEY must be set in production")
