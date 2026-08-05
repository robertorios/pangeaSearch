# pangeaSearch — semantic media search (RAG)

Python FastAPI service: index transcripts into Chroma, search by situation text, answer with Gemma.

## Parts

| Part | Status |
|------|--------|
| 1 JWT + health | done |
| 2 Embed + Chroma index/search | done |
| 3 RQ + Whisper process job | done |
| 3b Summary + chunk indexing | done (extended) |
| 4 Gemma RAG answer | done |
| 5 Admin + SPA | done |

## Run (local)

Terminal A — API:

```bash
cd pangeaSearch
source .venv/bin/activate
uvicorn app.main:app --reload --port 3003
```

Terminal B — Redis (if not already running for Sidekiq):

```bash
redis-server
```

Terminal C — RQ worker:

```bash
cd pangeaSearch
source .venv/bin/activate
python -m app.worker
```

Whisper on real video/audio needs **ffmpeg** installed (`brew install ffmpeg`).

### Part 2 / 3b endpoints (Bearer JWT)

- `POST /api/v1/index` — summarize + embed summary **and** transcript chunks  
- `POST /api/v1/search` — situation → hits + **Gemma human answer** + `suggested_media`  
- `GET /api/v1/index/stats`

Summary and RAG answers use Ollama Gemma (`GEMMA_API_URL` / `GEMMA_MODEL`); if Ollama is down, extractive/template fallbacks are used. Set `"include_answer": false` on search to skip RAG.

### Part 3 endpoints (Bearer JWT or `X-Internal-Token`)

- `POST /api/v1/process` — enqueue job  
  Body (one of): `transcript_text` | `local_path` | `source_url`  
  plus `media_id`, optional `title`  
- `GET /api/v1/process/{job_id}` — status / result  

Example (skip Whisper, test RQ → embed → Chroma):

```json
{
  "media_id": 201,
  "title": "Demo",
  "transcript_text": "After my divorce I felt alone..."
}
```

### Part 5 — SPA wiring

- Member Home (`/memberhome`): situation box → `POST /api/v1/search` (Bearer JWT)
- Supervisor Media tab: **Process** → `POST /api/v1/process` with `source_url` from `mediaUrl`
- SPA config: `getSearchApiUrl()` → `http://localhost:3003`

| Service | Port |
|---------|------|
| pangeaUsers | 3000 |
| pangeaMedia | 3001 |
| pangeaConversations | 3002 |
| **pangeaSearch** | **3003** |

## Docker (API + worker)

See **`DOCKER.md`**. Short form:

```bash
# redis-server (and optional ollama serve) on the Mac
cd pangeaSearch
docker compose up --build -d
curl http://127.0.0.1:3003/up
```

- **api:** `pangea-search-api` → :3003  
- **worker:** `pangea-search-worker` (RQ Process)  
- **Chroma / caches:** volume `./data`  
- **Redis / Ollama:** host via `host.docker.internal`  
