# pangeaSearch — Docker (API + worker)

## Goal

Two containers from **one image**:

| Container | Role | Port |
|-----------|------|------|
| `pangea-search-api` | FastAPI search / process enqueue | **3003** |
| `pangea-search-worker` | RQ Process (Whisper → embed → Chroma) | none |

| Not in these containers | How |
|-------------------------|-----|
| **Chroma data** | Volume `./data` → `/app/data` (stays on disk) |
| **Redis** | Host `redis-server` (DB 1) |
| **Ollama** | Host `ollama serve` (optional) |
| **ffmpeg** | Installed **inside** the image |

## Prerequisites

1. Docker Desktop  
2. `redis-server` on the Mac  
3. `pangeaSearch/.env` with same `JWT_SECRET_KEY` as Users  
4. Free **3003** (stop host `uvicorn` and host `python -m app.worker`)  
5. Optional: `ollama serve` + `gemma2:2b` for real Gemma summaries/answers  

Existing indexes under `pangeaSearch/data/chroma` are reused via the volume mount.

## Commands

```bash
cd pangeaSearch

# First build can take a long time (torch, whisper, etc.)
docker compose up --build

# Detached
docker compose up --build -d

docker compose logs -f
docker compose logs -f pangea-search-worker
docker compose down
```

## Verify

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:3003/up
# expect 200

docker ps --format 'table {{.Names}}\t{{.Ports}}\t{{.Status}}'
# pangea-search-api   0.0.0.0:3003->3003
# pangea-search-worker (no host port)
```

Then SPA: situation search + Supervisor Process (worker logs should show activity).

## Env overrides (compose)

| Variable | Docker value | Why |
|----------|--------------|-----|
| `REDIS_URL` | `redis://host.docker.internal:6379/1` | Mac Redis |
| `GEMMA_API_URL` | `http://host.docker.internal:11434` | Mac Ollama |
| `CHROMA_PATH` | `/app/data/chroma` | Inside volume |
| `MEDIA_DOWNLOAD_DIR` | `/app/data/downloads` | Temp video files |
| `RQ_SIMPLE_WORKER` | `true` | Avoid fork issues with torch |

Other settings (JWT, embedding model, Whisper size) come from `.env`.

## Files

| File | Role |
|------|------|
| `Dockerfile.dev` | Python 3.11 + ffmpeg + requirements |
| `docker-compose.yml` | api + worker + `./data` volume |
| `.dockerignore` | Skip `.venv`, etc. |

## Hybrid stack after Search Docker

| Port | Service |
|------|---------|
| 3000–3002 | Users / Media / Conversations — Docker |
| **3003** | Search API — Docker |
| worker | Search RQ — Docker |
| 6379 | Redis — host |
| 11434 | Ollama — host |
| 5173 | SPA — host |
| MySQL | host |

## Next (later)

- Redis / Ollama as Compose services  
- One shared Docker network with Rails containers  
- Production multi-stage image for Hetzner  
