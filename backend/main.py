import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Load .env from the backend directory (sibling of this file).
load_dotenv()

# NOTE: heavy ML imports (sentence-transformers, chromadb, sklearn models)
# are deferred to first request so the port binds before the 512Mi free-tier
# limit is hit at boot. Do NOT import run_agent at module top.
_run_agent = None


def get_run_agent():
    global _run_agent
    if _run_agent is None:
        from src.agent.graph import run_agent  # noqa: WPS433 (lazy import on purpose)

        _run_agent = run_agent
    return _run_agent

app = FastAPI(title="News Credibility Monitor API")

# CORS — allow the deployed frontend (Vercel) plus local dev to hit the API.
# The production Vercel URL is always allowed. Extra origins can be added via
# the CORS_ALLOW_ORIGINS env var in the Render dashboard (comma-separated).
_default_origins = [
    "https://news-credibility-monitor.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
_env_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
]
allow_origins = _env_origins + _default_origins if _env_origins else _default_origins

# Optional: regex to match Vercel preview deployments
# (e.g. https://news-credibility-monitor-git-*-yourname.vercel.app).
allow_origin_regex = os.getenv("CORS_ALLOW_ORIGIN_REGEX") or None

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ──

class AnalyzeRequest(BaseModel):
    text: str
    mode: str = "agentic"


@app.get("/")
def home():
    return {"status": "ok", "message": "News Credibility Monitor API is running"}


@app.post("/analyze")
def analyze(data: AnalyzeRequest):
    if not data.text.strip() or len(data.text.split()) < 50:
        raise HTTPException(
            status_code=422,
            detail="Text too short. Please provide at least 50 words for analysis.",
        )

    try:
        # Lazy-load the agent graph so cold-boot fits in Render's 512Mi free tier.
        result = get_run_agent()(data.text)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
