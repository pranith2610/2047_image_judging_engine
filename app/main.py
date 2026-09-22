"""
Main FastAPI application entrypoint for THE 2047 judging engine.
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import BASE_DIR, UPLOAD_DIR
from app.api.routes import router as api_router

app = FastAPI(
    title="THE 2047 — AI Image Comparison & Judging Engine",
    description="Multi-dimensional visual judging and evaluation system for image recreation competitions.",
    version="0.1.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router)

# Mount static and uploaded files
STATIC_DIR = BASE_DIR / "app" / "static"
if not STATIC_DIR.exists():
    STATIC_DIR = Path(__file__).resolve().parent / "static"

try:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

app.mount("/static", StaticFiles(directory=str(STATIC_DIR), check_dir=False), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR), check_dir=False), name="uploads")


@app.get("/")
async def get_index():
    """Serve the primary competition dashboard UI."""
    index_path = BASE_DIR / "app" / "templates" / "index.html"
    if not index_path.exists():
        index_path = Path(__file__).resolve().parent / "templates" / "index.html"
    return FileResponse(str(index_path))
