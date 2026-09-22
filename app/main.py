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
STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.get("/")
async def get_index():
    """Serve the primary competition dashboard UI."""
    index_path = BASE_DIR / "app" / "templates" / "index.html"
    return FileResponse(str(index_path))
