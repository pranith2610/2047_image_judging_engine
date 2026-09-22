"""
Application configuration for THE 2047 - AI Image Comparison & Judging Engine.
"""
from pathlib import Path

import os
import tempfile

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Detect serverless / read-only filesystem (Vercel)
if os.getenv("VERCEL") or not os.access(BASE_DIR, os.W_OK):
    TEMP_BASE = Path(tempfile.gettempdir()) / "the_2047"
    UPLOAD_DIR = TEMP_BASE / "uploads"
    SESSIONS_DIR = TEMP_BASE / "sessions"
else:
    UPLOAD_DIR = BASE_DIR / "uploads"
    SESSIONS_DIR = BASE_DIR / "sessions"

REF_UPLOAD_DIR = UPLOAD_DIR / "reference"
PART_UPLOAD_DIR = UPLOAD_DIR / "participants"

# Ensure upload and session directories exist safely
try:
    REF_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PART_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    TEMP_BASE = Path(tempfile.gettempdir()) / "the_2047"
    UPLOAD_DIR = TEMP_BASE / "uploads"
    REF_UPLOAD_DIR = UPLOAD_DIR / "reference"
    PART_UPLOAD_DIR = UPLOAD_DIR / "participants"
    SESSIONS_DIR = TEMP_BASE / "sessions"
    REF_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PART_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Validation limits
MAX_PARTICIPANT_IMAGES = 60
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_MIME_TYPES = {
    "image/jpeg": [".jpg", ".jpeg"],
    "image/png": [".png"],
    "image/webp": [".webp"]
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_IMAGE_DIMENSION = 64
MAX_IMAGE_DIMENSION = 8192

# Scoring categories and max points
CATEGORY_WEIGHTS = {
    "semantic_similarity": 30.0,
    "object_accuracy": 25.0,
    "composition_spatial": 20.0,
    "color_lighting": 15.0,
    "fine_details": 10.0,
}
TOTAL_POINTS = 100.0

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
