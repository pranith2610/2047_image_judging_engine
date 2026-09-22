"""
Application configuration for THE 2047 - AI Image Comparison & Judging Engine.
"""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
REF_UPLOAD_DIR = UPLOAD_DIR / "reference"
PART_UPLOAD_DIR = UPLOAD_DIR / "participants"
SESSIONS_DIR = BASE_DIR / "sessions"

# Ensure upload and session directories exist
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

import os

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
