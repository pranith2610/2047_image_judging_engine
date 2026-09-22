"""
Image validation pipeline for THE 2047 judging engine.
Performs format, size, dimension, and integrity checks.
"""
import io
import re
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image

from app.config import (
    MAX_FILE_SIZE_BYTES,
    ALLOWED_EXTENSIONS,
    MIN_IMAGE_DIMENSION,
    MAX_IMAGE_DIMENSION,
)
from app.models.schemas import ValidationResponse


class ImageValidator:
    """Validates uploaded images for competition processing."""

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent directory traversal or invalid characters."""
        clean = Path(filename).name
        # Keep only alphanumeric, hyphens, underscores and period
        clean = re.sub(r'[^a-zA-Z0-9._-]', '_', clean)
        return clean or "unnamed_image.png"

    @classmethod
    def validate_file_bytes(cls, filename: str, file_bytes: bytes) -> ValidationResponse:
        """Validate raw bytes of an uploaded image file."""
        sanitized = cls.sanitize_filename(filename)
        file_size = len(file_bytes)

        # 1. Size check
        if file_size == 0:
            return ValidationResponse(
                valid=False,
                filename=sanitized,
                error="Uploaded file is empty (0 bytes)."
            )

        if file_size > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
            actual_mb = file_size / (1024 * 1024)
            return ValidationResponse(
                valid=False,
                filename=sanitized,
                error=f"File exceeds maximum size limit of {max_mb:.1f}MB (was {actual_mb:.2f}MB).",
                file_size_bytes=file_size
            )

        # 2. Extension check
        ext = Path(sanitized).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return ValidationResponse(
                valid=False,
                filename=sanitized,
                error=f"Unsupported format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
                file_size_bytes=file_size
            )

        # 3. Image integrity and dimension check with Pillow
        try:
            image_stream = io.BytesIO(file_bytes)
            with Image.open(image_stream) as img:
                img.verify()  # Verify file header and integrity
            
            # Reopen stream to read image metadata (verify closes it or resets state)
            image_stream.seek(0)
            with Image.open(image_stream) as img:
                width, height = img.size
                img_format = img.format or "UNKNOWN"
                mime_type = f"image/{img_format.lower()}"

                if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                    return ValidationResponse(
                        valid=False,
                        filename=sanitized,
                        error=f"Dimensions {width}x{height} too small. Minimum dimension is {MIN_IMAGE_DIMENSION}px.",
                        dimensions=(width, height),
                        file_size_bytes=file_size,
                        mime_type=mime_type
                    )

                if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                    return ValidationResponse(
                        valid=False,
                        filename=sanitized,
                        error=f"Dimensions {width}x{height} exceed limit of {MAX_IMAGE_DIMENSION}px.",
                        dimensions=(width, height),
                        file_size_bytes=file_size,
                        mime_type=mime_type
                    )

                return ValidationResponse(
                    valid=True,
                    filename=sanitized,
                    dimensions=(width, height),
                    file_size_bytes=file_size,
                    mime_type=mime_type
                )

        except Exception as e:
            return ValidationResponse(
                valid=False,
                filename=sanitized,
                error=f"Corrupted or unreadable image file: {str(e)}",
                file_size_bytes=file_size
            )

    @classmethod
    def validate_file_path(cls, file_path: Path) -> ValidationResponse:
        """Validate an existing file path."""
        if not file_path.exists():
            return ValidationResponse(
                valid=False,
                filename=file_path.name,
                error="File does not exist on disk."
            )
        data = file_path.read_bytes()
        return cls.validate_file_bytes(file_path.name, data)
