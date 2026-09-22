"""
Unit tests for the image validator.
"""
import io
import pytest
from PIL import Image
from app.core.validator import ImageValidator
from app.config import MAX_FILE_SIZE_BYTES, MIN_IMAGE_DIMENSION


def create_test_image_bytes(width=200, height=200, fmt="PNG", color=(255, 0, 0)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_valid_image():
    data = create_test_image_bytes(300, 300, "PNG")
    val = ImageValidator.validate_file_bytes("test_sample.png", data)
    assert val.valid is True
    assert val.dimensions == (300, 300)
    assert val.error is None


def test_invalid_extension():
    data = create_test_image_bytes(300, 300, "PNG")
    val = ImageValidator.validate_file_bytes("malicious.exe", data)
    assert val.valid is False
    assert "Unsupported format" in val.error


def test_corrupted_image_bytes():
    corrupted_data = b"NOT_A_VALID_IMAGE_PAYLOAD_12345"
    val = ImageValidator.validate_file_bytes("corrupt.png", corrupted_data)
    assert val.valid is False
    assert "Corrupted or unreadable" in val.error


def test_image_dimensions_too_small():
    data = create_test_image_bytes(MIN_IMAGE_DIMENSION - 10, 100, "PNG")
    val = ImageValidator.validate_file_bytes("small.png", data)
    assert val.valid is False
    assert "too small" in val.error


def test_empty_file():
    val = ImageValidator.validate_file_bytes("empty.png", b"")
    assert val.valid is False
    assert "empty" in val.error


def test_sanitize_filename():
    unsafe = "../../../secret_path/evil image#$%.png"
    clean = ImageValidator.sanitize_filename(unsafe)
    assert ".." not in clean
    assert "/" not in clean
    assert "\\" not in clean
    assert clean.endswith(".png")
