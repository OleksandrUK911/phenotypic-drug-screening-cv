import pytest

from src.mlops.validation import MAX_UPLOAD_BYTES, UploadValidationError, validate_upload


def test_valid_upload_passes():
    validate_upload(content_type="image/png", size_bytes=1000, width=512, height=512)


def test_rejects_unsupported_content_type():
    with pytest.raises(UploadValidationError, match="Unsupported content type"):
        validate_upload(content_type="application/zip", size_bytes=1000)


def test_rejects_oversized_upload():
    with pytest.raises(UploadValidationError, match="too large"):
        validate_upload(content_type="image/png", size_bytes=MAX_UPLOAD_BYTES + 1)


def test_rejects_oversized_dimensions():
    with pytest.raises(UploadValidationError, match="width"):
        validate_upload(content_type="image/tiff", size_bytes=1000, width=10000, height=100)
