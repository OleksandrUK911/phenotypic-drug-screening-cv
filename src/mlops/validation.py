from __future__ import annotations

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/tiff", "image/png", "image/jpeg"}
MAX_IMAGE_DIMENSION = 4096  # guards against decompression-bomb-style abuse


class UploadValidationError(ValueError):
    pass


def validate_upload(
    content_type: str,
    size_bytes: int,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Reject an inference-endpoint upload before it reaches the model.

    A public demo endpoint is a much softer target than a local research
    script: an oversized file, wrong MIME type, or a tiny-file-that-decodes-huge
    ("decompression bomb") can degrade or crash the service for everyone else.
    See TODO.md section 20.
    """
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise UploadValidationError(f"Unsupported content type: {content_type!r}. Allowed: {ALLOWED_CONTENT_TYPES}")

    if size_bytes > MAX_UPLOAD_BYTES:
        raise UploadValidationError(f"Upload too large: {size_bytes} bytes > {MAX_UPLOAD_BYTES} byte limit")

    if width is not None and width > MAX_IMAGE_DIMENSION:
        raise UploadValidationError(f"Image width {width} exceeds max {MAX_IMAGE_DIMENSION}")
    if height is not None and height > MAX_IMAGE_DIMENSION:
        raise UploadValidationError(f"Image height {height} exceeds max {MAX_IMAGE_DIMENSION}")
