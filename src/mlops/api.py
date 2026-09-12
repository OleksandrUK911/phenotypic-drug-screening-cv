from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI, HTTPException, Request, UploadFile

from src.mlops.rate_limiter import RateLimiter
from src.mlops.validation import UploadValidationError, validate_upload


class Predictor(Protocol):
    def predict(self, image_bytes: bytes) -> dict: ...


class _NotLoadedPredictor:
    """Default predictor: no trained model is wired in yet (see TODO.md section 11).

    Kept as an explicit, honestly-named stand-in rather than mock predictions,
    so the API is truthful about pipeline status until a real model is trained
    and swapped in via `app.state.predictor`.
    """

    def predict(self, image_bytes: bytes) -> dict:
        raise HTTPException(status_code=503, detail="Model not yet trained/loaded — pipeline is still in development.")


def create_app(predictor: Predictor | None = None, rate_limiter: RateLimiter | None = None) -> FastAPI:
    app = FastAPI(title="Phenotypic Drug Screening Inference API")
    app.state.predictor = predictor or _NotLoadedPredictor()
    app.state.rate_limiter = rate_limiter or RateLimiter(max_requests=30, window_seconds=60.0)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(request: Request, file: UploadFile) -> dict:
        client_key = request.client.host if request.client else "unknown"
        if not app.state.rate_limiter.allow(client_key):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        contents = await file.read()
        try:
            validate_upload(content_type=file.content_type or "", size_bytes=len(contents))
        except UploadValidationError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc

        return app.state.predictor.predict(contents)

    return app


app = create_app()
