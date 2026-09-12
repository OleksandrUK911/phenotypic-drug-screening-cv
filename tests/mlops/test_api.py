from fastapi.testclient import TestClient

from src.mlops.api import create_app
from src.mlops.rate_limiter import RateLimiter


class _FakePredictor:
    def predict(self, image_bytes: bytes) -> dict:
        return {"anomaly_score": 0.42, "predicted_moa": "microtubule_destabilizer"}


def _client(rate_limiter: RateLimiter | None = None) -> TestClient:
    app = create_app(predictor=_FakePredictor(), rate_limiter=rate_limiter)
    return TestClient(app)


def test_health_check():
    client = _client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_returns_predictor_output_for_valid_upload():
    client = _client()
    resp = client.post("/predict", files={"file": ("cell.png", b"fake-png-bytes", "image/png")})
    assert resp.status_code == 200
    assert resp.json() == {"anomaly_score": 0.42, "predicted_moa": "microtubule_destabilizer"}


def test_predict_rejects_unsupported_content_type():
    client = _client()
    resp = client.post("/predict", files={"file": ("cell.zip", b"stuff", "application/zip")})
    assert resp.status_code == 415


def test_predict_rejects_oversized_upload():
    client = _client()
    big_payload = b"0" * (11 * 1024 * 1024)
    resp = client.post("/predict", files={"file": ("cell.png", big_payload, "image/png")})
    assert resp.status_code == 415


def test_predict_enforces_rate_limit():
    limiter = RateLimiter(max_requests=1, window_seconds=60.0, clock=lambda: 0.0)
    client = _client(rate_limiter=limiter)

    first = client.post("/predict", files={"file": ("cell.png", b"data", "image/png")})
    second = client.post("/predict", files={"file": ("cell.png", b"data", "image/png")})

    assert first.status_code == 200
    assert second.status_code == 429


def test_predict_with_no_trained_model_returns_503():
    from src.mlops.api import create_app as create_app_default

    client = TestClient(create_app_default())
    resp = client.post("/predict", files={"file": ("cell.png", b"data", "image/png")})
    assert resp.status_code == 503
