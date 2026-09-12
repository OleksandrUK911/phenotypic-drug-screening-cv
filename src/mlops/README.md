# mlops

Training/inference scripts, MLflow tracking, Dockerized FastAPI inference service.

## Implemented

- `api.py` — FastAPI app (`/health`, `/predict`) with a pluggable `Predictor` protocol via
  `app.state.predictor`, so tests inject a fake predictor instead of needing a trained model.
  Returns `503` on `/predict` until a real model is wired in.
- `validation.py` — upload hardening: rejects unsupported content types, oversized files
  (>10MB), and oversized image dimensions (>4096px), before anything reaches the model.
- `rate_limiter.py` — dependency-free fixed-window rate limiter, keyed per client IP. Adequate
  for a single-process demo; a multi-instance deployment would need a shared store (Redis) instead
  of the in-memory dict here.

## Not yet implemented

- MLflow tracking wiring, `scripts/train.py`/`evaluate.py`/`infer.py`, Dockerfiles, a real
  `Predictor` implementation connecting segmentation → features → anomaly/MOA scoring, structured
  logging, and a metrics endpoint. See root `TODO.md` sections 11 and 20.
