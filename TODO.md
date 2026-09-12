# Project TODO

Master checklist for the phenotypic drug screening CV pipeline. Each module also has its own
`TODO.md` with implementation-level detail — this file tracks project-wide milestones and
cross-cutting concerns that don't belong to a single module.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done

## 0. Repo foundations

- [ ] `pyproject.toml` — package `src` as an installable module (`pip install -e .`), pin Python version
- [ ] `LICENSE` (MIT)
- [ ] `CITATION.cff` — cite BBBC021/JUMP-CP dataset papers and Cellpose/SimCLR/DINO papers used
- [ ] `.pre-commit-config.yaml` — black, ruff, isort, mypy (basic), nbstripout for notebooks
- [ ] `.github/workflows/ci.yml` — lint + type-check + pytest on push/PR
- [ ] `.github/ISSUE_TEMPLATE/` — bug report + experiment-tracking issue templates
- [ ] `Makefile` or `justfile` — `make setup`, `make train`, `make test`, `make lint` shortcuts
- [ ] `docker-compose.yml` — local MLflow tracking server + inference API service

## 1. Data layer (`src/data/`) — currently missing entirely

- [ ] `download.py` — programmatic fetch of BBBC021 / JUMP-CP subset (checksum-verified, resumable)
- [ ] `preprocessing.py` — illumination correction, per-channel normalization, tiling of large plate scans
- [ ] `dataset.py` — PyTorch `Dataset`/`Dataset` classes for multi-channel Cell Painting images (5 stains)
- [ ] `splits.py` — **plate-aware** train/val/test splitting (never split by image — causes batch-effect leakage)
- [ ] `schema.py` — typed metadata schema (plate ID, well, compound, concentration, replicate)
- [ ] `data/README.md` — document expected raw layout, licensing/usage terms of BBBC021/JUMP-CP
- [ ] Sanity-check notebook: class balance, missing wells, per-plate intensity distributions

## 2. Segmentation (`src/segmentation/`)

- [ ] Baseline: pretrained Cellpose inference wrapper (fast path to get masks without training)
- [ ] U-Net trained from scratch on BBBC labeled subset (learning exercise / fallback if Cellpose insufficient)
- [ ] Post-processing: watershed splitting of touching cells, small-object filtering
- [ ] Instance-level QC metrics: per-image cell count sanity bounds, mask-area outlier flagging
- [ ] Export: per-cell crops + per-cell metadata table (feeds `features/`)

## 3. Self-supervised features (`src/features/`)

- [ ] SimCLR or DINO pretraining loop on unlabeled per-cell crops
- [ ] Embedding extraction pipeline (backbone → fixed-size vector per cell)
- [ ] Aggregate to per-well embeddings (mean/median pooling across cells)
- [ ] **Batch-effect correction**: ComBat or Typical Variation Normalization (TVN) on well-level embeddings
- [ ] Embedding quality check: UMAP/t-SNE plot colored by plate — should NOT cluster by plate after correction
- [ ] Ablation: with vs. without batch correction, effect on downstream MOA accuracy

## 4. Anomaly detection (`src/anomaly_detection/`)

- [ ] Autoencoder reconstruction-error scoring vs. DMSO/vehicle control wells
- [ ] Alternative: Mahalanobis distance / kNN distance in corrected embedding space
- [ ] **Z-factor / SSMD** computation per plate — standard HCS assay-quality metrics (shows domain depth)
- [ ] Hit-calling threshold selection + ROC-AUC against known active compounds (if labels available)
- [ ] Visualization: control vs. hit phenotype gallery (example images side by side)

## 5. MOA classification (`src/moa_classification/`)

- [ ] Multi-class classifier on corrected embeddings (compound → mechanism-of-action label)
- [ ] Handle **label noise** explicitly (documented MOA annotations are known to be imperfect) — e.g. label smoothing or noise-robust loss
- [ ] Uncertainty quantification: Deep Ensembles or MC Dropout, calibration curve (reliability diagram)
- [ ] Confusion matrix analysis grouped by MOA family, not just raw accuracy

## 6. Active learning (`src/active_learning/`) — NEW MODULE

- [ ] Simulate a compound-prioritization loop: start with small labeled subset, use model uncertainty
      (or embedding-space diversity) to pick next batch of compounds to "screen"
- [ ] Compare active-learning sampling vs. random sampling — learning curve (accuracy vs. #compounds screened)
- [ ] Short write-up connecting this to real AI-driven drug discovery triage (Exscientia-style narrative)

## 7. Explainability (`src/explainability/`)

- [ ] Grad-CAM (or attention rollout if using a ViT backbone) over per-cell crops
- [ ] Aggregate explanation maps per MOA class — "what morphological region drives this MOA call"
- [ ] Sanity check: explanations should highlight cell body/nucleus regions, not background artifacts

## 8. Evaluation & baselines (`src/evaluation/`) — NEW MODULE

- [ ] Classical baseline: CellProfiler feature extraction + simple classifier (RandomForest/SVM)
- [ ] Side-by-side benchmark table: classical pipeline vs. this deep pipeline (accuracy, F1, Z-factor, compute cost)
- [ ] Standardized metrics module shared across anomaly_detection/moa_classification (`src/utils/metrics.py`)
- [ ] `docs/results.md` — final numbers, plots, before/after segmentation examples

## 9. Multimodal fusion (`src/multimodal/`) — stretch goal, optional

- [ ] Molecular graph embeddings from SMILES (simple GNN, e.g. via RDKit + PyTorch Geometric)
- [ ] Fuse image embeddings + molecular graph embeddings for MOA classification
- [ ] Ablation: image-only vs. molecule-only vs. fused — does fusion actually help?

## 10. Transfer-to-defense demo (`src/transfer_defense_demo/`)

- [ ] Pick small public aerial-imagery dataset (DOTA or xView subset)
- [ ] Fine-tune the SSL backbone from `src/features/` on this dataset for object segmentation/anomaly detection
- [ ] Short comparative write-up: what transferred well, what didn't, why (framing for defense/remote-sensing roles)
- [ ] Explicit disclaimer: portfolio demo only, no operational claims

## 11. MLOps (`src/mlops/`)

- [ ] MLflow experiment tracking wired into all training scripts (params, metrics, artifacts)
- [ ] `scripts/train.py`, `scripts/evaluate.py`, `scripts/infer.py` — CLI entry points using `configs/*.yaml`
- [ ] Dockerfile for training environment + separate lightweight Dockerfile for inference
- [ ] FastAPI inference service: upload image → segmentation mask + anomaly score + MOA prediction
- [ ] Minimal frontend or Streamlit demo hitting the FastAPI service (for portfolio screenshots/GIF)

## 12. Configs (`configs/`)

- [ ] Hydra/OmegaConf YAML per experiment (segmentation.yaml, ssl_pretrain.yaml, moa_classifier.yaml, etc.)
- [ ] Environment config (paths, seeds, device) separated from model/experiment config

## 13. Testing (`tests/`)

- [ ] Unit tests for preprocessing (normalization correctness, shape checks)
- [ ] Unit tests for dataset/dataloader (batch shapes, plate-aware split correctness — no leakage)
- [ ] Segmentation smoke test (runs on a tiny fixture image, checks mask shape/dtype)
- [ ] Metrics unit tests (Z-factor/SSMD computed against known worked example)
- [ ] CI runs all of the above on every push

## 14. Documentation & portfolio polish

- [ ] `docs/architecture.md` — Mermaid pipeline diagram (data → segmentation → SSL → anomaly/MOA → explainability)
- [ ] `docs/results.md` — metrics tables + qualitative examples
- [ ] README: demo GIF/screenshot of FastAPI inference in action
- [ ] README: explicit "UK context" section referencing Exscientia/Recursion-style methodology and DASA-relevant transferability
- [ ] Write a short blog-style `docs/writeup.md` summarizing findings — good for LinkedIn/portfolio link

## 15. Reproducibility & versioning

- [ ] `DATASET_CARD.md` (in `data/`) — source, license terms of BBBC021/JUMP-CP, known biases/limitations, intended use
- [ ] `docs/model_card.md` — per Google/HF model-card convention: intended use, training data, evaluation results, known limitations, out-of-scope uses (important given pharma/defense-adjacent framing)
- [ ] DVC (or lightweight alternative: tracked hash manifest) for dataset/model artifact versioning — avoids "which weights produced these numbers" ambiguity
- [ ] Environment lockfile: `requirements.lock` or `poetry.lock` / `environment.yml` pinned versions, not just loose `requirements.txt`
- [ ] `CHANGELOG.md` (Keep a Changelog format) — tag milestones as they're completed
- [ ] Document exact hardware/compute budget used (GPU type, hours) — also note a **Colab/Kaggle-notebook path** for reviewers without a GPU to reproduce a small-scale run
- [ ] Model export: TorchScript/ONNX for the final inference model, with a load/inference smoke test

## 16. Ethics, compliance & responsible-use framing

- [ ] `ETHICS.md` — explicit statement: public data only, no PII, no clinical/diagnostic claims, no operational defense claims (ties into the transfer-to-defense demo's disclaimer)
- [ ] Dual-use awareness note: briefly acknowledge that phenotypic screening + defense-imagery transfer touches dual-use-adjacent territory, and state the project's boundaries explicitly (portfolio/research only)
- [ ] License compliance check for BBBC021/JUMP-CP redistribution terms (don't recommit raw data to the repo — already gitignored, but document *why* in `data/DATASET_CARD.md`)
- [ ] `.github/dependabot.yml` — automated dependency vulnerability alerts (cheap, signals security hygiene)

## 17. Deployment & live demo (portfolio-critical)

- [ ] Host a lightweight interactive demo (Streamlit/Gradio) on Hugging Face Spaces or Render free tier, linked at the top of the README — recruiters trying a live demo beats reading code
- [ ] `docs/uk_positioning.md` — standalone one-pager connecting the project to Exscientia/Recursion-style methodology and DASA-relevant transferability; reusable text block for cover letters/LinkedIn, kept separate from README so it can evolve independently
- [ ] Record a 60–90s demo GIF/video of the FastAPI + Streamlit flow for the README hero section
- [ ] `docs/related_work.md` — table comparing this project's MOA-classification results against published BBBC021 benchmark numbers from literature (shows awareness of SOTA, not just "it works")

## 18. Documentation site & repo polish

- [ ] `mkdocs.yml` + `docs/` as an MkDocs Material site, published via GitHub Pages — turns scattered `.md` files into a browsable site (nice portfolio link separate from the raw repo)
- [ ] README badges: CI status, license, Python version, (optional) docs-site link
- [ ] `CONTRIBUTING.md` — even solo, states code style/PR conventions; signals professionalism to reviewers
- [ ] `CODEOWNERS` — trivial for a solo repo but a recognized convention
- [ ] `.env.example` — documents expected env vars (MLflow tracking URI, etc.) without committing secrets

## 19. Scalability & large-data engineering

- [ ] Note in `data/DATASET_CARD.md` the realistic scale (JUMP-CP is multi-terabyte) and explicitly scope down to a
      tractable subset (e.g. a handful of plates/compounds) for the portfolio version — state this decision explicitly
      so it reads as a deliberate scoping choice, not an oversight
- [ ] Efficient data format for training: WebDataset or FFCV instead of loose file reads, if subset is still large
- [ ] Cloud storage integration (S3/GCS) for the dataset subset if it doesn't fit comfortably on local disk
- [ ] Multi-GPU / mixed-precision training notes in `src/mlops/` (even if only tested on a single GPU, document the
      `torch.cuda.amp` + `DistributedDataParallel` path for credibility)
- [ ] Document expected wall-clock/cost for each pipeline stage (segmentation, SSL pretraining, classification) at the
      chosen data scale — ties into the compute-budget doc from section 15

## 20. API hardening & observability (production-readiness of the demo service)

- [ ] Input validation on the FastAPI inference endpoint: max upload size, allowed MIME types/extensions,
      image-dimension bounds (defends against decompression-bomb-style abuse on a public demo)
- [ ] Rate limiting (e.g. `slowapi`) on the public-facing demo endpoint
- [ ] Structured request/response logging (no PII/raw images retained by default)
- [ ] Basic health-check endpoint (`/health`) + readiness check for the loaded model
- [ ] Optional: lightweight metrics endpoint (Prometheus-format) — request count, latency histogram, error rate
- [ ] Document these hardening choices in `src/mlops/README.md` — shows you thought about a *public* demo differently
      from a local research script

## 21. Statistical rigor

- [ ] Report confidence intervals (bootstrap) alongside point-estimate metrics (accuracy, Z-factor, AUC), not just
      single numbers — distinguishes a portfolio project from a tutorial copy-paste
- [ ] Multiple-seed reruns for headline results (mean ± std across ≥3 seeds) where compute allows
- [ ] Significance test (e.g. paired bootstrap) when claiming the deep pipeline "beats" the CellProfiler baseline in
      `docs/results.md` — an unqualified single-number comparison is a common credibility gap in ML portfolios

## 22. Developer experience

- [ ] `requirements-dev.txt` — separate dev-only deps (pytest, black, ruff, mypy, jupyter) from runtime `requirements.txt`
- [ ] `.devcontainer/devcontainer.json` — one-click reproducible dev environment (VS Code Dev Containers/Codespaces)
- [ ] `.github/PULL_REQUEST_TEMPLATE.md` — checklist (tests pass, lint clean, TODO.md updated) even for solo use
- [ ] Config validation via pydantic/dataclasses schemas for the Hydra configs in `configs/` — catch typos/bad values
      at config-load time rather than mid-training
- [ ] `docs/faq.md` — anticipate likely interviewer questions ("why plate-aware splits?", "why Z-factor?", "what
      would you do with more compute/data?") — doubles as interview prep while looking like documentation rigor

## Suggested build order (milestones)

1. Data layer + segmentation baseline (Cellpose) → get per-cell crops flowing
2. SSL pretraining + batch-effect correction → validate embeddings look sane (UMAP)
3. Anomaly detection + Z-factor/SSMD → first "real" scientific result
4. MOA classification + uncertainty + explainability → core ML story complete
5. Evaluation vs. CellProfiler baseline → quantify the "why deep learning helps" claim
6. MLOps (tracking, Docker, FastAPI) → make it demoable
7. Active learning module → adds an "AI-driven discovery" narrative
8. Transfer-to-defense demo → cross-domain positioning
9. Multimodal fusion → stretch goal if time allows
10. Reproducibility & versioning pass (lockfiles, model/dataset cards, CHANGELOG) → results are trustworthy/reproducible
11. Ethics/compliance pass (ETHICS.md, dependabot, license checks) → responsible-use framing is explicit
12. Deployment: hosted live demo + UK-positioning write-up + demo video → portfolio is *shareable*, not just readable
13. API hardening pass (validation, rate limiting, health checks) → demo survives public traffic without babysitting
14. Statistical rigor pass (CIs, multi-seed, significance test vs. baseline) → results hold up to scrutiny
15. Docs site (MkDocs) + badges + polish pass → portfolio-ready
