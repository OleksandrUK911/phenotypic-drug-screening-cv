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
10. Docs/polish pass → portfolio-ready
