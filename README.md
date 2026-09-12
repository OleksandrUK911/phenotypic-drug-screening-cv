# Phenotypic Drug Screening via Computer Vision

Deep learning pipeline for **phenotypic drug screening**: segmenting cell morphology from high-content microscopy imagery, detecting anomalous (toxic/effective) phenotypes, and classifying mechanism of action (MOA) — the same class of problem tackled by companies like **Recursion Pharmaceuticals** and **Exscientia** (Oxford/Dundee, UK).

A secondary module demonstrates that the same segmentation/anomaly-detection backbone transfers to overhead/aerial imagery, relevant to defense & security imagery-analysis pipelines (e.g. programmes funded via the UK's **DASA**).

## Why this project

High-content screening (HCS) generates thousands of microscopy images per experiment. Classical pipelines (CellProfiler) rely on hand-crafted features; modern approaches use self-supervised deep learning to learn phenotypic embeddings directly from pixels, at scale, with far less manual feature engineering.

## Pipeline

1. **Cell segmentation** — U-Net / Cellpose-style instance segmentation of individual cells from microscopy plates.
2. **Self-supervised representation learning** — SimCLR/DINO pretraining on unlabeled cell images to learn phenotypic embeddings without expensive annotation (mirrors published Recursion/Exscientia approaches).
3. **Batch effect correction** — controlling for plate/day/technical artifacts that confound raw phenotypic signal, a well-known HCS pitfall.
4. **Anomaly detection** — flagging phenotypes that deviate from DMSO/vehicle controls (autoencoder reconstruction error, or distance in embedding space) to identify bioactive compounds.
5. **Mechanism-of-action (MOA) classification** — multi-class classification of compounds by phenotypic signature.
6. **Uncertainty quantification** — Deep Ensembles / MC Dropout, so predictions carry a confidence estimate (important for pharma decision-making, in the spirit of MHRA-grade validation).
7. **Explainability** — Grad-CAM / attention visualization over the morphological features driving each prediction.
8. **Transfer-to-defense demo** (`src/transfer_defense_demo/`) — the same pretrained backbone fine-tuned on a small public aerial-imagery dataset (e.g. DOTA/xView) to show transferability of the segmentation/anomaly-detection approach to remote-sensing/surveillance imagery.
9. **MLOps** — experiment tracking (MLflow/W&B), Dockerized training/inference, FastAPI inference demo.

## Datasets

- Primary: [BBBC021](https://bbbc.broadinstitute.org/BBBC021) / [Cell Painting (JUMP-CP)](https://jump-cellpainting.broadinstitute.org/), Broad Institute.
- Transfer demo: a small public overhead-imagery dataset (DOTA / xView subset).

## Repository structure

```
configs/                 Hydra/OmegaConf experiment configs
scripts/                 CLI entry points (download_data.py, train.py, evaluate.py, infer.py)
src/
  data/                  download, preprocessing, dataset/dataloader, plate-aware splits
  segmentation/          cell instance segmentation (U-Net / Cellpose)
  features/              self-supervised pretraining + embedding extraction + batch-effect correction
  anomaly_detection/     phenotype anomaly scoring vs. controls (incl. Z-factor/SSMD)
  moa_classification/    mechanism-of-action classifier + uncertainty estimation
  active_learning/       simulated compound-prioritization loop
  multimodal/            (stretch) image + molecular graph fusion for MOA
  explainability/        Grad-CAM / attention visualizations
  evaluation/            benchmarking vs. classical CellProfiler pipeline
  transfer_defense_demo/ backbone transfer to aerial/overhead imagery
  mlops/                 training/inference scripts, tracking, API
  utils/                 seeding, logging, shared metrics
tests/                   unit tests mirroring src/ structure
docker/                  containerized training & inference
notebooks/               exploratory analysis
docs/                    write-ups, benchmark comparisons vs. CellProfiler
```

## Status

Project scaffolding in progress — see [TODO.md](TODO.md) for the full milestone checklist.

## Disclaimer

Research/portfolio project using public datasets only. Not intended for clinical, regulatory, or operational defense use.
