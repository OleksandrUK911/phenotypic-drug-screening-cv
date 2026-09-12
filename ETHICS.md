# Ethics & Responsible Use

- This project uses **public research datasets only** (BBBC021 / JUMP-CP Cell Painting). No patient data, no PII.
- Outputs are **not** validated for clinical or diagnostic use. Nothing here should inform real drug-safety or treatment decisions.
- The `src/transfer_defense_demo/` module is a **methodology demo** showing that the same computer-vision techniques
  (segmentation, anomaly detection) generalize to remote-sensing imagery. It does not use, train on, or claim
  applicability to any operational military/defense system. See that module's README for the full disclaimer.
- Dual-use awareness: phenotypic screening techniques and general-purpose imagery anomaly detection are dual-use-adjacent
  by nature (the same segmentation/detection math applies to cells or to overhead imagery). This repository's scope is
  limited to public-data research and portfolio demonstration — it makes no operational claims in either domain.
