# Pre-trained Model and Training Datasets

This folder contains the pre-trained classifier and the five training datasets
provided with the PLOS ONE GUI tool, so users can perform strict/loose
threshold prediction instantly without preparing training data.

## Contents

1. **Pre-trained classifier** — the resolution-aware 7D LightGBM model
   (VoxelCount + six log-ellipsoid tensor components), loaded via the
   **Load Last Model** button. Shipped in two forms:
   - **`last_time_model_portable/`** — a version-independent bundle (LightGBM
     native text + CSV + NumPy `.npy` + JSON) that loads across pandas / NumPy /
     scikit-learn versions. Loaded preferentially.
   - **`last_time_model.pkl`** — the original pickled bundle, kept as a
     same-environment fallback.

2. **Five training datasets (`total<Sample>.xlsx`)**
   - `totalAKAN20.xlsx`, `totalANA16937.xlsx`, `totalHL19335.xlsx`,
     `totalLE03.xlsx`, `totalLE19.xlsx`
   - Per-grain morphometric tables (35,745 segmented objects in total) used to
     train the shipped classifier. Produced from raw Avizo Label-Analysis
     exports with `tools/BatchFile.py` (object-name row removed,
     zero-eigenvalue and Anisotropy = 1 objects filtered).
   - Sample provenance, voxel sizes, grain counts, and the independently
     determined expert reference thresholds are documented in
     [`examples/README.md`](../examples/README.md) and
     [`examples/expert_thresholds.csv`](../examples/expert_thresholds.csv).

3. **`voxel_sizes.xlsx`**
   - Metadata table mapping each training sample to its scanning resolution
     (voxel size in mm). Required by the resolution-aware feature
     transformation to normalize features into physical volumes.

## Reproducing the reported validation

The leave-one-sample-out and five-fold cross-validation reported in the
article (AUC ≈ 0.96–0.99; S3 Fig) can be reproduced directly from the files
in this folder:

```bash
python cross_validation.py --data "trained model" --config examples/expert_thresholds.csv
```
