# Quick Start

## Reproduce the released LE01 example

1. Start the application:
   - Windows: double-click `run_app.bat`.
   - macOS: use `run_app.command`.
   - Terminal: run `python main.py` from the repository root.
2. Click **Load Last Model**. In a fresh clone, this loads the classifier in
   `data/released_model/`.
3. Click **6a. Load Single Test Data** and select
   `data/examples/Quantity_LE01.xlsx`.
4. Enter `0.030` mm/voxel for LE01.
5. Click **7. Predict Analysis**.

The released model returns:

| Candidate | Threshold (voxels) | Retained objects |
|---|---:|---:|
| Loose | 50 | 2,212 |
| Strict | 154 | 1,074 |

Use **Mean Fabric**, **Fabric Boxplots**, and **8. Export / Reports** as needed.
Generated files are written to `outputs/`.

## Analyse another sample

1. Load the released model or a model you trained locally.
2. Load one table with **6a** or a batch with **6b**.
3. Enter the measured voxel edge length for every sample.
4. Click **7. Predict Analysis**.
5. Review the loose and strict candidates against the segmented volume,
   stereonets and expected geological fabric before selecting an operating
   threshold.

No voxel size is inferred from another sample or supplied as a default.

## Optional: train a new model

The supplied training inputs are listed in
`data/training/training_config.csv`.

1. Select the five `total<SampleID>.xlsx` files in `data/training/` with
   **1. Load Training Data**.
2. Enter each sample's `ExpertThreshold_mm3` with
   **2. Input Expert Thresholds**.
3. Enter each sample's `VoxelSize_mm` with **3. Input Voxel Sizes**.
4. Optionally run **4. Feature Analysis**.
5. Click **5. Train Model**.

The new bundle is saved in `models/`; the released classifier is not
overwritten. See [`user_guide.md`](user_guide.md) for input requirements and
interpretation.
