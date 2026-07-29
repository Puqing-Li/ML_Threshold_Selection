# Quick Start

This project ships the released classifier in `data/released_model/` together
with the training data in `data/training/`. A first session can either load that
model directly through **Load Last Model** or start from training.

## Windows

1. Double-click `run_app.bat`.
2. Select the five XLSX files in `data/training/` with names beginning `total`.
3. Enter the expert thresholds from `data/training/training_config.csv` in Step
   2. Units are mm3.
4. Enter each sample's measured `VoxelSize_mm` in Step 3. Leave no row blank.
5. Click **5. Train Model**.

The application saves the resulting classifier in `models/`. A later session
can use **Load Last Model**.

## Training values supplied with the project

| Sample | Expert threshold (mm3) | Voxel size (mm/voxel) |
|---|---:|---:|
| AKAN20 | 0.0039 | 0.030 |
| ANA16937 | 0.0008 | 0.040 |
| HL19335 | 0.0010 | 0.035 |
| LE03 | 0.0010 | 0.030 |
| LE19 | 0.0018 | 0.035 |

## Analyse a new sample

1. Load the trained model.
2. Load the test XLSX/CSV table.
3. Enter that scan's measured voxel size in mm/voxel.
4. Run **Predict Analysis**.
5. Export the loose/strict tables and calculate Mean Fabric or bootstrap
   boxplots as needed.

The program will stop if a required voxel size, expert threshold, physical
volume, eigenvalue, or eigenvector is missing or invalid. It does not insert a
default resolution or replace invalid geometry with a small number.
