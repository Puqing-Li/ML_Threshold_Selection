# User Guide

This guide describes the v1.3.1 resolution-aware GUI workflow.

## Install and launch

On Windows, double-click `run_app.bat`. On macOS, use `run_app.command`. The
application can also be started from the repository root with:

```bash
python -m pip install -r requirements.txt
python main.py
```

The released classifier ships in `released_model/`. Use **Load Last Model** to
load it, or train your own model, before loading test data.

## Required object-table columns

| Column | Meaning | Required units |
|---|---|---|
| `Volume3d (mm^3) ` | Object volume | mm^3 |
| `EigenVal1-3` | Coordinate-covariance eigenvalues | mm^2 when calculated in calibrated physical coordinates |
| `EigenVec1X` to `EigenVec3Z` | Principal-axis direction cosines | unit vectors |

The trailing space in `Volume3d (mm^3) ` is retained for compatibility with
the Avizo export. `tools/BatchFile.py` prepares app-format tables and records
invalid-row exclusions.

Do not replace missing values with zero. The active workflow requires finite,
strictly positive volumes and eigenvalues, finite direction cosines, and a
full-rank principal-axis basis.

## Train a model

Start at step 0 only when working from a raw Avizo Label-Analysis export. The
tables shipped with the repository are already prepared.

0. Click **0. Prepare Raw Data** to open the preparation tool
   (`tools/BatchFile.py`) in its own window. It removes objects whose fitted
   ellipsoid is degenerate and writes an app-format table. Loading a raw export
   at step 1 or 6 instead stops with
   `EigenVal1-3 must be finite and strictly positive`.
1. Click **1. Load Training Data** and select every training table.
2. Click **2. Input Expert Thresholds** and enter one physical-volume threshold
   in mm^3 for every selected sample.
3. Click **3. Input Voxel Sizes** and enter the measured isotropic voxel edge
   length in mm/voxel for every selected sample.
4. Optionally click **4. Feature Analysis** to inspect the seven calculated
   features.
5. Click **5. Train Model**.

Training stops if a sample lacks either required input. No default voxel size,
first-sample fallback, or integer rounding of the expert threshold is used.

The saved model bundle in `models/` contains the classifier, fitted scaler,
feature-schema marker, input mapping, and provenance metadata. **Load Last
Model** accepts only the current resolution-aware schema.

The displayed training AUC and accuracy measure fit to expert-derived
pseudo-labels. They are not independent validation of physical artifacts or
proof that the model reproduces a historical scalar threshold.

## Analyse a sample

1. Train a model or click **Load Last Model**.
2. Click **6a. Load Single Test Data** for one sample or **6b. Load Multi Test
   Data** for a batch.
3. Enter the measured voxel edge length for every test sample. The value is
   never inferred from a filename or another sample.
4. Click **7. Predict Analysis**.
5. Inspect the loose and strict operating points and the number of retained
   objects.
6. Use **Mean Fabric**, **Fabric Boxplots**, and **8. Export / Reports** as
   required.

The loose threshold is an inflection-rule candidate from the cumulative model
score curve. The strict threshold excludes every object whose score exceeds
the configured tolerance. These are sensitivity bounds that still require
geological and image-quality review; neither is a physical ground truth.

## Multi-sample input

Sample IDs are derived from filenames and preserved in the combined table.
Before analysis, verify that each displayed ID maps to the correct scan and
voxel size. The workflow stops if any selected ID lacks a valid mapping.

## Outputs

Generated files are written under `outputs/` and may include:

- loose and strict filtered object tables;
- voxel and physical-volume threshold reports;
- mean-fabric tensors and principal directions;
- `P'` and `T` summaries; and
- bootstrap figures and tables.

Generated results are ignored by Git. Freeze only the outputs produced by the
same model, code commit, voxel-size map, and probability tolerance used in the
reported analysis.

## Pseudo-label ranking evaluation

The supplied five-sample evaluation can be rerun with:

```bash
python scripts/cross_validation.py \
  --data training_data \
  --config training_data/training_config.csv \
  --out outputs/S3_validation
```

The reported AUC values quantify ranking of the configured expert-derived
pseudo-labels. They do not independently validate physical artifact identity
or scalar-threshold recovery.

## Troubleshooting

### Training reports a missing threshold or voxel size

Enter one valid value for every listed sample. Blank values are not allowed.

### Prediction reports a schema mismatch

The selected bundle was produced by the legacy global-0.03 workflow or an
incompatible development schema. Retrain with the current code; do not bypass
the schema check.

### Eigenvalue or principal-axis validation fails

Return to the source Avizo export and verify the covariance coordinate basis,
units, and row completeness. Do not replace zero or negative eigenvalues with
small constants.

### A stereonet remains scan-axis concentrated after filtering

Do not assume that every concentration is an artifact. Compare the unfiltered,
loose, strict, and expert-reviewed results with the segmented volume and the
sample's expected geological fabric.

## Support

- Issues: https://github.com/Puqing-Li/ML_Threshold_Selection/issues
- Discussions: https://github.com/Puqing-Li/ML_Threshold_Selection/discussions
