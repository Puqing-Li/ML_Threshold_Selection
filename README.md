# ML Threshold Selection v1.3.0

Machine-learning-assisted selection of the minimum object-volume threshold
(`Vmin`) for XRCT particle analysis. The application trains a classifier from
expert-labelled samples, applies sample-specific scan resolution during feature
construction, and computes 3D mean-fabric, P' and T after filtering.

The released classifier used for the reported results ships in `trained model/`
and carries the current `resolution_aware_v2_per_sample_sqrt5` feature schema.
**Load Last Model** uses it when no locally trained model is present. You can
also train a new model from the supplied five tables or from your own
expert-labelled samples; training writes to `models/` and never overwrites the
released bundle.

## What changed in v1.3.0

- Every training and test sample requires its own measured voxel edge length in
  mm/voxel. There is no `0.03 mm` default or first-sample fallback.
- The model input is continuous voxel count,
  `Volume3d_mm3 / VoxelSize_mm^3`, plus six log-ellipsoid tensor components.
- Expert labels are assigned directly in physical volume:
  `Volume3d_mm3 < ExpertThreshold_mm3`. The expert threshold is not rounded to
  an integer voxel count during labelling.
- Equivalent-ellipsoid semiaxes are calculated as `sqrt(5 * EigenVal)`.
- Invalid volumes, eigenvalues, eigenvectors, and incomplete voxel-size maps
  stop the workflow instead of being replaced by numerical defaults.
- Saved models carry feature schema
  `resolution_aware_v2_per_sample_sqrt5`; legacy global-0.03 bundles are
  rejected by the current loader.

## Repository layout

| Content | Location |
|---|---|
| Five training tables and authoritative input values | `training_data/` |
| Generated model bundles | `models/` |
| Generated tables, reports and figures | `outputs/` |
| Worked input and TomoFab-format examples | `examples/` |
| Scientific definitions | `docs/SCIENTIFIC_METHODS.md` |
| Detailed GUI guide | `docs/user_guide.md` |
| Avizo conversion utilities | `tools/` |
| Tests | `tests/` |

## Installation

Python 3.9 or newer with Tkinter is required.

```bash
git clone https://github.com/Puqing-Li/ML_Threshold_Selection.git
cd ML_Threshold_Selection
python -m pip install -r requirements.txt
python main.py
```

On Windows, double-click `run_app.bat` to install dependencies and launch the
application.

## Train a new model

The supplied values are recorded in `training_data/training_config.csv`.

1. Click **1. Load Training Data** and select all five
   `training_data/total<SampleID>.xlsx` files.
2. Click **2. Input Expert Thresholds** and enter, in mm3:

   ```text
   AKAN20:0.0039
   ANA16937:0.0008
   HL19335:0.0010
   LE03:0.0010
   LE19:0.0018
   ```

3. Click **3. Input Voxel Sizes** and enter the measured values in mm/voxel:

   ```text
   AKAN20     0.030
   ANA16937  0.040
   HL19335   0.035
   LE03      0.030
   LE19      0.035
   ```

4. Optionally run **4. Feature Analysis**.
5. Click **5. Train Model**. The new classifier and fitted scaler are saved in
   `models/` only after training succeeds.

The displayed training AUC and accuracy are fit diagnostics. They are not an
independent validation of scalar-threshold recovery.

The active workflow is supervised: expert-selected physical-volume thresholds
generate deterministic grain-level pseudo-labels, and LightGBM is fitted to
those labels. No unlabeled observations enter model fitting. The resulting
scores therefore measure agreement with the configured pseudo-label rule, not
independent identification of physical artifacts.

## Apply the trained model

1. In a new session, click **Load Last Model**.
2. Load one or more test tables with **6a** or **6b**.
3. Enter the measured voxel size for every test sample. No value is inferred.
4. Click **7. Predict Analysis**.
5. Use **Mean Fabric**, **Fabric Boxplots**, and **8. Export / Reports** as
   required.

The loose threshold is derived from the cumulative model-score curve. The
strict threshold is one voxel above the largest object whose predicted
artifact probability exceeds the configured tolerance, so every flagged
object is excluded by a `Volume >= threshold` filter.

## P' definition

The implementation follows the manuscript definition. For positive fabric-axis
magnitudes `V1`, `V2`, and `V3`:

```text
f1 = ln(V1), f2 = ln(V2), f3 = ln(V3)
f_mean = (f1 + f2 + f3) / 3
P' = exp(sqrt(2 * [(f1-f_mean)^2 + (f2-f_mean)^2 + (f3-f_mean)^2]))
```

The code uses the mean of the three logarithms, not the logarithm of their
arithmetic mean. Mean-fabric and bootstrap calculations call the same tested
definition.

## Input columns

Each XLSX or CSV table must contain:

- `Volume3d (mm^3) `, including the trailing space used by the Avizo export;
- `EigenVal1`, `EigenVal2`, `EigenVal3`;
- `EigenVec1X` through `EigenVec3Z`.

`tools/BatchFile.py` prepares app-format tables from raw Avizo Label Analysis
exports. It removes rows with missing, non-finite, zero, or negative
eigenvalues and preserves complete sample identifiers. `tools/To_tomofab.py`
creates TomoFab-compatible tables.

## Pseudo-label ranking evaluation

After reviewing the intended endpoint, run the object-level script with:

```bash
python cross_validation.py \
  --data training_data \
  --config training_data/training_config.csv \
  --out outputs/S3_validation
```

The script uses the same per-sample voxel-size feature implementation as the
GUI. Its object-level AUC describes ranking of the configured expert-derived
labels; it does not independently prove recovery of each historical scalar
`Vmin`.

## References

- Brandon, M.T., 1995. *Journal of Structural Geology* 17, 1375-1385.
  https://doi.org/10.1016/0191-8141(95)00032-9
- Jelinek, V., 1981. *Tectonophysics* 79, T63-T67.
  https://doi.org/10.1016/0040-1951(81)90110-4
- Petri, B., Almqvist, B.S.G., Pistone, M., 2020. *Computers & Geosciences*
  138, 104444. https://doi.org/10.1016/j.cageo.2020.104444

## License

MIT. See `LICENSE`.
