# ML Threshold Selection

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Machine-learning-based selection of the minimum grain-volume threshold (Vmin) for X-ray computed tomography (XRCT) particle analysis, with mean fabric-tensor computation (P′ and T; Jelínek, 1981) and bootstrap uncertainty estimation.

Segmented objects that span only a few voxels have poorly constrained best-fit ellipsoids whose axes align with the scanner coordinate system, producing artificial orientation clustering that masks the true rock fabric. This toolkit classifies such artifacts with a gradient-boosted model (LightGBM) built on seven resolution-aware features — a continuous voxel count plus the six components of the log-ellipsoid tensor — derives objective "loose" and "strict" volume thresholds from the cumulative artifact-rate curve, and computes the mean fabric tensor with bootstrap confidence intervals from the filtered grain population.

- **Companion article**: Li P, Chatzaras V, Foley M, Özaydın S, Rey PF. *A standardized XRCT protocol for quantitative 3D shape preferred orientation analysis in rocks.* PLOS ONE (in revision).
- **Step-by-step protocol**: [protocols.io](https://dx.doi.org/10.17504/protocols.io.n92ld16znl5b/v2)
- **Archived releases**: [doi.org/10.5281/zenodo.18979422](https://doi.org/10.5281/zenodo.18979422)

New to Python? See [QUICKSTART.md](QUICKSTART.md); on Windows, simply double-click `run_app.bat`.

## Repository guide

| Content | Location |
|---|---|
| Method details (features, thresholds, fabric parameters) | [`docs/SCIENTIFIC_METHODS.md`](docs/SCIENTIFIC_METHODS.md) |
| User guide (GUI, command line, Python API) | [`docs/user_guide.md`](docs/user_guide.md) |
| Five training datasets (per-grain tables) | `trained model/total<Sample>.xlsx` |
| Expert reference thresholds and voxel sizes | [`examples/expert_thresholds.csv`](examples/expert_thresholds.csv) · `trained model/voxel_sizes.xlsx` |
| Data provenance (localities, voxel sizes, grain counts) | [`examples/README.md`](examples/README.md) |
| Pre-trained classifier | `trained model/last_time_model.pkl` |
| Cross-validation script (reproduces the reported AUC) | [`cross_validation.py`](cross_validation.py) |
| Avizo-export conversion scripts | [`tools/BatchFile.py`](tools/BatchFile.py) · [`tools/To_tomofab.py`](tools/To_tomofab.py) |
| Worked example (sample LE01) | `examples/Quantity_LE01.xlsx` · `outputs/LE01_*_MeanFabric.txt` |
| Example TomoFab input file | `examples/TT_totalLE19.xls` |

## Installation

Requires Python 3.8 or newer with Tkinter (included in the python.org installers; on conda: `conda install tk`). A standard desktop or laptop workstation is sufficient (16–32 GB RAM, no GPU); threshold prediction and fabric calculation take under one minute per sample.

```bash
git clone https://github.com/Puqing-Li/ML_Threshold_Selection.git
cd ML_Threshold_Selection
pip install -r requirements.txt
python main.py
```

Windows users can double-click `run_app.bat` instead; it checks the environment, installs the dependencies, and launches the application.

## Usage

With the pre-trained model:

1. **Load Last Model** — loads the shipped classifier (`trained model/last_time_model.pkl`).
2. **6a. Load Single Test Data** or **6b. Load Multi Test Data** — select the per-sample grain table(s); you will be asked for each sample's voxel size in mm.
3. **7. Predict Analysis** — computes the objective loose and strict Vmin thresholds.
4. **Mean Fabric** and **Fabric Boxplots** — compute the mean fabric tensor and the P′ and T parameters with 1,000-iteration bootstrap confidence intervals; **8. Export / Reports** writes tables and figures to `outputs/`.

To retrain the classifier on your own samples, follow the numbered buttons 1–5 (protocol Steps 52–55; see `docs/user_guide.md`).

## Input data

Input tables (XLSX or CSV) must contain the following columns, as exported by Avizo Label Analysis:

| Column | Description |
|---|---|
| `Volume3d (mm^3) ` | Object volume (note the trailing space in the Avizo header) |
| `EigenVal1`–`EigenVal3` | Ellipsoid tensor eigenvalues |
| `EigenVec1X`–`EigenVec3Z` | Principal axis orientations (nine components) |

## Data preparation (from a raw Avizo export)

Raw Avizo Label-Analysis exports are not used directly: the first row holds the object name, most headers carry a trailing space, and the column order does not match TomoFab's schema. Two scripts in `tools/` automate the conversion (both open a file-picker dialog; Windows users can double-click the matching `run_*.bat`):

1. `tools/BatchFile.py` — cleaning and app format. Removes the object-name row and degenerate objects (zero eigenvalues or Anisotropy = 1), and writes `total<Sample>.xlsx` (the app input; the files in `trained model/` were produced this way) and, with an optional volume threshold, `Quantity_<Sample>.xlsx`. Residual zeros in derived eigen tables are replaced with 1e-8 to prevent logarithmic singularities.
2. `tools/To_tomofab.py` — TomoFab format. Converts the cleaned table to a tab-separated `TT_<Sample>.xls` with the TomoFab header schema. An example output is provided (`examples/TT_totalLE19.xls`).

TomoFab is a separate, third-party open-source MATLAB code (Petri et al., 2020); it is not part of this repository — download it from https://github.com/benpetri/tomofab.

Pipeline: Avizo export → `BatchFile.py` → `total*.xlsx` → GUI app (threshold selection and fabric), and → `To_tomofab.py` → `TT_*.xls` → TomoFab (expert reference threshold, training samples only).

## Reproducing the reported validation

The classifier validation reported in the article (leave-one-sample-out and pooled five-fold cross-validation, AUC ≈ 0.96–0.99 across the five training samples, 35,745 segmented objects) can be reproduced from the shipped data:

```bash
python cross_validation.py --data "trained model" --config examples/expert_thresholds.csv
```

## How to cite

See [`CITATION.cff`](CITATION.cff), or:

```bibtex
@software{ml_threshold_selection,
  title   = {ML Threshold Selection: Machine Learning-Driven Adaptive Threshold Selection for XRCT Particle Analysis},
  author  = {Li, Puqing},
  year    = {2026},
  version = {1.2},
  url     = {https://github.com/Puqing-Li/ML_Threshold_Selection},
  doi     = {10.5281/zenodo.18979422},
  license = {MIT}
}
```

## References

- Jelínek V (1981). Characterization of the magnetic fabric of rocks. *Tectonophysics* 79, T63–T67.
- Davis JR, Roberts NM, Garibaldi N, Chatzaras V, Lusk AD, Titus SJ (2023). A unified framework for statistics of anisotropy of magnetic susceptibility and other ellipsoidal data. AGU Fall Meeting. Software: [geologyGeometry](http://www.joshuadavis.us/software/index.html).
- Chatzaras V, et al. (2021). Section 3.2 for the 6D log-ellipsoid vectorization on which the feature space builds.
- Petri B, Almqvist BSG, Pistone M (2020). 3D rock fabric analysis using micro-tomography: an introduction to the open-source TomoFab MATLAB code. *Computers & Geosciences* 138, 104444.
- Ke G, et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS* 30.

## License

MIT — see [LICENSE](LICENSE).
