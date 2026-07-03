# ML Threshold Selection: Adaptive Volume Threshold Selection Tool for XRCT Analysis

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the official, publication-ready software implementation corresponding to our PLOS ONE manuscript. It is a machine learning-driven toolkit designed for adaptive volume threshold selection in High-Resolution X-ray Computed Tomography (XRCT) particle analysis. The tool provides researchers with an automated, algorithmically robust method to determine statistically optimal volume thresholds for artifact removal in large crystalline or vesicular datasets.

**Companion article**: *A standardized XRCT protocol for quantitative 3D shape preferred orientation analysis in rocks* (PLOS ONE, in revision). Step-by-step lab protocol: [protocols.io](https://dx.doi.org/10.17504/protocols.io.n92ld16znl5b/v2). Code and data archived at Zenodo: [10.5281/zenodo.19065901](https://doi.org/10.5281/zenodo.19065901).

🟢 **No coding experience?** Follow the **[Quick Start guide](QUICKSTART.md)** — on Windows, just double-click **`run_app.bat`**.

## 📌 Where to find what

| Looking for | Location |
|---|---|
| **Quick start for non-programmers** (double-click launchers) | [`QUICKSTART.md`](QUICKSTART.md) · `run_app.bat` · `tools/run_BatchFile.bat` · `tools/run_To_tomofab.bat` |
| **Scientific background** — the 7 resolution-aware features, log-ellipsoid tensor math, P′/T fabric parameters | [`docs/SCIENTIFIC_METHODS.md`](docs/SCIENTIFIC_METHODS.md) |
| **User guide** — GUI walkthrough, model & feature details | [`docs/user_guide.md`](docs/user_guide.md) · [`docs/USER_GUIDE_MODEL_AND_FEATURES_EN.md`](docs/USER_GUIDE_MODEL_AND_FEATURES_EN.md) |
| **Step-by-step lab protocol** (acquisition → segmentation → fabric) | [protocols.io](https://dx.doi.org/10.17504/protocols.io.n92ld16znl5b/v2) |
| **Five training datasets** (per-grain tables) | [`trained model/`](trained%20model) — `totalAKAN20.xlsx` … `totalLE19.xlsx` |
| **Expert reference thresholds & voxel sizes** | [`examples/expert_thresholds.csv`](examples/expert_thresholds.csv) · `trained model/voxel_sizes.xlsx` |
| **Pre-trained classifier** | `trained model/last_time_model.pkl` |
| **Reproduce the reported validation** (LOSO + five-fold, AUC ≈ 0.96–0.99) | [`cross_validation.py`](cross_validation.py) |
| **Convert a raw Avizo export** to app / TomoFab input | [`tools/BatchFile.py`](tools/BatchFile.py) · [`tools/To_tomofab.py`](tools/To_tomofab.py) |
| **Worked example** (input & outputs, sample LE01) | [`examples/Quantity_LE01.xlsx`](examples/Quantity_LE01.xlsx) · `outputs/LE01_*_MeanFabric.txt` |

## 🚀 Key Features

- **Pre-trained Generalization Model**: Includes a robust LightGBM core pre-trained on high-fidelity multi-volcano datasets. No manual model training required.
- **Advanced Feature Engineering**: Employs resolution-aware 7D log-ellipsoid tensor mappings to standardize physical particle properties across varying CT voxel resolutions.
- **Dual Threshold Output**: Automatically outputs both a Loose Threshold (inflection point geometry) and a Strict Threshold (noise-removal boundaries).
- **Automated Fabric Boxplots**: In-app generation of standardized, publication-quality Matplotlib figures for comparative structural geology parameters ($P'$ and $T$).
- **Export Control**: Supports `.xlsx` spreadsheet exports for raw prediction tracking, and both `.png` and `.svg` high-resolution graphic exports designed strictly to PLOS ONE figure standards.

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- Windows/macOS/Linux compatible
- A standard desktop or laptop workstation (16–32 GB RAM; **no GPU required**); threshold prediction plus fabric calculation take under one minute per sample

### Setup & Cross-Platform Installation

This software is built entirely on standard Python libraries, ensuring 100% native compatibility across **Windows, macOS, and Linux**.

To run the program, users simply need to clone the repository and install the standard dependencies:

```bash
# 1. Clone the repository
git clone https://github.com/Puqing-Li/ML_Threshold_Selection.git

# 2. Enter the directory
cd ML_Threshold_Selection

# 3. Install required computational libraries
pip install -r requirements.txt
```
*(Note for Linux users: ensure your python distribution includes the `python3-tk` package for the GUI to render).*

## 🚀 Quick Start (GUI Interface)

The primary interaction mode is our self-contained `Tkinter` GUI, ensuring a reproducible environment without requiring coding expertise.

### 1. Launch the Application
```bash
python main.py
```
*(Windows users can simply double-click **`run_app.bat`** instead — it checks Python, installs the libraries, and launches the app; see [QUICKSTART.md](QUICKSTART.md).)*

### 2. Follow the 4-Step Analysis Flow
1. **Load Pre-Trained Weights**: Instead of training a blank model, click **`Load Last Model`**. This instantly pulls the latest validated geological machine learning weights located in the `trained model` folder.
2. **Load Test Data**: Click **`6a. Load Single Test Data`** or **`6b. Load Multi Test Data`** to input your target samples (must be standard CT particle data spreadsheets). You will be prompted to confirm their scanning resolution / voxel size.
3. **Run Predictions**: Click **`7. Predict Analysis`**. The AI will scan your dataset, mapping log-ellipsoid dimensions against its learned artifact signatures, and output statistically rigorous threshold boundaries.
4. **Generate Analytics & Publishable Plots**:
   - Click **`Prediction Visualization`** to explore interactive and exportable distribution graphs of artifact probabilities vs. minimum volumes.
   - Click **`Fabric Boxplots`** and **`Mean Fabric`** to compute tensor statistics, render publication-standard distributions of the $P'$ and $T$ parameters, and output results to `.txt` files compatible with Avizo. Save everything with **`8. Export / Reports`**.

*All generated graphics are rendered in publication quality (300 DPI, white background, removed spines), explicitly calibrated for rigorous journal submission.*

## 📊 Data Requirements

Your input particle data (XLSX or CSV) must contain these essential standard stereological columns:

| Column Name | Description | Units |
|-------------|-------------|-------|
| `Volume3d (mm^3) ` | Sub-particle volume parameter | mm³ |
| `EigenVal1`, `EigenVal2`, `EigenVal3` | Ellipsoid tensor eigenvalues | Dimensionless |
| `EigenVec1X`-`Vec1Z` | Primary structural axis orientation | Unit Vector |
| `EigenVec2X`-`Vec2Z` | Secondary structural axis orientation | Unit Vector |
| `EigenVec3X`-`Vec3Z` | Tertiary structural axis orientation | Unit Vector |

## 🏗️ Project Architecture

```
ML_Threshold_Selection/
├── main.py                     # Primary GUI application entry point
├── cross_validation.py         # Stand-alone cross-validation of the artifact classifier
├── src/ml_threshold_selection/ # Core algorithmic backbone
├── tools/                      # Avizo-export preparation scripts (BatchFile.py, To_tomofab.py)
├── trained model/              # Pre-trained models, voxel calibrations & the five training datasets
├── examples/                   # Demonstration files (Quantity_LE01.xlsx) & expert_thresholds.csv
├── outputs/                    # Default destination for generated plots and spreadsheets
└── requirements.txt            # Environment specifiers
```

## 🔄 Data Preparation: from a raw Avizo export to app / TomoFab input

Raw Avizo *Label-Analysis* CSV exports are not used directly: the first row holds the
Avizo object name, most headers carry a trailing space (e.g. `Volume3d (mm^3) `), and
the column order does not match TomoFab's schema. Two scripts in `tools/` automate the
conversion (both open a simple file-picker dialog):

1. **`tools/BatchFile.py`** — cleaning + app format. Reads the raw Avizo CSV (skipping
   the object-name row), removes grains with zero eigenvalues or `Anisotropy = 1`, and
   writes `total<Sample>.xlsx` (all columns, the app's training/test input — the files
   in `trained model/` were produced this way) and, with an optional volume threshold,
   `Quantity_<Sample>.xlsx`. Residual zeros in derived eigen tables are replaced with
   1e-8 to prevent logarithmic singularities.
2. **`tools/To_tomofab.py`** — TomoFab format. Converts the cleaned spreadsheet to a
   tab-separated `TT_<Sample>.xls` with the TomoFab headers (`Number`, `Component`,
   `Unique#`, `Volume (mm^3)`, `PEllipsoid ...`), avoiding any manual header copying.

Pipeline: Avizo export → `BatchFile.py` → `total*.xlsx` → the GUI app (threshold
selection + fabric), and → `To_tomofab.py` → `TT_*.xls` → TomoFab (expert reference
threshold for the training set).

**Note**: TomoFab is a separate, third-party open-source MATLAB code (Petri,
Almqvist & Pistone 2020, *Computers & Geosciences*) — it is **not** part of this
repository; download it from https://github.com/benpetri/tomofab. An example
TomoFab-ready file (`examples/TT_totalLE19.xls`) is provided for comparison.

## ✅ Reproducible validation

`cross_validation.py` reproduces the classifier validation reported in the manuscript
(leave-one-sample-out and pooled five-fold AUC ≈ 0.96–0.99) from the five training
datasets shipped in `trained model/`:

```bash
python cross_validation.py --data "trained model" --config examples/expert_thresholds.csv
```

## 📚 Key References
- **Foundational 6D Tensors and Software**: The dimensional basis of this algorithm's tensor structure is an extension of the ellipsoidal data framework and software tools developed by **Joshua R. Davis**. Our 7D resolution-aware spatial tensor incorporates this mathematics as its core building block. 
  - *Reference: Davis, J.R., Roberts, N.M., Garibaldi, N., Chatzaras, V., Lusk, A.D., Titus, S.J. A unified framework for statistics of anisotropy of magnetic susceptibility and other ellipsoidal data. AGU fall meeting (2023).*
  - *Reference: Chatzaras, V. et al. (2021). See Section 3.2 for the methodological foundation of ellipsoidal statistics.*
  - *Software Website*: We additionally credit the **geologyGeometry** software package maintained by Joshua R. Davis and collaborators for ellipsoidal statistics. Available at: [http://www.joshuadavis.us/software/index.html](http://www.joshuadavis.us/software/index.html).
- **Foliation & Lineation**: Based on the generalized orientation matrix algorithms pioneered by **Jelínek, V. (1981)**. *Characterization of the magnetic fabric of rocks. Tectonophysics, 79(1-4).*
- Structural visualization mappings are inspired by methodologies inherent in **TomoFab Software**.

---

**Citation**: If you use this software in your research, please cite:

```bibtex
@software{ml_threshold_selection_2026,
  title={ML Threshold Selection: Machine Learning-Driven Adaptive Threshold Selection for XRCT Particle Analysis},
  author={Puqing Li},
  year={2026},
  version={2.0.0},
  url={https://github.com/Puqing-Li/ML_Threshold_Selection},
  license={MIT},
  keywords={xrct, shape-fabric, particle analysis, machine learning, clustering threshold}
}
```