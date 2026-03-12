# ML Threshold Selection: Adaptive Volume Threshold Selection Tool for XRCT Analysis

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the official, publication-ready software implementation corresponding to our PLOS ONE manuscript. It is a machine learning-driven toolkit designed for adaptive volume threshold selection in High-Resolution X-ray Computed Tomography (XRCT) particle analysis. The tool provides researchers with an automated, algorithmically robust method to determine statistically optimal volume thresholds for artifact removal in large crystalline or vesicular datasets.

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

### Setup & Cross-Platform Installation

This software is built entirely on standard Python libraries, ensuring 100% native compatibility across **Windows, macOS, and Linux**.

To run the program, users simply need to clone the repository and install the standard dependencies:

```bash
# 1. Clone the repository
git clone https://github.com/VincentDD1125/ML-Threshold-Selection.git

# 2. Enter the directory
cd ML-Threshold-Selection

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

### 2. Follow the 4-Step Analysis Flow
1. **Load Pre-Trained Weights**: Instead of training a blank model, click **`[Click to Load Last Time Model]`**. This instantly pulls the latest validated geological machine learning weights located in the `trained model` folder.
2. **Load Test Data**: Click **`[Load Multi Test Data]`** or **`[Load Single Test Data]`** to input your target samples (must be standard CT particle data spreadsheets). You will be prompted to confirm their scanning resolution / voxel size.
3. **Run Predictions**: Click **`[Predict Analysis]`**. The AI will scan your dataset, mapping log-ellipsoid dimensions against its learned artifact signatures, and output statistically rigorous threshold boundaries.
4. **Generate Analytics & Publishable Plots**:
   - Click **`[Prediction Visualization]`** to explore interactive and exportable distribution graphs of artifact probabilities vs. minimum volumes.
   - Click **`[Fabric Boxplots]`** and **`[Mean Fabric]`** to compute tensor statistics, render publication-standard distributions of the $P'$ and $T$ parameters, and output results to `.txt` files compatible with Avizo.

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
├── src/ml_threshold_selection/ # Core algorithmic backbone
├── trained model/              # Houses pre-trained standard models & voxel calibrations
├── examples/                   # Baseline demonstration files (e.g. Quantity_LE01.xlsx)
├── outputs/                    # Default destination for generated plots and spreadsheets
└── requirements.txt            # Environment specifiers
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
  author={Vincent DD},
  year={2026},
  version={2.0.0},
  url={https://github.com/VincentDD1125/ML-Threshold-Selection},
  license={MIT},
  keywords={xrct, shape-fabric, particle analysis, machine learning, clustering threshold}
}
```