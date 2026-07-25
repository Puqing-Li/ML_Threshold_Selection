# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.3.0] - 2026-07-24

### Added
- Audited five-sample retraining tables and `training_config.csv`
- Explicit v1.3 feature-schema marker in saved model bundles
- Regression tests for per-sample voxel sizes, physical-volume labels,
  equivalent-ellipsoid semiaxes, P', strict thresholds, and model persistence

### Changed
- Training and prediction now require each sample's measured voxel-edge length;
  the `0.03 mm` default and first-sample fallback were removed
- Training and test voxel-size maps are independent
- Expert pseudo-labels use the configured physical-volume threshold directly,
  without integer-voxel ceiling
- Avizo covariance eigenvalues are converted to equivalent-ellipsoid semiaxes
  using `sqrt(5 * EigenVal)`
- P' and T use the manuscript's mean-log semiaxis definition throughout
- Locally trained models are written to `models/`; **Load Last Model** reads
  that directory first and falls back to the released bundle in
  `trained model/`, so training never overwrites the released model
- Bundles carrying a pre-v1.3 feature schema are rejected by the active loader
- Documentation now distinguishes training diagnostics from independent
  physical-artifact validation

### Removed
- Unsupported claims of expert-free or independently validated thresholding
## [1.2.0] - 2026-07-02

### Added
- `cross_validation.py`: stand-alone leave-one-sample-out evaluation of the
  expert-derived below-threshold classifier, plus a clearly labelled pooled
  object-level diagnostic, reproducing the pseudo-label evaluation reported in
  the PLOS ONE article (S3 Fig)
- `tools/BatchFile.py` and `tools/To_tomofab.py`: data-preparation scripts
  converting raw Avizo Label-Analysis exports into the app input (Option A)
  and the TomoFab input (Option B); English UI/logs
- `examples/expert_thresholds.csv`: expert reference thresholds and voxel
  sizes for the five training samples
- `examples/README.md` and expanded `trained model/README.md`: data
  provenance documentation (localities, voxel sizes, grain counts)
- Double-click launchers for non-programmers: `run_app.bat`,
  `tools/run_BatchFile.bat`, `tools/run_To_tomofab.bat`
- `QUICKSTART.md`: plain-language guide requiring no coding experience
- `CITATION.cff`: citation metadata (GitHub "Cite this repository")
- README: companion-article links, "Where to find what" navigation table,
  data-preparation pipeline, and reproducible-evaluation instructions

### Changed
- `pyproject.toml`: author/maintainer and repository URLs corrected to
  Puqing-Li/ML_Threshold_Selection
- README: clone URL and citation corrected to this repository

### Removed
- Stale timestamped development plots from `outputs/` (kept one
  representative pair and the LE01 worked-example outputs)

## [0.1.0] - 2026-03

### Added
- Initial release
- Supervised learning approach for threshold selection
- Expert-threshold pseudo-label training (historically named semi-supervised in the public API)
- Interactive GUI interface
- Fabric analysis with T and P' parameters
- Resolution-aware feature engineering
- Dual threshold prediction (loose and strict)
- Bootstrap confidence intervals
- Unit tests for feature engineering
- Example data and scripts
- Documentation and user guides

### Features
- **Machine Learning Pipeline**: Support for LightGBM, Random Forest, and other scikit-learn classifiers
- **Feature Engineering**: 7D log-ellipsoid tensor features with resolution normalization
- **Dual Thresholds**: Automatic detection of loose and strict operating thresholds
- **Fabric Analysis**: Jelínek (1981) methodology with bootstrap validation
- **GUI Interface**: User-friendly Tkinter application for end-to-end workflow
- **Command Line Tools**: Scripts for batch processing and automation
- **Data Validation**: Built-in data quality checks and error handling
- **Export Capabilities**: Excel reports, plots, and statistical summaries

### Technical Details
- Python 3.8+ support
- NumPy, Pandas, scikit-learn, Matplotlib, Seaborn
- Modular architecture with clear separation of concerns
- Comprehensive error handling and logging
- Cross-platform compatibility (Windows, macOS, Linux)

## [0.0.1] - 2026-03

### Added
- Initial project setup
- Basic project structure
- Core dependencies
- Development environment configuration
