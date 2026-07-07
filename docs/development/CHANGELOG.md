# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive README with scientific methodology
- Contributing guidelines
- Project badges and metadata

### Changed
- Updated project URLs to correct GitHub repository
- Improved documentation structure

## [1.2.0] - 2026-07-02

### Added
- `cross_validation.py`: stand-alone leave-one-sample-out and five-fold
  cross-validation of the artifact classifier, reproducing the validation
  reported in the PLOS ONE article (S3 Fig)
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
  data-preparation pipeline, and reproducible-validation instructions

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
- Semi-supervised learning with expert thresholds
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
- **Dual Thresholds**: Automatic detection of inflection point and zero-artifact thresholds
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
