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

## [0.1.0] - 2024-01-XX

### Added
- Initial release
- Supervised learning approach for threshold selection
- Semi-supervised learning with expert thresholds
- Interactive GUI interface
- Fabric analysis with T and P' parameters
- Resolution-aware feature engineering
- Dual threshold prediction (loose and strict)
- Bootstrap confidence intervals
- Comprehensive test suite
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
- Python 3.7+ support
- NumPy, Pandas, scikit-learn, Matplotlib, Seaborn
- Modular architecture with clear separation of concerns
- Comprehensive error handling and logging
- Cross-platform compatibility (Windows, macOS, Linux)

## [0.0.1] - 2024-01-XX

### Added
- Initial project setup
- Basic project structure
- Core dependencies
- Development environment configuration
