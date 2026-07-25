# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.3.0] - 2026-07-25

### Added
- Audited five-sample retraining tables and `training_config.csv`
- Explicit v1.3 feature-schema marker in saved model bundles
- Released classifier in `trained model/`, carrying the
  `resolution_aware_v2_per_sample_sqrt5` schema, so the reported thresholds can
  be reproduced without retraining
- Analysis scripts that produce the reported figures and validation tables:
  `generate_main_figures.py`, `generate_fig1_stereonets.py`,
  `axis_locking_validation.py`, `loso_threshold_validation.py`,
  `generate_cross_sample_threshold_audit.py`, `threshold_sensitivity_analysis.py`,
  `training_axis_locking_audit.py` and `rebuild_revision_results.py`. They were
  named in the documentation but not tracked, and two tracked test modules
  imported them, so the suite did not collect on a clean clone
- Per-grain tables for the four additional samples of Fig 5 in `examples/`
  (12RH26, BG02-4B, BG04-44B, CC10-18), with their voxel sizes and thresholds
- **0. Prepare Raw Data** in the main window, which opens `tools/BatchFile.py`
  in its own process
- Regression tests for per-sample voxel sizes, physical-volume labels,
  equivalent-ellipsoid semiaxes, P', strict thresholds, model persistence, the
  prefilter conditions, load-time validation, single-colour plotting, and the
  application attributes each module reads

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
- `rebuild_revision_results.py` loads the released bundle by default and returns
  the reported LE01 thresholds, 50 and 154 voxels. Refitting is available through
  `--retrain` and returns 204 voxels for the strict threshold, because that
  threshold is the largest object whose artifact probability still exceeds 0.01
  and therefore a maximum over the fitted model's low-probability tail
- `cross_validation.py` applies the balanced class weighting that
  `training_pipeline.py` uses, so the evaluation and the training pipeline fit
  the same classifier. Leave-one-sample-out AUC is 0.906 to 0.996 and the pooled
  five-fold value is 0.989; the reported range is unchanged
- Probability-versus-volume points are drawn in one colour, with no colour bar,
  matching Fig 3 of the article. Shading them by artifact probability repeated
  the vertical axis
- The documentation states that no classifier ships only where that is true:
  models trained locally go to `models/`, which is empty in a fresh clone

### Fixed
- The prefilter in `tools/BatchFile.py` also removes objects with
  `Anisotropy == 1`, which marks a vanishing shortest axis. Those objects pass
  the positivity test, their smallest eigenvalue underflowing to about 1e-25
  rather than to exactly zero, but the fabric calculation takes the logarithm of
  each eigenvalue. Without this condition the prefilter did not reproduce the
  deposited per-grain tables
- Loading validates eigenvalues, so a raw Avizo export is refused when it is
  read rather than accepted and then failing inside the feature builder several
  steps later
- `ui_visualization.py` and `analysis_pipeline.py` read `app.voxel_sizes`, which
  is never assigned; the attribute is `app.training_voxel_sizes`. In the
  prediction visualization the resulting `AttributeError` left an empty window,
  because the window was created before the line that failed
- Failures in the prediction visualization are reported in the log instead of
  going to the Tk callback handler
- The released bundle manifest no longer carries the training machine's absolute
  paths
- Tracked reference outputs in `outputs/` are regenerated at 50 and 154 voxels
  with the corrected P'

### Removed
- Unsupported claims of expert-free or independently validated thresholding
- `outputs/Prob_vs_Vol_example.*`, which predates this release and can only be
  exported from the interface
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
