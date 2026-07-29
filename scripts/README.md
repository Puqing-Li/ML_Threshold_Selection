# Analysis scripts

Command-line scripts that reproduce the manuscript figures and validation
results. The graphical application is `main.py` in the repository root; nothing
here is needed to run it.

Run them from the repository root, not from this directory, because their
default paths are given relative to the root:

```bash
python scripts/rebuild_revision_results.py --output outputs/revision_rebuild
```

| Script | Produces |
|---|---|
| `rebuild_revision_results.py` | The reported LE01 thresholds from the released model. `--retrain` refits instead. |
| `cross_validation.py` | Leave-one-sample-out and pooled ranking evaluation. |
| `loso_threshold_validation.py` | Per-sample thresholds under leave-one-sample-out refitting. |
| `generate_main_figures.py` | Main-text figures. |
| `generate_fig1_stereonets.py` | Fig 1 orientation-density stereonets. |
| `generate_cross_sample_threshold_audit.py` | Cross-sample threshold audit table. |
| `axis_locking_validation.py` | Axis-locking percentages for a test sample. |
| `training_axis_locking_audit.py` | The same audit across the five training samples. |
| `threshold_sensitivity_analysis.py` | Threshold response to the classifier decision cutoff. |

Data inputs come from `training_data/` and `examples/`; the released classifier
comes from `released_model/`. Outputs are written to `outputs/`.

Use `requirements-reproducibility.txt` with Python 3.13.13 to regenerate the
reported cross-validation values shown in S3 Fig. `requirements.txt` remains
the general application environment.
