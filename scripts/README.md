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
| `generate_main_figures.py` | Main-text figures. |
| `generate_fig1_stereonets.py` | Fig 1 orientation-density stereonets. |
| `axis_locking_validation.py` | Axis-locking percentages for a test sample. |

Data inputs come from `training_data/` and `examples/`; the released classifier
comes from `released_model/`. Outputs are written to `outputs/`.

Use `requirements-reproducibility.txt` with Python 3.13.13 to regenerate the
reported cross-validation values shown in S3 Fig. `requirements.txt` remains
the general application environment.
