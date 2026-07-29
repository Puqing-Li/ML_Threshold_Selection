# Analysis scripts

Command-line scripts that reproduce the manuscript figures and validation
results. The graphical application is `main.py` in the repository root; nothing
here is needed to run it.

Run them from the repository root, not from this directory, because their
default paths are given relative to the root:

```bash
python scripts/analysis/rebuild_revision_results.py --output outputs/revision_rebuild
```

| Script | Produces |
|---|---|
| `analysis/rebuild_revision_results.py` | The reported LE01 thresholds from the released model. `--retrain` refits instead. |
| `analysis/cross_validation.py` | Leave-one-sample-out and pooled ranking evaluation. |
| `analysis/generate_main_figures.py` | Main-text figures. |
| `analysis/generate_fig1_stereonets.py` | Fig 1 orientation-density stereonets. |
| `analysis/axis_locking_validation.py` | Axis-locking percentages for a test sample. |
| `data_preparation/BatchFile.py` | Prepared app-format tables from raw Avizo exports. |
| `data_preparation/To_tomofab.py` | TomoFab-format tables from prepared inputs. |

Data inputs come from `data/training/` and `data/examples/`; the released
classifier comes from `data/released_model/`. Outputs are written to
the runtime-created `outputs/` folder.

Use `scripts/requirements-reproducibility.txt` with Python 3.13.13 to regenerate
the reported cross-validation values shown in S3 Fig. The root
`requirements.txt` remains the general application environment.
