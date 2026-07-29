# ML Threshold Selection v1.3.2

ML Threshold Selection is a graphical application for selecting candidate
minimum object-volume thresholds (`Vmin`) in XRCT particle analysis. It learns
from expert-labelled samples, uses the measured scan resolution of each sample,
and calculates 3D mean fabric, P' and T after filtering.

The released classifier used for the accompanying manuscript is included in
`data/released_model/`. Its loose and strict outputs are reproducible candidate
operating points, not independent physical ground truth; they should be
reviewed against the segmented volume, stereonets and expected geological
fabric.

## Quick start

Python 3.8 or newer with Tkinter is required.

```bash
git clone https://github.com/Puqing-Li/ML_Threshold_Selection.git
cd ML_Threshold_Selection
python -m pip install -r requirements.txt
python main.py
```

Windows users can instead double-click `run_app.bat`; macOS users can use
`run_app.command`.

To reproduce the released LE01 example from a fresh clone:

1. Click **Load Last Model**.
2. Click **6a. Load Single Test Data** and select
   `data/examples/Quantity_LE01.xlsx`.
3. Enter the measured voxel size, `0.030` mm/voxel.
4. Click **7. Predict Analysis**.

The released model returns loose and strict candidates of 50 and 154 voxels,
retaining 2,212 and 1,074 objects, respectively. See
[`docs/QUICKSTART.md`](docs/QUICKSTART.md) for the complete no-code workflow.

## Train your own model

The application can also train a classifier from your own expert-labelled
samples. Each training and test sample requires its own measured voxel edge
length; the software does not insert a default or reuse the first sample's
resolution. Locally trained bundles are written to `models/` and do not
overwrite the released classifier.

The five manuscript training tables and their input values are in
`data/training/`. Follow the step-by-step instructions in
[`docs/user_guide.md`](docs/user_guide.md).

## Reproduce reported results

The released-model LE01 analysis can be rebuilt with:

```bash
python scripts/analysis/rebuild_revision_results.py \
  --output outputs/revision_rebuild
```

The object-level ranking evaluation reported in S3 Fig can be regenerated with:

```bash
python -m pip install -r scripts/requirements-reproducibility.txt
python scripts/analysis/cross_validation.py \
  --data data/training \
  --config data/training/training_config.csv \
  --out outputs/S3_validation
```

The AUC values quantify ranking against expert-derived physical-volume
pseudo-labels. They are not an independent validation of artifact identity or
recovery of each historical scalar threshold.

## Repository layout

| Content | Location |
|---|---|
| Application entry point | `main.py` |
| Application source | `src/` |
| Training data, examples and released classifier | `data/` |
| Analysis and data-preparation utilities | `scripts/` |
| User and scientific documentation | `docs/` |
| Locally trained models | `models/` (created at runtime) |
| Generated results | `outputs/` (created at runtime) |

## Documentation

- [Quick start](docs/QUICKSTART.md)
- [GUI user guide](docs/user_guide.md)
- [Scientific methods and definitions](docs/SCIENTIFIC_METHODS.md)
- [Analysis scripts](scripts/README.md)
- [Release history](docs/CHANGELOG.md)

## Citation

Use the citation metadata in [`CITATION.cff`](CITATION.cff). The archived
software record is available at
[doi:10.5281/zenodo.18979422](https://doi.org/10.5281/zenodo.18979422).

## License

MIT. See [`LICENSE`](LICENSE).
