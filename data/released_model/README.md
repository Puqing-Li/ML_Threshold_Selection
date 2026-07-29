# Released model

This directory holds the classifier used for the results reported in the
accompanying manuscript.

`last_time_model_portable/` holds the classifier in formats that do not depend
on the library versions or the platform that produced it: the gradient-boosted
model in LightGBM's own text format, the tables as gzipped CSV, and the arrays
as `.npy`. It carries the feature schema `resolution_aware_v2_per_sample_sqrt5`,
which is the schema the v1.3 loader accepts.

**Load Last Model** reads `models/` first and falls back to this directory when
no locally trained model is present, so a fresh clone can reproduce the reported
thresholds without retraining, and training your own model never overwrites this
bundle.

The classifier was trained on the five per-object tables in `data/training/`
using the measured per-sample voxel sizes and expert-selected thresholds in
`data/training/training_config.csv`. Those tables are not duplicated here; the
portable bundle already records the exact training matrix it was fitted on.

The released model was fitted with LightGBM 4.6.0. Its component seeds are
feature fraction 2, bagging 3, data 1, extra 6, drop 4, and objective 5. The GUI
training path makes this recorded profile explicit. Leave-one-sample-out
validation instead uses seed 42 and balanced class weights within each training
fold; it is an evaluation procedure, not the source of this released model.
