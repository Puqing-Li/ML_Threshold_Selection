# Released model

This directory holds the classifier used for the results reported in the
accompanying manuscript.

`last_time_model_portable/` is the version-portable bundle and
`last_time_model.pkl` is the same-environment pickle. Both carry the current
feature schema `resolution_aware_v2_per_sample_sqrt5`, which is the schema the
v1.3 loader accepts.

**Load Last Model** reads `models/` first and falls back to this directory when
no locally trained model is present, so a fresh clone can reproduce the reported
thresholds without retraining, and training your own model never overwrites this
bundle.

The classifier was trained on the five per-object tables in `training_data/`
using the measured per-sample voxel sizes and expert-selected thresholds in
`training_data/training_config.csv`. Those tables are not duplicated here; the
portable bundle already records the exact training matrix it was fitted on.
