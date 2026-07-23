# Legacy v1.2 Archive

This directory is retained only to preserve the repository's published
history. Its classifier was produced by the legacy global-`0.03 mm` training
workflow and is not loaded by v1.3.0.

Do not use `last_time_model.pkl` or `last_time_model_portable/` with the v1.3.0
resolution-aware workflow. Train a new model from the audited tables in
`training_data/`; the application will save the schema-marked result in
`models/`.

The historical spreadsheets remain here for traceability. The active copies
and their authoritative voxel sizes and expert thresholds are documented in
`training_data/README.md` and `training_data/training_config.csv`.
