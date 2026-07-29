# Training data

This directory contains the five per-object tables used to train a new model.
The classifier trained from these tables for the reported results is released in
`data/released_model/`.

`training_config.csv` is the authoritative input sheet for manual entry in the
GUI. `VoxelSize_mm` is the measured voxel edge length for each scan and
`ExpertThreshold_mm3` is the corresponding expert-selected physical-volume
threshold. The application requires both values for every selected sample and
does not supply a default.

| SampleID | VoxelSize_mm | ExpertThreshold_mm3 |
|---|---:|---:|
| AKAN20 | 0.030 | 0.0039 |
| ANA16937 | 0.040 | 0.0008 |
| HL19335 | 0.035 | 0.0010 |
| LE03 | 0.030 | 0.0010 |
| LE19 | 0.035 | 0.0018 |

The five `total<SampleID>.xlsx` files contain 35,745 segmented objects in
total. `voxel_sizes.xlsx` is retained as the original spreadsheet form of the
scan-resolution metadata; `training_config.csv` combines the values needed by
the current training workflow in one auditable table.
