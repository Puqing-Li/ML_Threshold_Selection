# Examples and reference data

## `expert_thresholds.csv` — expert reference thresholds

One row per training sample: `SampleID, ExpertThreshold_mm3, VoxelSize_mm`.

The configured expert threshold (Vmin, in mm³) predates classifier fitting and
was entered in the original interface as the result of stereonet inspection in
TomoFab: the volume threshold was increased until scan-axis-aligned maxima were
judged to have disappeared. The repository history preserves these values from
the initial March 2026 commit, but it does not contain the original reader,
date, stereonet sequence, or decision log; users should therefore treat them as
historical configured expert inputs rather than independently adjudicated
ground truth. The reported model is supervised using these labels; it is not
semi-supervised learning because no unlabeled observations enter model fitting.
High leave-one-sample-out AUC measures object ranking against the volume rule,
not independent physical artifact identification. Expert thresholds remain in
mm3; voxel equivalents use each row's measured voxel size.

| Sample | Locality (Newer Volcanic Province, Australia) | Voxel size (mm) | Segmented objects | Expert Vmin (mm³) |
|---|---|---|---|---|
| AKAN20 | Mount Anakie | 0.030 | 6,681 | 3.9e-03 |
| ANA16937 | Mount Anakie | 0.040 | 6,120 | 8.0e-04 |
| HL19335 | Hepburn Lagoon | 0.035 | 6,388 | 1.0e-03 |
| LE03 | Mount Leura | 0.030 | 6,582 | 1.0e-03 |
| LE19 | Mount Leura | 0.035 | 9,974 | 1.8e-03 |

Total: 35,745 segmented objects across the five training samples.
The corresponding per-grain tables (`total<Sample>.xlsx`) ship in the
`training_data/` folder; the voxel sizes and expert thresholds are recorded in
`training_data/training_config.csv`.

## `Quantity_LE01.xlsx` — worked-example input

Per-grain morphometric table for sample LE01 (lherzolite, Mount Leura), the
worked example of the article (Figs 1, 3, 4). This is a legacy table produced
from the raw Avizo Label-Analysis export with an earlier `tools/BatchFile.py`,
which removed rows with a zero eigenvalue or exported `Anisotropy == 1` before
the retained 4,991-object table was saved. The raw prefilter counts are not
available in this repository. Load it in the GUI via
**6a. Load Single Test Data** to reproduce the LE01 analysis; the corresponding
outputs (`LE01_Loose_MeanFabric.txt`, `LE01_Strict_MeanFabric.txt`) are in
the `outputs/` folder.

## `TT_totalLE19.xls` — example TomoFab input

Shows exactly what a TomoFab-ready file looks like (protocol Step 51.2,
Option B): tab-separated, with the TomoFab header schema (`Number`,
`Component`, `Unique#`, `Volume (mm^3)`, `PEllipsoid ...`). Produced from
training sample LE19 with `tools/To_tomofab.py`. TomoFab itself is a separate,
third-party MATLAB code (Petri et al. 2020): https://github.com/benpetri/tomofab
