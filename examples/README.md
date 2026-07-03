# Examples and reference data

## `expert_thresholds.csv` — expert reference thresholds (ground truth)

One row per training sample: `SampleID, ExpertThreshold_mm3, VoxelSize_mm`.

The expert threshold (Vmin, in mm³) was determined **independently of the
classifier** by stereonet inspection in TomoFab, following Step 53 of the
protocol: the volume threshold was increased until the aliasing artifacts
(point maxima aligned with the XRCT instrument X/Y/Z axes) disappeared from
the eigenvector stereonets. These values seed the semi-supervised labelling
and serve as the independent benchmark in the cross-validation
(`cross_validation.py`, S3 Fig of the article).

| Sample | Locality (Newer Volcanic Province, Australia) | Voxel size (mm) | Segmented objects | Expert Vmin (mm³) |
|---|---|---|---|---|
| AKAN20 | Mount Anakie | 0.030 | 6,681 | 3.9e-03 |
| ANA16937 | Mount Anakie | 0.040 | 6,120 | 8.0e-04 |
| HL19335 | Hepburn Lagoon | 0.035 | 6,388 | 1.0e-03 |
| LE03 | Mount Leura | 0.030 | 6,582 | 1.0e-03 |
| LE19 | Mount Leura | 0.035 | 9,974 | 1.8e-03 |

Total: 35,745 segmented objects across the five training samples.
The corresponding per-grain tables (`total<Sample>.xlsx`) ship in the
`trained model/` folder; the voxel sizes are also recorded in
`trained model/voxel_sizes.xlsx`.

## `Quantity_LE01.xlsx` — worked-example input

Per-grain morphometric table for sample LE01 (lherzolite, Mount Leura), the
worked example of the article (Figs 1, 3, 4). Produced from the raw Avizo
Label-Analysis export with `tools/BatchFile.py`. Load it in the GUI via
**6. Load Test Data** to reproduce the LE01 analysis; the corresponding
outputs (`LE01_Loose_MeanFabric.txt`, `LE01_Strict_MeanFabric.txt`) are in
the `outputs/` folder.

## `TT_totalLE19.xls` — example TomoFab input

Shows exactly what a TomoFab-ready file looks like (protocol Step 51.2,
Option B): tab-separated, with the TomoFab header schema (`Number`,
`Component`, `Unique#`, `Volume (mm^3)`, `PEllipsoid ...`). Produced from
training sample LE19 with `tools/To_tomofab.py`. TomoFab itself is a separate,
third-party MATLAB code (Petri et al. 2020): https://github.com/benpetri/tomofab
