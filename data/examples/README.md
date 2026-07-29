# Examples and reference data

## `expert_thresholds.csv`: expert reference thresholds

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
`data/training/` folder; the voxel sizes and expert thresholds are recorded in
`data/training/training_config.csv`.

## `Quantity_LE01.xlsx`: worked-example input

Per-grain morphometric table for sample LE01 (lherzolite, Mount Leura), the
worked example of the article (Figs 1, 3, 4). This is a legacy table produced
from the raw Avizo Label-Analysis export with an earlier
`scripts/data_preparation/BatchFile.py`, which removed rows with a zero
eigenvalue or exported `Anisotropy == 1` before the retained 4,991-object table
was saved. The raw prefilter counts are not available in this repository. Load
it in the GUI via
**6a. Load Single Test Data** to reproduce the LE01 analysis; the corresponding
outputs (`LE01_Loose_MeanFabric.txt`, `LE01_Strict_MeanFabric.txt`) are in
`reference_outputs/`. New runs write their results to the local, Git-ignored
`outputs/` folder.

## `Quantity_12RH26.xlsx`, `Quantity_BG02-4B.xlsx`, `Quantity_BG04-44B.xlsx`, `Quantity_CC10.xlsx`

Per-grain morphometric tables for the four additional samples of Fig 5, in the
same format as `Quantity_LE01.xlsx`. They allow the Fig 5 sweeps and the
corresponding S4 Table rows to be recomputed. Each carries its own voxel size and
its own machine-learning thresholds:

| Sample | Locality | Voxel (mm) | Loose (vox) | Strict (vox) |
|---|---|---:|---:|---:|
| 12RH26 | Red Hills, New Zealand | 0.041 | 40 | 144 |
| BG02-4B | New Caledonia | 0.039 | 39 | 145 |
| BG04-44B | New Caledonia | 0.041 | 14 | 144 |
| CC10 | North Anatolian Fault | 0.074 | 14 | 143 |

Apply a threshold as a volume: `Vmin_mm3 = Vmin_vox * voxel_mm ** 3`. Use the
voxel count rather than the rounded millimetre value, because rounding the
threshold changes which objects are retained.

Enter each sample's own voxel size. Using another sample's value rescales every
feature and silently produces the wrong threshold.

These tables, like `Quantity_LE01.xlsx`, are the prefiltered inputs used for the
article, produced from the raw Avizo Label-Analysis exports with
`scripts/data_preparation/BatchFile.py`. A raw export is not interchangeable
with them, and loading one stops the run with
`EigenVal1-3 must be finite and strictly positive`. That message means a raw
export was loaded, not that the analysis failed.

The prefilter removes two kinds of degenerate object:

| Sample | Raw | Invalid eigenvalue | `Anisotropy == 1` | Retained |
|---|---:|---:|---:|---:|
| 12RH26 | 39232 | 6198 | 1138 | 31896 |
| BG02-4B | 26037 | 14367 | 1501 | 10169 |
| BG04-44B | 25853 | 9498 | 1835 | 14520 |
| CC10 | 21353 | 8206 | 1827 | 11320 |

The first column of removals is a missing, non-finite, zero or negative
eigenvalue. The second is `Anisotropy == 1`, which Avizo reports when the
shortest principal axis vanishes relative to the longest. Those objects pass the
positivity test, because their smallest eigenvalue underflows to a positive value
near 1e-25 rather than to exactly zero, but the fabric calculation takes the
logarithm of each eigenvalue, so such a value would enter the log-Euclidean mean
as roughly -57 and dominate it.

The shipped tables were checked to contain no degenerate object under these
two conditions.

These four samples were provided by collaborators; see the article's
acknowledgements and sample-provenance statement.

## `TT_totalLE19.xls`: example TomoFab input

This is a file-format demonstration generated from the LE19 training table;
LE01 remains the manuscript worked example. The file shows exactly what a
TomoFab-ready table looks like (protocol Step 51.2,
Option B): tab-separated, with the TomoFab header schema (`Number`,
`Component`, `Unique#`, `Volume (mm^3)`, `PEllipsoid ...`). Produced from
training sample LE19 with `scripts/data_preparation/To_tomofab.py`. The converter writes
equivalent-ellipsoid semiaxes as `sqrt(5 * EigenVal)`; the `PEllipsoid Rad`
columns are lengths in mm, not raw covariance eigenvalues. TomoFab itself is a
separate, third-party MATLAB code (Petri et al. 2020):
https://github.com/benpetri/tomofab
