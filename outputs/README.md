# Outputs

Prediction tables, threshold reports, mean-fabric text files, and bootstrap
figures are written here. Newly generated results are ignored by Git.

The files tracked in this directory are reference outputs for the worked example
`examples/Quantity_LE01.xlsx`, produced with the released v1.3.0 code at the
released thresholds of 50 voxels (loose, 1.350000e-03 mm3) and 154 voxels
(strict, 4.158000e-03 mm3):

- `LE01_Loose_MeanFabric.txt`, `LE01_Strict_MeanFabric.txt`
- `Fabric_Pprime_boxplot.png/.svg`, `Fabric_T_boxplot.png/.svg`

Running the worked example should reproduce these values.
