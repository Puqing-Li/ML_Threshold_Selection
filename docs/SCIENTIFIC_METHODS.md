# Scientific Methods

This document defines the analysis implemented for the revised manuscript. The
authoritative executable paths are:

- `src/features/res_aware_feature_engineering.py`
- `src/ml_threshold_selection/labeling.py`
- `src/ml_threshold_selection/training_pipeline.py`
- `src/ml_threshold_selection/prediction_analysis.py`
- `src/ml_threshold_selection/fabric_bootstrap.py`
- `cross_validation.py`

Historical exploratory classes remain for compatibility but are not the paper
pipeline unless this document explicitly names them.

## Input geometry and validity

Avizo exports `EigenVal1-3` and `EigenVec1-3` as the eigensystem of the binary
object coordinate-covariance matrix. For a uniform solid equivalent ellipsoid,
the covariance eigenvalue along principal direction `i` is `lambda_i = a_i^2/5`.
The physical semiaxis is therefore:

```text
a_i = sqrt(5 lambda_i)
```

Rows with missing, non-finite, zero, or negative eigenvalues are invalid for
logarithmic ellipsoid calculations and are excluded and counted. Zero-valued
eigenvector components are valid direction cosines and are preserved.

The released model accepts one scalar voxel edge length per sample and computes
voxel count as `V/d^3`. This assumes isotropic reconstructed voxels. For
anisotropic sampling, users must resample to isotropic physical coordinates or
extend the implementation to use `dx * dy * dz` for voxel volume. They must
also verify whether the exported covariance was calculated from calibrated
physical coordinates; if it was calculated in voxel-index coordinates, the
covariance must be transformed or recalculated in physical coordinates before
eigendecomposition. A scalar edge length must
not be used to imply that `dx = dy = dz` unless the reconstruction metadata
confirm it.

## Classifier features

### Ellipsoid tensor

Let the rows of `Q` be the orthonormal principal-axis directions. The classifier
uses a dimensionless quadratic-form ellipsoid tensor and its matrix logarithm.
With the fixed reference length `a_ref = 1 mm`:

```text
E* = Q^T diag[(a_1/a_ref)^-2, (a_2/a_ref)^-2, (a_3/a_ref)^-2] Q
L = log(E*)
  = Q^T diag[-2 log(a_1/a_ref), -2 log(a_2/a_ref),
             -2 log(a_3/a_ref)] Q
```

Because the input semiaxes are expressed in millimetres and `a_ref = 1 mm`,
making the reference explicit does not change the reported numerical features.
It prevents the logarithm from being interpreted as acting on a dimensional
quantity and fixes the unit convention needed for reproduction.

This follows the log-ellipsoid representation implemented in the 20 June 2017
static `geologyGeometry` distribution. The six Frobenius-preserving Mandel
components are reordered in this Python implementation as diagonal components
first:

```text
[L11, L22, L33, sqrt(2)L12, sqrt(2)L13, sqrt(2)L23]
```

The Avizo covariance-to-semiaxis conversion above follows the audited vendor
field definition, not the older `geologyGeometry` Avizo import helper, which
treated the exported values as squared semiaxes.

### Seven predictors

Each object has seven predictors:

1. `VoxelCount = Volume3d (mm^3) / d^3`, retained as a continuous value.
2. `L11`.
3. `L22`.
4. `L33`.
5. `sqrt(2)L12`.
6. `sqrt(2)L13`.
7. `sqrt(2)L23`.

The scaler is fitted on training samples only and applied unchanged to target or
held-out samples. Voxel count is not rounded before fitting or prediction.

## Expert-derived pseudo-labels

The reported model uses hard labels in physical-volume units:

```text
label = 1 (below-threshold class) if Volume3d < expert threshold
label = 0 (retained class)        otherwise
```

This is supervised model fitting with expert-derived pseudo-labels. It is not
semi-supervised learning because no unlabeled observations enter fitting.
Experimental soft-label methods remain in a legacy module and were not used for
the manuscript analyses.

## Model

The primary classifier is LightGBM binary GBDT with 31 leaves, learning rate
0.05, feature fraction 0.9, bagging fraction 0.8, bagging every five rounds, and
100 boosting rounds. The global, feature-fraction, bagging, and data seeds are
42; deterministic mode and force-col-wise are enabled. No class weighting is
used in the reported LightGBM analysis. The random-forest fallback is not the
model used for the reported results.

## Evaluation scope

Leave-one-sample-out evaluation holds out each complete training sample. Feature
standardization is fitted within each training fold. Corrected AUC values are:

| Held-out sample | AUC |
|---|---:|
| AKAN20 | 0.992 |
| ANA16937 | 0.997 |
| HL19335 | 0.989 |
| LE03 | 0.995 |
| LE19 | 0.910 |

These values measure object ranking against the expert volume-derived
pseudo-label rule. They are not independent estimates of physical
artifact-classification accuracy because voxel count is a predictor and the
labels are defined by volume. They do not independently validate physical
artifact identity or scalar-threshold recovery.

## Threshold determination

For candidate `Vmin`, define:

```text
A(Vmin) = mean predicted below-threshold probability among objects with
          VoxelCount >= Vmin
```

`A` is evaluated at 50 log-spaced continuous thresholds including the observed
positive voxel-count minimum and maximum. It is smoothed with a one-dimensional
Gaussian filter with `sigma = 1` grid interval and reflecting boundary handling.
Numerical derivatives with respect to `log10(Vmin)` use centred differences at
interior points and first-order one-sided differences at endpoints.

- **Loose threshold:** the interior grid point with maximum second derivative.
  If the maximum is an endpoint, no internal inflection threshold is reported.
  The operational threshold is the ceiling of the continuous value.
- **Strict threshold:** one voxel above the floor of the largest continuous
  voxel count whose predicted probability is greater than the tolerance (0.01
  for the reported seed-42 analysis). If no object exceeds the tolerance, use
  the ceiling of the smallest observed continuous voxel count. If the strict
  candidate is below a reported loose inflection point, raise it to that point.
  The applied strict threshold is the ceiling of the resulting value.
- **Retention:** both thresholds use `VoxelCount >= Vmin`.

The scalar cutoff is a sample-specific candidate sensitivity rule. It is not
the full multidimensional classifier boundary, a universal physical resolution
limit, or a validated replacement for expert threshold selection.

## Mean fabric ellipsoid

For fabric averaging, object `i` is represented by a semiaxis tensor. Semiaxes
are first normalized by the same `a_ref = 1 mm` before the logarithm:

```text
S_i* = Q_i^T diag(a_i1/a_ref, a_i2/a_ref, a_i3/a_ref) Q_i
log(S_i*) = Q_i^T diag[log(a_i1/a_ref), log(a_i2/a_ref),
                       log(a_i3/a_ref)] Q_i
S_mean* = exp[(1/n) sum_i log(S_i*)]
```

The relation to the classifier's quadratic-form tensor is exact:

```text
S_i* = (E_i*)^(-1/2)
log(S_i*) = -0.5 log(E_i*)
```

Every retained object receives equal weight; the mean is not volume-weighted.
The operation is the arithmetic mean of the six log-ellipsoid coordinates
followed by the matrix exponential. It is algebraically equivalent to `ellMean`
in the audited static `geologyGeometry` implementation, which averages
`log(E_i*)` and converts the mean quadratic form back to semiaxes. The two
representations therefore return the same mean semiaxes and principal
directions. Multiplication by `a_ref` recovers semiaxes in millimetres if
absolute mean size is required. `S_mean*` is a population mean shape ellipsoid,
not by definition a finite-strain or deformation tensor.

Let `V1 >= V2 >= V3 > 0` be the dimensionless principal semiaxes of `S_mean*`, `f_i = ln(V_i)`,
and `f_bar = (f1 + f2 + f3)/3`. The reported geometric descriptors are:

```text
P' = exp{sqrt[2((f1-f_bar)^2 + (f2-f_bar)^2 + (f3-f_bar)^2)]}
T  = (2f2 - f1 - f3) / (f1 - f3)
```

`P' = 1` for a sphere. Under this ordering and sign convention, `T = -1` for an
axially prolate ellipsoid, `T = +1` for an axially oblate ellipsoid, and `T = 0`
at the prolate-oblate boundary. `T` is Jelinek's shape parameter and has a
Lode-type normalized-intermediate-value form; terminology is not transferred
between fields without stating conventions.

## Bootstrap and scan-axis endpoint

The Fig 4 sensitivity grid is separate from the classifier's 50-point grid. It
begins at the observed minimum object volume and advances in `0.25 log10`
increments while at least 50 objects remain; the loose and strict operating
points are inserted explicitly. The 50-object stopping rule limits unstable
visualization at very small retained `n` and is not an artifact-classification
criterion. The LE01 implementation evaluates 17 thresholds.

At each threshold, each of 1,000 bootstrap iterations samples `n` retained
objects with replacement from the `n` retained objects and recalculates
`S_mean`, `P'`, and `T` using seed 42. Box plots show median, interquartile range,
and Tukey whiskers. Tabulated 95% bootstrap intervals use the 2.5th and 97.5th
percentiles. This object bootstrap treats retained objects as exchangeable. It
does not model spatial dependence or propagate reconstruction, segmentation,
voxel-size, expert-threshold, model, or registration uncertainty.

The separate LE01 endpoint measures the fraction of unoriented fitted axes
within 5 degrees of any scan axis. Its isotropic expectation is:

```text
3[1 - cos(5 degrees)] = 0.011416
```

This endpoint tests attenuation of the targeted low-volume scan-axis signal. It
does not identify the physical truth of every object.

## Fig 1 orientation-density calculation

`generate_fig1_stereonets.py` plots the three exported Avizo principal axes as
unoriented axial data on lower-hemisphere equal-area Schmidt nets. Antipodal
directions are combined through `abs(dot(u, v))`. Density follows the modified
Kamb calculation used in TomoFab `DataDens.m`, locked to commit
`2697865623c3afa34626abdd765183825180a069`: sigma 3, a 50 by 50 base grid,
linear display interpolation at 0.2 base-grid-cell spacing, and ten equal
contour intervals over each panel's evaluated range. TomoFab's three-sigma
grid is divided by 3 to report multiples of uniform density (m.u.d.). The
complete base grids, contour levels, retained counts, settings, and input hash
are exported with the figure.

The corrected panels use inclusive retention at 50 and 154 voxels. A regression
test reproduces the printed raw-grid extrema of the superseded figure and shows
that its old loose and strict panels used 75 and 145 voxels, respectively.

## Legacy-input limitation

The five training tables and the 4,991-object LE01 table are legacy derivatives
created by an earlier preparation script that removed rows with a zero
eigenvalue or exported `Anisotropy == 1`. Raw exclusion counts are unavailable.
The corrected preparation script no longer removes solely on `Anisotropy == 1`
and never substitutes an arbitrary positive constant for invalid eigenvalues.

## References

- Brandon MT. Analysis of geologic strain data in strain-magnitude space. J Struct Geol. 1995;17:1375-1385. doi:10.1016/0191-8141(95)00032-9.
- Chatzaras V, Lusk ADJ, Chapman T, Aldanmaz E, Davis JR, Tikoff B. Transpressional deformation in the lithospheric mantle beneath the North Anatolian Fault Zone. Tectonophysics. 2021;815:228989. doi:10.1016/j.tecto.2021.228989.
- Davis JR. geologyGeometry: structural-geology functions for R, static distribution dated 20 June 2017 [software]. Available from: https://nicolasmroberts.github.io/geologyGeometry.zip.
- Efron B, Tibshirani RJ. An Introduction to the Bootstrap. New York: Chapman & Hall/CRC; 1993.
- FEI. Release notes: Avizo 9.0 Beta, September 2014. Available from: https://assets.thermofisher.com/TFS-Assets/MSD/Product-Updates/release-notes-avizo-900-beta.pdf.
- Jelinek V. Characterization of the magnetic fabric of rocks. Tectonophysics. 1981;79:T63-T67. doi:10.1016/0040-1951(81)90110-4.
- Ke G, Meng Q, Finley T, Wang T, Chen W, Ma W, et al. LightGBM: a highly efficient gradient boosting decision tree. Adv Neural Inf Process Syst. 2017;30:3146-3154.
- Thermo Fisher Scientific. Moments and orientations. ImageDev Reference Manual, version 2025.1. Available from: https://developer.imageviz.com/refmans/2025-1/ImageDev/html/Processing_ImageAnalysis_MomentsAndOrientations.html.
