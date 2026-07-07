# Scientific Methods

Core algorithms and methods used in the ML Threshold Selection toolkit.
This document mirrors the Methods section of the companion PLOS ONE article;
the implementation lives in `src/` and every reported number can be reproduced
with `cross_validation.py`.

## Ellipsoid Feature Engineering

### Mathematical Foundation

Each particle is represented as an ellipsoid defined by its principal axes and eigenvalues. The ellipsoid tensor is constructed as:

```
T = [λ₁ 0  0 ]
    [0  λ₂ 0 ]
    [0  0  λ₃]
```

Where λ₁, λ₂, λ₃ are the eigenvalues (principal values) of the particle.

### Log-Euclidean Mapping

The system uses a specialized log-ellipsoid tensor representation for particle shape analysis:

```
L = Q^T × log(E) × Q
```

Where:
- `Q` is the rotation matrix from eigenvectors
- `E` is the diagonal matrix with `-2×log(√λᵢ)` on diagonal
- The resulting 6D tensor components are: L₁₁, L₂₂, L₃₃, L₁₂, L₁₃, L₂₃

This 6D vectorization follows the ellipsoidal-statistics framework of the
geologyGeometry package (Davis, 2019; see also Chatzaras et al. 2021,
Section 3.2).

### 7D Feature Vector

The following 7 features are extracted from each particle:

1. **VoxelCount**: `VoxelCount = Volume(mm³) / voxel_size_mm³` (continuous)
2. **L11**: Log-ellipsoid tensor component L₁₁
3. **L22**: Log-ellipsoid tensor component L₂₂
4. **L33**: Log-ellipsoid tensor component L₃₃
5. **sqrt2_L12**: `√2 × L₁₂` (off-diagonal component)
6. **sqrt2_L13**: `√2 × L₁₃` (off-diagonal component)
7. **sqrt2_L23**: `√2 × L₂₃` (off-diagonal component)

VoxelCount makes the model resolution-aware (the voxel size of each sample is
an explicit physical input), while the log-ellipsoid components encode shape
anisotropy and 3D orientation. Features are standardized (zero mean, unit
variance) before training.

## Classifier and Semi-Supervised Labelling

### Model

The classifier is a **LightGBM** gradient-boosted decision-tree model
(`num_leaves = 31`, `learning_rate = 0.05`, 100 boosting rounds), with a
RandomForest fallback (100 trees, max depth 10) when LightGBM is unavailable.
Output probabilities are **calibrated** with `CalibratedClassifierCV`
(isotonic calibration for hard labels; sigmoid calibration for soft labels).

### Two training modes

1. **Supervised**: individual grains are manually labelled as true grains or
   artifacts.
2. **Semi-supervised** (default): for each training sample an expert specifies,
   from stereonet inspection in TomoFab, the reference volume threshold above
   which the axis-parallel artifact clustering disappears (protocol Step 53).
   Pseudo-labels are then assigned by this reference:

   ```
   label(grain) = artifact  if  Volume3d < expert Vmin
                  true grain otherwise
   ```

   Two optional refinements are implemented: a sigmoid soft-label transition in
   log-volume around the expert threshold, and an uncertainty band (±10% of the
   threshold) in which labels grade between the two classes.

### Validation

Classification performance is assessed by **leave-one-sample-out** and
**stratified five-fold** cross-validation across the five training samples
(35,745 segmented objects; see `examples/README.md` for provenance). Across
held-out samples the classifier attains an **AUC of 0.96–0.99**, and per-sample
machine-learning thresholds reproduce the expert reference thresholds to within
1–6% (S3 Fig of the article). Reproduce with:

```bash
python cross_validation.py --data "trained model" --config examples/expert_thresholds.csv
```

## Threshold Determination

From the calibrated per-grain artifact probabilities, the **cumulative
artifact-rate curve** A(Vmin) is computed: the mean artifact probability of all
objects retained at a given minimum-volume threshold.

- **Loose threshold**: the *inflection point* of A(Vmin), defined as the volume
  that maximises the second derivative of the smoothed artifact-rate curve (the
  curve is evaluated over 50 log-spaced volume levels and Gaussian-smoothed),
  where the steep initial decline in artifact rate levels off.
- **Strict threshold**: the smallest volume at which the retained population
  contains no object with an artifact probability above a small tolerance
  (0.01 by default; configurable via the Config Threshold dialog).

### Why a single scalar Vmin (not per-grain filtering)

Although the classifier evaluates each object in the full 7D feature space, its
decision is applied as a single global volume cut-off, for two reasons:

1. **Compatibility**: standard 3D fabric software (TomoFab, Avizo) filters by a
   minimum volume.
2. **No selection bias**: probability-based per-grain removal would
   preferentially discard small *true* grains whose shape resembles an
   artifact, distorting the SPO of the retained population.

Complementing the volume threshold, objects whose **shortest axis spans fewer
than five voxels** are removed, because they cannot define a reliable ellipsoid
regardless of total volume.

## Fabric Analysis (T and P' Parameters)

### Jelínek (1981) Methodology

The fabric analysis follows the Jelínek (1981) approach for characterizing particle orientation distributions.

### Log-Euclidean Mean Tensor

For each volume threshold, the mean fabric tensor is computed using log-Euclidean averaging (tensor averaging following Brandon, 1995):

```
T_mean = exp(1/n * Σᵢ log(Tᵢ))
```

Where n is the number of particles and Tᵢ is the tensor of particle i.

### Eigenvalue Analysis

The principal values of the mean tensor are extracted:

```
T_mean = [λ₁ 0  0 ]
         [0  λ₂ 0 ]
         [0  0  λ₃]
```

### T and P' Parameters

The fabric parameters are calculated following Jelínek (1981):

**T Parameter**:
```
T = (2f₂ - f₁ - f₃) / (f₁ - f₃)
```

**P' Parameter**:
```
P' = exp√2[(f₁ - f)² + (f₂ - f)² + (f₃ - f)²]
```

Where:
- f₁, f₂, f₃ are the natural logs of the normalized magnitudes of the maximum (Φ₁), intermediate (Φ₂), and minimum (Φ₃) axes of the fabric ellipsoid
- f = (f₁ + f₂ + f₃)/3

For detailed interpretation of these parameters, refer to the original literature (Jelínek, 1981).

## Bootstrap Analysis

### Statistical Resampling

For each volume threshold, grains are resampled with replacement to estimate
confidence intervals: at each of the 1,000 iterations, n grains are drawn with
replacement from the filtered population, where n is the number of retained
grains for that sample (Efron & Tibshirani 1993).

### Algorithm

1. **Bootstrap Sampling**: for each threshold, create B bootstrap samples (default B = 1000)
2. **Tensor Computation**: compute the mean tensor for each bootstrap sample
3. **Parameter Calculation**: calculate T and P' for each bootstrap sample
4. **Confidence Intervals**: compute 95% confidence intervals from the bootstrap distribution

```
CI_95 = [percentile_2.5, percentile_97.5]
```

## Implementation Details

- Modular pipeline under `src/` (feature engineering, labelling, training,
  threshold finding, fabric calculation, GUI controller)
- Numerical stability: zero eigenvalues are replaced with a negligible
  constant (1e-8) before logarithms (performed by `tools/BatchFile.py`)
- Efficient tensor operations using vectorized NumPy

## References

- Brandon, M.T., 1995. Analysis of geologic strain data in strain-magnitude space. Journal of Structural Geology 17, 1375–1385. https://doi.org/10.1016/0191-8141(95)00035-4
- Chatzaras, V., Lusk, A.D.J., Chapman, T., Aldanmaz, E., Davis, J.R., Tikoff, B., 2021. Transpressional deformation in the lithospheric mantle beneath the North Anatolian Fault Zone. Tectonophysics 815, 229007. https://doi.org/10.1016/j.tecto.2021.229007 (Section 3.2: 6D log-ellipsoid vectorization.)
- Davis, J.R., 2019. geologyGeometry: an R package for structural geology. http://www.joshuadavis.us/software/index.html
- Efron, B., Tibshirani, R.J., 1993. An Introduction to the Bootstrap. Chapman & Hall/CRC, New York.
- Jelínek, V., 1981. Characterization of the magnetic fabric of rocks. Tectonophysics 79, T63–T67. https://doi.org/10.1016/0040-1951(81)90110-4
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y., 2017. LightGBM: a highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems 30, 3146–3154.
- Petri, B., Almqvist, B.S.G., Pistone, M., 2020. 3D rock fabric analysis using micro-tomography: An introduction to the open-source TomoFab MATLAB code. Computers & Geosciences 138, 104444. https://doi.org/10.1016/j.cageo.2020.104444. Code: https://github.com/benpetri/tomofab