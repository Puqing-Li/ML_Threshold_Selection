# Scientific Methods

Core algorithms and methods used in the ML Threshold Selection toolkit.

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

### 7D Feature Vector

The following 7 features are extracted from each particle:

1. **VoxelCount**: `VoxelCount = Volume(mm³) / voxel_size_mm³` (continuous)
2. **L11**: Log-ellipsoid tensor component L₁₁
3. **L22**: Log-ellipsoid tensor component L₂₂  
4. **L33**: Log-ellipsoid tensor component L₃₃
5. **sqrt2_L12**: `√2 × L₁₂` (off-diagonal component)
6. **sqrt2_L13**: `√2 × L₁₃` (off-diagonal component)
7. **sqrt2_L23**: `√2 × L₂₃` (off-diagonal component)

### Resolution-Aware Normalization

Features are normalized by voxel size to ensure cross-sample compatibility:

```
normalized_feature = feature / (voxel_size_mm)³
```

## Fabric Analysis (T and P' Parameters)

### Jelínek (1981) Methodology

The fabric analysis follows the Jelínek (1981) approach for characterizing particle orientation distributions.

### Log-Euclidean Mean Tensor

For each volume threshold, the mean fabric tensor is computed using log-Euclidean averaging:

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

For each volume threshold, particles are resampled with replacement to estimate confidence intervals.

### Algorithm

1. **Bootstrap Sampling**: For each threshold, create B bootstrap samples (default B=1000)
2. **Tensor Computation**: Compute mean tensor for each bootstrap sample
3. **Parameter Calculation**: Calculate T and P' for each bootstrap sample
4. **Confidence Intervals**: Compute 95% confidence intervals from bootstrap distribution

### Confidence Interval Calculation

```
CI_95 = [percentile_2.5, percentile_97.5]
```

## Machine Learning Pipeline

### Feature Standardization

All features are standardized using StandardScaler:

```
X_scaled = (X - μ) / σ
```

Where μ is the mean and σ is the standard deviation.

### Model Training

The system supports multiple algorithms:

- **LightGBM** (default): Gradient boosting with high performance
- **Random Forest**: Ensemble method with good interpretability
- **SVM**: Support vector machine for non-linear boundaries

### Cross-Validation

K-fold cross-validation is used for robust performance estimation:

```
CV_score = 1/k * Σᵢ score(fold_i)
```

### Hyperparameter Optimization

Grid search or random search is used to optimize hyperparameters:

```python
param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
```

## Probability Calculation

### Artifact Probability

The model outputs the probability that a particle is an artifact:

```
P(artifact) = sigmoid(Σᵢ wᵢ * fᵢ + b)
```

Where wᵢ are learned weights, fᵢ are features, and b is the bias term.

### Threshold-Based Probability

For semi-supervised learning, probabilities are derived from expert thresholds:

```
P(artifact) = 1 - exp(-α * (volume - threshold))
```

Where α is a scaling parameter.

## Threshold Determination

### Loose Threshold (Inflection Point)

The loose threshold is identified as the inflection point of the artifact rate curve:

```
d²(artifact_rate)/d(volume)² = 0
```

This represents the optimal balance between artifact removal and particle retention.

### Strict Threshold (Zero Artifacts)

The strict threshold is determined as the volume where all particles have artifact probability > P_threshold:

```
strict_threshold = max(volume | P(artifact) > P_threshold)
```

Default P_threshold = 0.01 (configurable).

### Grid Search

A volume grid is generated for threshold evaluation:

```python
v_min = min(volumes) * 0.1
v_max = max(volumes) * 10
volume_grid = np.logspace(log10(v_min), log10(v_max), n_points)
```

### Optimization Criteria

The optimal threshold minimizes the combined cost:

```
Cost = α * artifact_rate + β * particle_loss_rate
```

Where α and β are weighting factors.

## Implementation Details

### Memory Optimization

- Use `float32` for large datasets
- Process samples in batches
- Enable data type optimization

### Performance Optimization

- Parallel processing where available
- Feature selection for high-dimensional data
- Efficient tensor operations using NumPy

### Numerical Stability

- Add small epsilon values to prevent division by zero
- Use log-sum-exp trick for numerical stability
- Regularize eigenvalues to prevent singular matrices

## References

1. **Jelínek, V. (1981)**. Characterization of the magnetic fabric of rocks. *Tectonophysics*, 79(1-4), T63-T67.
2. **TomoFab Software**: https://github.com/ctlab/TomoFab
