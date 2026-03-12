# User Guide

Complete user manual for the ML Threshold Selection toolkit.

## Table of Contents

- [Quick Start](#quick-start)
- [GUI Workflow](#gui-workflow)
- [Data Requirements](#data-requirements)
- [Command Line Usage](#command-line-usage)
- [Python API](#python-api)
- [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Installation

```bash
git clone https://github.com/VincentDD1125/ML-Threshold-Selection.git
cd ML-Threshold-Selection
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Determine Expert Thresholds

Before using the system, determine expert thresholds using TomoFab or similar software:

1. Load your particle data in TomoFab
2. Generate lower hemisphere equal-area projections
3. Analyze fabric patterns to identify optimal volume thresholds
4. Record thresholds for input into the system

### 3. Launch Application

```bash
python main.py
```

## GUI Workflow

### Step 1: Load Training Data
- Click **"1. Load Training Data"**
- Select directory containing your training files
- Verify loaded samples

### Step 2: Input Expert Thresholds
- Click **"2. Input Expert Thresholds"**
- Enter thresholds in format: `sample_id:threshold_value`
- Use scientific notation (e.g., `1.23e-06`)
- Click **"Save Thresholds"**

### Step 3: Input Voxel Sizes
- Click **"3. Input Voxel Sizes"**
- Enter voxel size for each sample in millimeters
- Click **"Save Voxel Sizes"**

### Step 4: Feature Analysis (Optional)
- Click **"4. Feature Analysis"**
- Review feature extraction results
- Check for data quality issues

### Step 5: Train Model
- Click **"5. Train Model"**
- Select model type (LightGBM recommended)
- Wait for training completion
- Review training results

### Step 6: Load Test Data
- Click **"6. Load Test Data"**
- Select your test data file
- Verify loaded data

### Step 7: Predict Analysis
- Click **"7. Predict Analysis"**
- Review predicted thresholds
- Check dual threshold analysis plot

### Step 8: Export Results
- Click **"📤 Export Results"**
- Select output directory
- Choose export format

## Data Requirements

### Required Columns

| Column Name | Description | Units |
|-------------|-------------|-------|
| `Volume3d (mm^3) ` | Particle volume | mm³ |
| `EigenVal1`, `EigenVal2`, `EigenVal3` | Ellipsoid eigenvalues | dimensionless |
| `EigenVec1X`, `EigenVec1Y`, `EigenVec1Z` | First principal axis direction | unit vector |
| `EigenVec2X`, `EigenVec2Y`, `EigenVec2Z` | Second principal axis direction | unit vector |
| `EigenVec3X`, `EigenVec3Y`, `EigenVec3Z` | Third principal axis direction | unit vector |

### Data Quality Guidelines

- **Missing Values**: Will be filled with 0
- **Data Types**: All feature columns must be numeric
- **Volume Units**: Ensure volumes are in mm³ (not voxels)
- **Coordinate System**: Ensure consistent coordinate system across samples

## Command Line Usage

### Supervised Learning

```bash
python examples/scripts/run_supervised_demo.py \
  --train_dir /path/to/training/data \
  --test_file /path/to/test/file.xlsx \
  --voxel_size_mm 0.03
```

### Semi-supervised Learning

```bash
python examples/scripts/run_semi_supervised_demo.py \
  --thresholds examples/data/expert_thresholds.csv \
  --data_dir examples/data/train/
```

## Python API

### Supervised Learning

```python
from src.ml_threshold_selection.supervised_learner import SupervisedThresholdLearner
from src.ml_threshold_selection.feature_engineering import FeatureEngineer
import pandas as pd

# Load data
df = pd.read_excel('your_particle_data.xlsx')

# Extract features
feature_engineer = FeatureEngineer()
features = feature_engineer.extract_all_features(df)

# Train model
learner = SupervisedThresholdLearner()
learner.train(features, df['label'].values)

# Analyze sample
results = learner.analyze_sample(df)
print(f"Optimal threshold: {results['threshold']:.2e} mm³")
```

### Semi-supervised Learning

```python
from src.ml_threshold_selection.semi_supervised_learner import SemiSupervisedThresholdLearner

# Initialize learner
learner = SemiSupervisedThresholdLearner()

# Add expert thresholds
learner.add_expert_threshold('sample_001', 1.23e-06, confidence=1.0)
learner.load_sample_data('sample_001', 'sample_001_data.xlsx')

# Train model
learner.train(method='threshold_based', model_type='lightgbm')

# Analyze sample
results = learner.analyze_sample(df)
```

## Troubleshooting

### Common Issues

#### Data Loading Errors
- **Problem**: "No valid data files found"
- **Solution**: Check file format and required columns

#### Feature Extraction Errors
- **Problem**: "Missing required columns"
- **Solution**: Verify column names and data types

#### Model Training Errors
- **Problem**: "Training failed"
- **Solution**: Check data quality and sufficient samples

#### Prediction Errors
- **Problem**: "Prediction failed"
- **Solution**: Ensure model is trained and test data format matches

### Performance Issues

#### Memory Usage
- Use `float32` for large datasets
- Process samples in batches
- Enable data type optimization

#### Speed Optimization
- Use LightGBM for faster training
- Enable parallel processing
- Consider feature selection

## FAQ

### Q: What is the difference between loose and strict thresholds?
**A**: Loose threshold is the inflection point for optimal balance. Strict threshold removes all particles with artifact probability > 0.01.

### Q: How do I determine expert thresholds?
**A**: Use TomoFab to generate stereographic projections and analyze fabric patterns to identify optimal volume thresholds.

### Q: What if I don't have expert thresholds?
**A**: Use supervised learning with labeled data instead.

### Q: How accurate are the predictions?
**A**: Typically 85-95% accuracy on well-prepared datasets. Cross-validation scores are provided during training.

### Q: Can I use my own data format?
**A**: The system expects specific column names. Modify your data to match the required format.

### Q: What if my data has different units?
**A**: Ensure all volumes are in mm³ and all distances are in mm.

### Q: How do I interpret fabric analysis results?
**A**: For detailed interpretation of T and P' parameters, refer to the original literature (Jelínek, 1981).

## Support

- **Issues**: [GitHub Issues](https://github.com/VincentDD1125/ML-Threshold-Selection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/VincentDD1125/ML-Threshold-Selection/discussions)