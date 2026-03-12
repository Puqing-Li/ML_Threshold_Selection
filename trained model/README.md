# Pre-trained Model and Samples Directory

This folder contains the standardized machine learning artifacts and datasets provided with the PLOS ONE GUI Tool. It allows users to instantly perform strict/loose threshold mapping without needing to prepare extensive training data manually.

## Contents

1. **last_time_model.pkl / Trainmodel.pkl**:
   - The pre-trained LightGBM model utilizing our specialized resolution-aware feature engineering.
   - It is loaded instantly via the "Load Last Time Model" button in the GUI.
   
2. **Standard 5 Training Samples (.xlsx)**:
   - `totalAKAN20.xlsx`
   - `totalANA16937.xlsx`
   - `totalHL19335.xlsx`
   - `totalLE03.xlsx`
   - `totalLE19.xlsx`
   - These 5 extensive micro-CT sample datasets contain the raw volumetric and shape parameters used to originally train the provided pipeline.

3. **voxel_sizes.xlsx**:
   - This metadata table maps the exact scanning resolution (Voxel Size in mm) to each of the aforementioned training samples.
   - Crucial for the `Resolution-Aware Feature Space Transformation` logic to normalize all features correctly into physical volumes.
