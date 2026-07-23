# Model and Feature Guide

The active v1.3.0 workflow is documented in
[`user_guide.md`](user_guide.md) and the scientific definitions are documented
in [`SCIENTIFIC_METHODS.md`](SCIENTIFIC_METHODS.md).

The application requires one measured voxel-edge length per training or test
sample. It constructs seven features: continuous voxel count and six
log-ellipsoid tensor components. No default resolution is inserted.
