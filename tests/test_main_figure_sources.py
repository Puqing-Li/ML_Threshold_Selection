import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_main_figures.py"
SPEC = importlib.util.spec_from_file_location("generate_main_figures", MODULE_PATH)
FIGURES = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FIGURES)


def test_retained_probability_curve_uses_inclusive_threshold_and_physical_volume():
    voxels = np.array([1.0, 2.0, 4.0])
    probabilities = np.array([0.9, 0.5, 0.1])
    voxel_size_mm = 0.03

    curve = FIGURES._retained_probability_curve(voxels, probabilities, voxel_size_mm)

    first = curve.iloc[0]
    last = curve.iloc[-1]
    assert first["retained_n"] == 3
    assert np.isclose(first["mean_predicted_below_threshold_probability"], 0.5)
    assert np.isclose(first["candidate_threshold_mm3"], voxel_size_mm ** 3)
    assert last["retained_n"] == 1
    assert np.isclose(last["mean_predicted_below_threshold_probability"], 0.1)
