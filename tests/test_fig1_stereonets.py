import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "generate_fig1_stereonets.py"
SPEC = importlib.util.spec_from_file_location("generate_fig1_stereonets", MODULE_PATH)
FIG1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FIG1)


def test_modified_kamb_density_is_invariant_to_axis_sign():
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    _, _, positive = FIG1.modified_kamb_mud_grid(vectors, grid_n=10)
    _, _, negative = FIG1.modified_kamb_mud_grid(-vectors, grid_n=10)
    np.testing.assert_allclose(positive, negative, rtol=0.0, atol=1e-12)


def test_legacy_figure_numerical_fingerprint_identifies_old_settings():
    workbook = ROOT / "examples" / "Quantity_LE01.xlsx"
    compact = ROOT.parent / "input" / "Quantity_LE01.csv.gz"
    if workbook.is_file():
        data = pd.read_excel(workbook)
    else:
        data = pd.read_csv(compact, float_precision="round_trip")
    voxel_counts = data["Volume3d (mm^3) "].to_numpy() / 0.03 ** 3
    expected = {
        (0, 1): (0.006233, 38.996131, 4991),
        (75, 1): (0.066453, 3.353247, 1726),
        (145, 1): (0.088990, 3.107228, 1110),
    }
    for (threshold, axis_number), (minimum, maximum, retained_n) in expected.items():
        mask = np.ones(len(data), dtype=bool) if threshold == 0 else voxel_counts >= threshold
        retained = data.loc[mask]
        assert len(retained) == retained_n
        vectors = FIG1._unit_axial_vectors(retained, axis_number)
        _, _, density = FIG1.modified_kamb_mud_grid(vectors, grid_n=50, sigma=3.0)
        assert np.isclose(density.min(), minimum, atol=1e-6)
        assert np.isclose(density.max(), maximum, atol=1e-6)
