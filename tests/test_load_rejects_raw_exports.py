# -*- coding: utf-8 -*-
"""Loading must reject an unusable table, not accept it and fail several steps later.

A raw Avizo Label-Analysis export carries objects whose fitted ellipsoid is
degenerate. The fabric calculation takes the logarithm of each eigenvalue, so
those objects cannot be used at all, and the check belongs at load time: the
column check alone reported such a file as loaded successfully and the run only
stopped at prediction, after the voxel size had been entered.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml_threshold_selection.data_io import validate_training_data  # noqa: E402

COLUMNS = [
    "Volume3d (mm^3) ",
    "EigenVal1", "EigenVal2", "EigenVal3",
    "EigenVec1X", "EigenVec1Y", "EigenVec1Z",
    "EigenVec2X", "EigenVec2Y", "EigenVec2Z",
    "EigenVec3X", "EigenVec3Y", "EigenVec3Z",
]


class Recorder:
    """Stands in for the application, capturing what the user would be told."""

    def __init__(self):
        self.lines = []

    def log(self, message):
        self.lines.append(str(message))


def _table(eigenvalue_rows):
    rows = []
    for e1, e2, e3 in eigenvalue_rows:
        row = {c: 1.0 for c in COLUMNS}
        row["EigenVal1"], row["EigenVal2"], row["EigenVal3"] = e1, e2, e3
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def test_accepts_a_table_whose_eigenvalues_are_all_usable():
    app = Recorder()
    assert validate_training_data(app, _table([(4.0, 2.0, 1.0), (9.0, 3.0, 1.0)]))


@pytest.mark.parametrize("degenerate", [0.0, -1.0, np.nan, np.inf])
def test_rejects_a_table_containing_an_unusable_eigenvalue(degenerate):
    app = Recorder()
    table = _table([(4.0, 2.0, 1.0), (4.0, 2.0, degenerate)])
    assert not validate_training_data(app, table)
    assert any("cannot be used" in line for line in app.lines)


def test_says_how_many_objects_are_unusable_and_what_to_do():
    app = Recorder()
    table = _table([(4.0, 2.0, 1.0), (4.0, 2.0, 0.0), (4.0, 2.0, -1.0)])
    assert not validate_training_data(app, table)
    joined = " ".join(app.lines)
    assert "2 of 3 objects" in joined
    assert "0. Prepare Raw Data" in joined


def test_still_rejects_a_table_that_is_missing_a_column():
    app = Recorder()
    table = _table([(4.0, 2.0, 1.0)]).drop(columns=["EigenVal2"])
    assert not validate_training_data(app, table)
    assert any("Missing required columns" in line for line in app.lines)


@pytest.mark.parametrize("name", [
    "Quantity_LE01.xlsx",
    "Quantity_12RH26.xlsx",
    "Quantity_BG02-4B.xlsx",
    "Quantity_BG04-44B.xlsx",
    "Quantity_CC10-18.xlsx",
])
def test_every_shipped_table_passes_load_validation(name):
    path = ROOT / "examples" / name
    if not path.is_file():
        pytest.skip(f"{name} is not present")
    app = Recorder()
    assert validate_training_data(app, pd.read_excel(path)), app.lines
