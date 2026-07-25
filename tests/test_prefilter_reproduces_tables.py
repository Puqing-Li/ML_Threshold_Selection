# -*- coding: utf-8 -*-
"""The released prefilter must reproduce the deposited per-grain tables.

`tools/BatchFile.py` turns a raw Avizo Label-Analysis export into the
`Quantity_*.xlsx` tables shipped in `examples/`. Two conditions are removed: a
missing, non-finite, zero or negative eigenvalue, and `Anisotropy == 1`, which
marks a vanishing shortest axis whose eigenvalue underflows to about 1e-25
rather than to exactly zero.

Dropping either condition changes the retained population, so these tests pin
both.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_batchfile():
    spec = importlib.util.spec_from_file_location("batchfile", ROOT / "tools" / "BatchFile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(rows):
    """Build a minimal export-shaped frame from (eigenvalues, anisotropy) rows."""
    return pd.DataFrame([
        {"EigenVal1": e1, "EigenVal2": e2, "EigenVal3": e3, "Anisotropy": a}
        for (e1, e2, e3), a in rows
    ])


def test_removes_non_positive_and_non_finite_eigenvalues():
    batchfile = _load_batchfile()
    frame = _frame([
        ((4.0, 2.0, 1.0), 0.5),
        ((4.0, 2.0, 0.0), 0.5),
        ((4.0, 2.0, -1.0), 0.5),
        ((4.0, np.nan, 1.0), 0.5),
        ((np.inf, 2.0, 1.0), 0.5),
    ])
    kept, qc = batchfile.filter_invalid_eigenvalue_rows(frame)
    assert len(kept) == 1
    assert qc["invalid_eigenvalue_count"] == 4
    assert qc["retained_count"] == 1


def test_removes_vanishing_shortest_axis_that_survives_the_positivity_test():
    batchfile = _load_batchfile()
    underflowed = 1.2e-25
    frame = _frame([
        ((4.0, 2.0, 1.0), 0.5),
        ((4.3e-3, 7.6e-14, underflowed), 1.0),
    ])
    kept, qc = batchfile.filter_invalid_eigenvalue_rows(frame)
    assert qc["invalid_eigenvalue_count"] == 0, "the underflowed value is positive and finite"
    assert qc["degenerate_anisotropy_count"] == 1
    assert len(kept) == 1


def test_tolerates_an_export_without_an_anisotropy_column():
    batchfile = _load_batchfile()
    frame = _frame([((4.0, 2.0, 1.0), 0.5), ((4.0, 2.0, 0.0), 0.5)]).drop(columns=["Anisotropy"])
    kept, qc = batchfile.filter_invalid_eigenvalue_rows(frame)
    assert len(kept) == 1
    assert qc["degenerate_anisotropy_count"] == 0


@pytest.mark.parametrize("table,expected", [
    ("Quantity_LE01.xlsx", 4991),
    ("Quantity_12RH26.xlsx", 31896),
    ("Quantity_BG02-4B.xlsx", 10169),
    ("Quantity_BG04-44B.xlsx", 14520),
    ("Quantity_CC10-18.xlsx", 11320),
])
def test_deposited_tables_carry_only_usable_objects(table, expected):
    """Whatever produced them, the shipped tables must contain no degenerate object."""
    path = ROOT / "examples" / table
    if not path.is_file():
        pytest.skip(f"{table} is not present")
    frame = pd.read_excel(path)
    assert len(frame) == expected

    eigenvalues = frame[["EigenVal1", "EigenVal2", "EigenVal3"]].to_numpy(dtype=float)
    assert np.isfinite(eigenvalues).all()
    assert (eigenvalues > 0).all()
    if "Anisotropy" in frame.columns:
        assert (pd.to_numeric(frame["Anisotropy"], errors="coerce") != 1).all()
