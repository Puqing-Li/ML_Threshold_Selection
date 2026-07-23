from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from BatchFile import (
    extract_sample_id,
    extract_sample_id_from_processed_xlsx,
    filter_invalid_eigenvalue_rows,
)


def test_raw_sample_ids_preserve_meaningful_underscore_segments():
    cases = {
        "12RH26_41-Y_0000.tif.Label-Analysis(2).csv": "12RH26_41-Y_0000",
        "14RH_7_57um.tif.Label-Analysis(2).csv": "14RH_7",
        "BG02_4B_39um.Label-Analysis(2).csv": "BG02_4B",
        "CC10_18-74spinelrawvolume.Label-Analysis.csv": "CC10_18-74spinelrawvolume",
    }

    for filename, expected in cases.items():
        assert extract_sample_id(filename) == expected


def test_processed_sample_ids_preserve_meaningful_underscore_segments():
    cases = {
        "totalBG02_4B.xlsx": "BG02_4B",
        "Quantity_BG02_4B.xlsx": "BG02_4B",
        "EigensBG02_4B.xlsx": "BG02_4B",
        "VolumeEigenBG02_4B.xlsx": "BG02_4B",
    }

    for filename, expected in cases.items():
        assert extract_sample_id_from_processed_xlsx(filename) == expected


def test_prefilter_excludes_all_invalid_eigenvalues_without_substitution():
    df = pd.DataFrame({
        "EigenVal1": [1.0, 0.0, 1.0, 1.0, np.nan],
        "EigenVal2": [2.0, 2.0, -1.0, np.inf, 2.0],
        "EigenVal3": [3.0, 3.0, 3.0, 3.0, 3.0],
        "EigenVec1X": [0.0, 0.1, 0.2, 0.3, 0.4],
        "Anisotropy": [1.0, 0.2, 0.3, 0.4, 0.5],
    })

    filtered, qc = filter_invalid_eigenvalue_rows(df)

    assert filtered.index.tolist() == [0]
    assert filtered.loc[0, "EigenVec1X"] == 0.0
    assert filtered.loc[0, "Anisotropy"] == 1.0
    assert qc == {
        "initial_count": 5,
        "invalid_eigenvalue_count": 4,
        "retained_count": 1,
    }


def test_prefilter_requires_all_three_eigenvalue_columns():
    df = pd.DataFrame({"EigenVal1": [1.0], "EigenVal2": [2.0]})

    try:
        filter_invalid_eigenvalue_rows(df)
    except ValueError as exc:
        assert "EigenVal3" in str(exc)
    else:
        raise AssertionError("missing EigenVal3 should fail")
