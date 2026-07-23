import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.res_aware_feature_engineering import ResolutionAwareFeatureEngineer
from ml_threshold_selection.fabric_bootstrap import calculate_T_Pprime_from_vals
from ml_threshold_selection.fabric_boxplots_dual_thresholds import compute_fabric_params
from ml_threshold_selection.labeling import generate_labels_from_thresholds
from ml_threshold_selection.prediction_analysis import compute_dual_thresholds
from ml_threshold_selection.voxel_config import require_voxel_sizes
from tools.To_tomofab import convert_dataframe


def _geometry_rows(sample_ids, volumes):
    n = len(sample_ids)
    return pd.DataFrame(
        {
            'SampleID': sample_ids,
            'Volume3d (mm^3) ': volumes,
            'EigenVal1': np.full(n, 0.8),
            'EigenVal2': np.full(n, 0.2),
            'EigenVal3': np.full(n, 0.05),
            'EigenVec1X': np.ones(n),
            'EigenVec1Y': np.zeros(n),
            'EigenVec1Z': np.zeros(n),
            'EigenVec2X': np.zeros(n),
            'EigenVec2Y': np.ones(n),
            'EigenVec2Z': np.zeros(n),
            'EigenVec3X': np.zeros(n),
            'EigenVec3Y': np.zeros(n),
            'EigenVec3Z': np.ones(n),
        },
        index=[7] * n,
    )


def test_extract_by_sample_uses_each_samples_voxel_size():
    df = _geometry_rows(['A', 'B'], [8.0e-6, 8.0e-6])
    engineer = ResolutionAwareFeatureEngineer()

    scaled = engineer.extract_by_sample(
        df,
        voxel_sizes={'A': 0.02, 'B': 0.04},
        fit_scaler=True,
    )
    raw = engineer.scaler.inverse_transform(scaled)

    np.testing.assert_allclose(raw[:, 0], [1.0, 0.125])


def test_training_voxel_sizes_are_explicit_complete_and_finite():
    assert require_voxel_sizes({'A': 0.02, 'B': 0.04}, ['A', 'B']) == {
        'A': 0.02,
        'B': 0.04,
    }
    assert require_voxel_sizes({'AKAN20': 0.03}, ['totalAKAN20']) == {
        'totalAKAN20': 0.03,
    }
    with pytest.raises(ValueError, match='Missing measured voxel sizes'):
        require_voxel_sizes({'A': 0.02}, ['A', 'B'])
    with pytest.raises(ValueError, match='finite and greater than 0'):
        require_voxel_sizes({'A': np.nan}, ['A'])


def test_labels_use_expert_mm3_boundary_without_voxel_ceiling():
    threshold = 1.0e-3
    df = _geometry_rows(
        ['A', 'A', 'A'],
        [threshold - 1.0e-9, threshold, threshold + 1.0e-9],
    )

    labeled = generate_labels_from_thresholds(
        df,
        expert_thresholds={'A': threshold},
        voxel_sizes={'A': 0.04},
        sample_list=['A'],
    )

    assert labeled['label'].tolist() == [1, 0, 0]


def test_labels_do_not_silently_drop_a_sample_without_a_threshold():
    df = _geometry_rows(['A', 'B'], [1.0e-4, 1.0e-4])
    with pytest.raises(ValueError, match='Missing valid expert thresholds'):
        generate_labels_from_thresholds(
            df,
            expert_thresholds={'A': 1.0e-3},
            voxel_sizes={'A': 0.03, 'B': 0.04},
            sample_list=['A', 'B'],
        )


def test_strict_threshold_excludes_flagged_objects_at_integer_voxel_counts():
    voxels = np.array([1.0, 2.0, 3.0, 4.0])
    probabilities = np.array([0.8, 0.2, 0.001, 0.001])

    _, strict = compute_dual_thresholds(
        voxels, probabilities, strict_probability_threshold=0.01
    )

    assert strict == 3.0
    assert np.all(probabilities[voxels >= strict] <= 0.01)


def test_log_ellipsoid_tensor_uses_equivalent_semiaxes_and_rotation():
    angle = np.deg2rad(45.0)
    c, s = np.cos(angle), np.sin(angle)
    df = _geometry_rows(['A'], [1.0e-3])
    df.loc[:, ['EigenVec1X', 'EigenVec1Y', 'EigenVec1Z']] = [c, s, 0.0]
    df.loc[:, ['EigenVec2X', 'EigenVec2Y', 'EigenVec2Z']] = [-s, c, 0.0]

    engineer = ResolutionAwareFeatureEngineer()
    actual = engineer._compute_joshua_tensor(df)[0]

    semiaxes = np.sqrt(5.0 * np.array([0.8, 0.2, 0.05]))
    diagonal = np.diag(-2.0 * np.log(semiaxes))
    q = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    expected_tensor = q.T @ diagonal @ q
    expected = np.array(
        [
            expected_tensor[0, 0],
            expected_tensor[1, 1],
            expected_tensor[2, 2],
            np.sqrt(2.0) * expected_tensor[0, 1],
            np.sqrt(2.0) * expected_tensor[0, 2],
            np.sqrt(2.0) * expected_tensor[1, 2],
        ]
    )

    np.testing.assert_allclose(actual, expected, atol=1.0e-12)


def test_tomofab_converter_writes_equivalent_ellipsoid_semiaxes():
    df = _geometry_rows(['A'], [1.0e-3]).reset_index(drop=True)
    df['index'] = [1]
    df['BaryCenterX (mm) '] = [0.0]
    df['BaryCenterY (mm) '] = [0.0]
    df['BaryCenterZ (mm) '] = [0.0]

    converted = convert_dataframe(df)

    np.testing.assert_allclose(
        converted[
            ['PEllipsoid Rad1 (mm)', 'PEllipsoid Rad2 (mm)', 'PEllipsoid Rad3 (mm)']
        ].to_numpy()[0],
        [2.0, 1.0, 0.5],
    )


def test_pprime_matches_manuscript_mean_of_log_magnitudes():
    covariance_eigenvalues = np.array([5.5, 3.0, 1.25])
    semiaxes = np.sqrt(5.0 * covariance_eigenvalues)
    f = np.log(semiaxes)
    expected = np.exp(np.sqrt(2.0 * np.sum((f - f.mean()) ** 2)))

    _, bootstrap_pprime = calculate_T_Pprime_from_vals(semiaxes)
    _, boxplot_pprime = compute_fabric_params(semiaxes)

    assert bootstrap_pprime == pytest.approx(expected)
    assert boxplot_pprime == pytest.approx(expected)
