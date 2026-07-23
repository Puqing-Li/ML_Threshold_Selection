import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.res_aware_feature_engineering import ResolutionAwareFeatureEngineer
from features.ellipsoid_feature_engineering import JoshuaFeatureEngineerFixed
from ml_threshold_selection.fabric_bootstrap import (
    calculate_T_Pprime_from_vals,
    eigvals_from_logMean,
    precompute_logE_block,
)
from ml_threshold_selection.fabric_boxplots_dual_thresholds import compute_fabric_params
from ml_threshold_selection.labeling import generate_labels_from_thresholds
from ml_threshold_selection.prediction_analysis import compute_dual_thresholds
from tools.To_tomofab import convert_dataframe
from loso_threshold_validation import (
    candidate_threshold_units,
    expert_threshold_units,
    project_decisions_to_vmin,
)


def _rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _frame(eigenvalues, q):
    return pd.DataFrame({
        'Volume3d (mm^3) ': [0.01],
        'EigenVal1': [eigenvalues[0]],
        'EigenVal2': [eigenvalues[1]],
        'EigenVal3': [eigenvalues[2]],
        'EigenVec1X': [q[0, 0]], 'EigenVec1Y': [q[0, 1]], 'EigenVec1Z': [q[0, 2]],
        'EigenVec2X': [q[1, 0]], 'EigenVec2Y': [q[1, 1]], 'EigenVec2Z': [q[1, 2]],
        'EigenVec3X': [q[2, 0]], 'EigenVec3Y': [q[2, 1]], 'EigenVec3Z': [q[2, 2]],
    })


class GeometryDefinitionTests(unittest.TestCase):
    def test_compatibility_feature_engineer_matches_reported_features(self):
        q = np.eye(3)
        df = _frame(np.array([9.0, 4.0, 1.0]), q)
        df['Volume3d (mm^3) '] = 50.5 * 0.03 ** 3

        compatibility = JoshuaFeatureEngineerFixed(voxel_size_um=30.0)
        actual = compatibility.extract_joshua_features(df, fit_scaler=False)
        expected = ResolutionAwareFeatureEngineer()._build_unscaled_features(df, 0.03)

        np.testing.assert_allclose(
            actual.rename(columns={'Volume': 'VoxelCount'}).values,
            expected.values,
        )

    def test_resolution_aware_tensor_is_symmetric_and_rotation_covariant(self):
        eigenvalues = np.array([9.0, 4.0, 1.0])
        q = _rotation_z(np.deg2rad(31.0))
        df = _frame(eigenvalues, q)

        values = ResolutionAwareFeatureEngineer()._compute_joshua_tensor(df)[0]
        actual = np.array([
            [values[0], values[3] / np.sqrt(2.0), values[4] / np.sqrt(2.0)],
            [values[3] / np.sqrt(2.0), values[1], values[5] / np.sqrt(2.0)],
            [values[4] / np.sqrt(2.0), values[5] / np.sqrt(2.0), values[2]],
        ])
        expected = q.T @ np.diag(-np.log(5.0 * eigenvalues)) @ q

        np.testing.assert_allclose(actual, actual.T, atol=1e-12)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_resolution_aware_tensor_is_invariant_to_eigenvector_sign(self):
        eigenvalues = np.array([9.0, 4.0, 1.0])
        q = _rotation_z(np.deg2rad(31.0))
        sign_flipped = q.copy()
        sign_flipped[[0, 2]] *= -1.0

        original = ResolutionAwareFeatureEngineer()._compute_joshua_tensor(
            _frame(eigenvalues, q)
        )
        flipped = ResolutionAwareFeatureEngineer()._compute_joshua_tensor(
            _frame(eigenvalues, sign_flipped)
        )

        np.testing.assert_allclose(original, flipped, atol=1e-12)

    def test_fabric_tensor_uses_avizo_covariance_semiaxes(self):
        eigenvalues = np.array([9.0, 4.0, 1.0])
        q = _rotation_z(np.deg2rad(23.0))
        block = np.hstack([eigenvalues, q.reshape(-1)])[None, :]

        actual = precompute_logE_block(block)[0]
        expected = q.T @ np.diag(
            np.log(np.sqrt(5.0 * eigenvalues) / 1.0)
        ) @ q
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_semiaxis_mean_matches_quadratic_form_ellmean(self):
        first = _frame(np.array([9.0, 4.0, 1.0]), _rotation_z(np.deg2rad(23.0)))
        second = _frame(np.array([6.0, 2.5, 0.8]), _rotation_z(np.deg2rad(-17.0)))
        build = pd.concat([first, second], ignore_index=True)
        log_s = precompute_logE_block(
            build[[
                'EigenVal1', 'EigenVal2', 'EigenVal3',
                'EigenVec1X', 'EigenVec1Y', 'EigenVec1Z',
                'EigenVec2X', 'EigenVec2Y', 'EigenVec2Z',
                'EigenVec3X', 'EigenVec3Y', 'EigenVec3Z',
            ]].to_numpy(float)
        ).mean(axis=0).astype(float)

        s_log_vals, s_vecs = np.linalg.eigh(log_s)
        s_mean = s_vecs @ np.diag(np.exp(s_log_vals)) @ s_vecs.T
        log_e_mean = -2.0 * log_s
        e_log_vals, e_vecs = np.linalg.eigh(log_e_mean)
        e_mean = e_vecs @ np.diag(np.exp(e_log_vals)) @ e_vecs.T

        np.testing.assert_allclose(e_mean, np.linalg.inv(s_mean) @ np.linalg.inv(s_mean), atol=1e-6)
        axes_from_e = np.sort(np.linalg.eigvalsh(e_mean) ** -0.5)[::-1]
        np.testing.assert_allclose(axes_from_e, eigvals_from_logMean(log_s), rtol=1e-6)

    def test_pprime_uses_geometric_log_normalization(self):
        vals = np.array([3.0, 2.0, 1.0])
        logs = np.log(vals)
        expected = np.exp(np.sqrt(2.0 * np.sum((logs - logs.mean()) ** 2)))

        t1, p1 = calculate_T_Pprime_from_vals(vals)
        t2, p2 = compute_fabric_params(vals)
        np.testing.assert_allclose(p1, expected)
        np.testing.assert_allclose(p2, expected)
        np.testing.assert_allclose(t1, t2)

    def test_tomofab_converter_maps_covariance_eigenvalues_to_radii(self):
        q = np.eye(3)
        df = _frame(np.array([9.0, 4.0, 1.0]), q)
        df.insert(0, 'index', [1])
        df['BaryCenterX (mm) '] = 0.1
        df['BaryCenterY (mm) '] = 0.2
        df['BaryCenterZ (mm) '] = 0.3

        converted = convert_dataframe(df)
        np.testing.assert_allclose(
            converted[['PEllipsoid Rad1 (mm)', 'PEllipsoid Rad2 (mm)', 'PEllipsoid Rad3 (mm)']].iloc[0],
            np.sqrt(5.0 * np.array([9.0, 4.0, 1.0])),
        )

    def test_strict_threshold_excludes_the_largest_flagged_boundary_object(self):
        voxels = np.array([10.0, 20.0, 30.0, 31.0])
        probabilities = np.array([0.8, 0.02, 0.011, 0.001])

        _, strict = compute_dual_thresholds(voxels, probabilities, 0.01)

        self.assertEqual(strict, 31.0)
        self.assertTrue(np.all(probabilities[voxels >= strict] <= 0.01))

    def test_threshold_calculation_rejects_empty_inputs(self):
        with self.assertRaises(ValueError):
            compute_dual_thresholds(np.array([]), np.array([]), 0.01)

    def test_training_features_use_each_samples_voxel_size(self):
        q = np.eye(3)
        first = _frame(np.array([9.0, 4.0, 1.0]), q)
        second = _frame(np.array([9.0, 4.0, 1.0]), q)
        first['SampleID'] = 'A'
        second['SampleID'] = 'B'
        combined = pd.concat([first, second], ignore_index=True)
        engineer = ResolutionAwareFeatureEngineer()

        scaled = engineer.extract_by_sample(
            combined,
            voxel_sizes={'A': 0.01, 'B': 0.02},
            fit_scaler=True,
        )
        unscaled = engineer.scaler.inverse_transform(scaled)

        np.testing.assert_allclose(unscaled[:, 0], [10000.0, 1250.0])

    def test_expert_labels_use_exact_physical_volume_threshold(self):
        voxel_size = 0.03
        threshold_mm3 = 0.00134
        data = pd.DataFrame({
            'SampleID': ['A', 'A'],
            'Volume3d (mm^3) ': [49.5 * voxel_size ** 3, 49.8 * voxel_size ** 3],
        })

        labeled = generate_labels_from_thresholds(
            data,
            expert_thresholds={'A': threshold_mm3},
            voxel_sizes={'A': voxel_size},
            sample_list=['A'],
        )

        self.assertEqual(labeled['label'].tolist(), [1, 0])

    def test_projected_vmin_minimizes_binary_decision_disagreement(self):
        voxel_counts = np.array([5.0, 10.0, 20.0, 40.0])
        decisions = np.array([1, 1, 0, 0])

        threshold, disagreement = project_decisions_to_vmin(voxel_counts, decisions)

        self.assertEqual(threshold, 15.0)
        self.assertEqual(disagreement, 0.0)

    def test_expert_threshold_units_keep_mm3_as_source_quantity(self):
        equivalent, boundary = expert_threshold_units(3.9e-3, 0.03)

        self.assertAlmostEqual(equivalent, 144.44444444444446)
        self.assertEqual(boundary, 145)

    def test_candidate_threshold_units_apply_ceiling_before_mm3_conversion(self):
        applied, threshold_mm3 = candidate_threshold_units(40.711955380877356, 0.03)

        self.assertEqual(applied, 41)
        self.assertAlmostEqual(threshold_mm3, 41 * 0.03**3)


if __name__ == '__main__':
    unittest.main()
