#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature analysis pipeline and related helpers, extracted from main.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


def run_feature_analysis(app):
    if app.training_data is None:
        app.log("❌ Please load training data first")
        return
    if not app.expert_thresholds:
        app.log("❌ Please enter expert thresholds first")
        return
    try:
        app.log("🔬 Starting feature analysis...")
        # Labels
        app.generate_labels_from_thresholds()
        df = app.training_data.copy()
        # Remove string columns
        string_columns = []
        for col in df.columns:
            if col in ['SampleID', 'label']:
                continue
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col], errors='raise')
                except Exception:
                    string_columns.append(col)
        if string_columns:
            app.log(f"   Removed string columns: {string_columns}")
            df = df.drop(columns=string_columns)
        labels = df['label'].values
        sample_ids = df['SampleID'].values if 'SampleID' in df.columns else None
        if not app.voxel_sizes:
            app.log("❌ Please input voxel sizes first")
            return
        first_sample_id = list(app.voxel_sizes.keys())[0]
        app.ellipsoid_feature_engineer.voxel_size_mm = float(app.voxel_sizes[first_sample_id])
        app.log(f"🔧 Setting feature analysis voxel size: {app.ellipsoid_feature_engineer.voxel_size_mm:.4f} mm")
        app.ellipsoid_analysis_results = app.ellipsoid_feature_analyzer.analyze_feature_differences(
            df, labels, sample_ids, app.voxel_sizes
        )
        display_feature_analysis_results(app)
        app.log("📊 Feature analysis completed! Generate visualization charts?")
        app.log("   Click '📊 Training Visualization' to view detailed charts")
        app.log("✅ Feature analysis completed")
    except Exception as e:
        app.log(f"❌ Feature analysis failed: {e}")
        import traceback
        app.log(f"Detailed error: {traceback.format_exc()}")


def display_feature_analysis_results(app):
    if app.ellipsoid_analysis_results is None:
        return
    feature_stats = app.ellipsoid_analysis_results['feature_stats']
    features_df = app.ellipsoid_analysis_results['features_df']
    significant = [(n, s) for n, s in feature_stats.items() if s['is_significant']]
    significant.sort(key=lambda x: x[1]['cohens_d'], reverse=True)
    app.log("🎯 Feature analysis significant features (p<0.05, Cohen's d>0.2):")
    for name, stats in significant:
        app.log(f"   - {name}: d={stats['cohens_d']:.3f}, p={stats['p_value']:.2e}")
    app.log(f"🔬 Feature analysis features (total {len(features_df.columns)}):")
    for col in features_df.columns:
        stats = feature_stats[col]
        significance = "significant" if stats['is_significant'] else "not significant"
        app.log(f"   - {col}: d={stats['cohens_d']:.3f}, {significance}")
    feature_descriptions = app.ellipsoid_feature_engineer.get_feature_descriptions()
    app.log("📋 Feature descriptions:")
    for feature, description in feature_descriptions.items():
        app.log(f"   - {feature}: {description}")


def calculate_adaptive_threshold(volumes: np.ndarray, probabilities: np.ndarray, target_artifact_rate: float = 0.05):
    volume_indices = np.argsort(volumes)
    sorted_volumes = volumes[volume_indices]
    sorted_probabilities = probabilities[volume_indices]
    best_threshold = None
    best_error = float('inf')
    percentiles = np.linspace(1, 50, 50)
    for p in percentiles:
        threshold = np.percentile(sorted_volumes, p)
        removed_mask = sorted_volumes < threshold
        if np.sum(removed_mask) > 0:
            actual = float(np.mean(sorted_probabilities[removed_mask]))
            error = abs(actual - target_artifact_rate)
            if error < best_error:
                best_error = error
                best_threshold = threshold
    retained_mask = volumes >= best_threshold
    retained_count = int(np.sum(retained_mask))
    removed_count = int(len(volumes) - retained_count)
    retention_rate = retained_count / float(len(volumes))
    actual_artifact_rate = float(np.mean(probabilities[~retained_mask])) if removed_count > 0 else 0.0
    return {
        'threshold': float(best_threshold),
        'retained_count': retained_count,
        'removed_count': removed_count,
        'retention_rate': retention_rate,
        'target_artifact_rate': target_artifact_rate,
        'actual_artifact_rate': actual_artifact_rate,
        'artifact_rate_error': abs(actual_artifact_rate - target_artifact_rate),
    }


def multi_sample_test(app):
    if app.model is None:
        app.log("❌ Please train or load model first")
        return
    from tkinter import filedialog
    try:
        test_files = filedialog.askopenfilenames(
            title="Select Multiple Test Files",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not test_files:
            return
        app.log(f"🔄 Loading {len(test_files)} test files...")
        results = []
        for i, file_path in enumerate(test_files):
            app.log(f"📁 Processing file {i+1}/{len(test_files)}: {os.path.basename(file_path)}")
            import os
            try:
                test_data = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
                sample_id = os.path.splitext(os.path.basename(file_path))[0]
                test_data['SampleID'] = sample_id
                feature_columns = app.features.columns.tolist() if app.features is not None else app.extract_simple_features(test_data).columns.tolist()
                X_test = test_data[feature_columns].fillna(0)
                if hasattr(app.model, 'predict_proba'):
                    probabilities = app.model.predict_proba(X_test)[:, 1]
                else:
                    probabilities = app.model.predict(X_test)
                volumes = test_data['Volume3d (mm^3) '].values
                threshold_result = calculate_adaptive_threshold(volumes, probabilities)
                results.append({
                    'sample_id': sample_id,
                    'file_path': file_path,
                    'total_particles': len(test_data),
                    'retained_particles': threshold_result['retained_count'],
                    'removed_particles': threshold_result['removed_count'],
                    'retention_rate': threshold_result['retention_rate'],
                    'predicted_threshold': threshold_result['threshold'],
                    'target_artifact_rate': threshold_result['target_artifact_rate'],
                    'actual_artifact_rate': threshold_result['actual_artifact_rate'],
                    'artifact_rate_error': threshold_result['artifact_rate_error'],
                })
                app.log(f"   ✅ {sample_id}: {results[-1]['retained_particles']}/{results[-1]['total_particles']} retained, threshold={results[-1]['predicted_threshold']:.6f}")
            except Exception as e:
                app.log(f"   ❌ Error processing {file_path}: {e}")
                continue
        app.log(f"\n📊 Multi-Sample Test Results Summary:")
        app.log(f"   Total samples processed: {len(results)}")
        if results:
            import numpy as np
            app.log(f"   Average retention rate: {np.mean([r['retention_rate'] for r in results]):.1%}")
            app.log(f"   Average threshold: {np.mean([r['predicted_threshold'] for r in results]):.6f}")
            from tkinter import filedialog
            results_df = pd.DataFrame(results)
            output_path = filedialog.asksaveasfilename(title="Save Multi-Sample Results", defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
            if output_path:
                results_df.to_csv(output_path, index=False)
                app.log(f"✅ Results saved to: {output_path}")
    except Exception as e:
        app.log(f"❌ Multi-sample test failed: {e}")
        import traceback
        app.log(f"Detailed error: {traceback.format_exc()}")


