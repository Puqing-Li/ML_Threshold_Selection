#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application controller: minimal GUI class that delegates all logic to modules.
"""

from __future__ import annotations

import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional
import numpy as np
import pandas as pd

from src.ml_threshold_selection.ui_layout import build_main_ui
from src.ml_threshold_selection.data_io import (
    load_multiple_training_data as io_load_multiple_training_data,
    input_expert_thresholds as io_input_expert_thresholds,
    load_test_data as io_load_test_data,
    load_test_data_multiple as io_load_test_data_multiple,
    input_voxel_sizes as io_input_voxel_sizes,
    validate_training_data as io_validate_training_data,
    load_file as io_load_file,
    derive_test_sample_id,
)
from src.ml_threshold_selection.ui_visualization import (
    show_training_visualization as ui_show_training,
    show_prediction_visualization as ui_show_prediction,
    save_chart as ui_save_chart,
)
from src.ml_threshold_selection.labeling import generate_labels_from_thresholds as gen_labels_from_thresholds
from src.ml_threshold_selection.training_pipeline import train_model_pipeline
from src.ml_threshold_selection.io_persistence import auto_save as persist_auto_save, load_last as persist_load_last
from src.ml_threshold_selection.prediction_analysis import compute_dual_thresholds
from src.ml_threshold_selection.export_results import export_filtered_results, export_threshold_report
from src.ml_threshold_selection.feature_utils import extract_simple_features as util_extract_simple_features
from src.ml_threshold_selection.fabric_logging import UILogger
from src.ml_threshold_selection.fabric_pipeline import run_fabric_boxplots
from src.ml_threshold_selection.mean_fabric_calculator import (
    compute_mean_fabric_single,
    export_mean_fabric_txt,
    format_mean_fabric_for_display
)

# Optional project modules
try:
    from ml_threshold_selection.feature_engineering import FeatureEngineer
    from ml_threshold_selection.threshold_finder import AdaptiveThresholdFinder
    from ml_threshold_selection.semi_supervised_learner import SemiSupervisedThresholdLearner
    FULL_MODULES_AVAILABLE = True
except Exception:
    FULL_MODULES_AVAILABLE = False

try:
    import lightgbm as lgb  # noqa: F401
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

from src.analysis.ellipsoid_feature_analyzer import JoshuaFeatureAnalyzer as EllipsoidFeatureAnalyzer
from src.features.ellipsoid_feature_engineering import JoshuaFeatureEngineerFixed as EllipsoidFeatureEngineer
from src.features.res_aware_feature_engineering import ResolutionAwareFeatureEngineer


class FixedMLGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ML Threshold Selection - Enhanced Version")
        self.root.geometry("1400x900")

        # Expose availability flags for UI
        self.LIGHTGBM_AVAILABLE = LIGHTGBM_AVAILABLE
        self.FULL_MODULES_AVAILABLE = FULL_MODULES_AVAILABLE

        # State
        self.model = None
        self.feature_engineer = FeatureEngineer() if FULL_MODULES_AVAILABLE else None
        self.threshold_finder = AdaptiveThresholdFinder() if FULL_MODULES_AVAILABLE else None
        self.training_data = None
        self.test_data = None
        self.features = None
        self.probabilities = None
        self.training_files = []
        self.expert_thresholds = {}
        self.sample_list = []
        self.threshold_input_window = None
        self.training_results = None
        self.visualization_window = None
        self.ellipsoid_feature_analyzer = EllipsoidFeatureAnalyzer()
        self.ellipsoid_feature_engineer = EllipsoidFeatureEngineer()
        self.resolution_aware_engineer = ResolutionAwareFeatureEngineer()
        self.ellipsoid_analysis_results = None
        self.voxel_sizes = {}
        # Configuration parameters
        try:
            from config.config import STRICT_PROBABILITY_THRESHOLD
            self.strict_probability_threshold = STRICT_PROBABILITY_THRESHOLD
        except ImportError:
            self.strict_probability_threshold = 0.01  # Default fallback

        # Build UI
        build_main_ui(self)

    # Basic logging to UI
    def log(self, message: str):
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
        self.root.update()

    # IO delegates
    def load_multiple_training_data(self):
        io_load_multiple_training_data(self)

    def validate_training_data(self, df):
        return io_validate_training_data(self, df)

    def input_expert_thresholds(self):
        io_input_expert_thresholds(self)

    def load_test_data(self):
        io_load_test_data(self)

    def load_test_data_multiple(self):
        """Load multiple test data files and run full analysis for each."""
        filepaths = io_load_test_data_multiple(self)
        if not filepaths:
            return
        if self.model is None:
            self.log("❌ Please train or load a model before running multi-test analysis")
            return
        # Derive sample IDs from filenames (e.g. Quantity_LE01.xlsx -> LE01)
        sample_ids = [derive_test_sample_id(p) for p in filepaths]
        # Configure voxel sizes for all test samples in a single window
        confirmed = self.configure_test_voxel_sizes(sample_ids)
        if not confirmed:
            self.log("⚠️ Multi-sample test analysis cancelled (voxel sizes not confirmed)")
            return
        self.log("")
        self.log("════════════════════════════════════════════════════════════")
        self.log(f"🔄 Starting multi-sample test analysis for {len(filepaths)} files")
        self.log("════════════════════════════════════════════════════════════")
        for filepath in filepaths:
            try:
                df = io_load_file(self, filepath)
                if df is None:
                    continue
                self.test_data = df
                self.test_file_path = filepath
                sample_id = derive_test_sample_id(filepath)
                if sample_id not in self.voxel_sizes:
                    self.log(f"⚠️ Skipping sample {sample_id}: voxel size not set")
                    continue
                self.log("")
                self.log(f"==== Sample {sample_id} ====")
                self.run_full_pipeline_for_current_test_sample()
            except Exception as e:
                self.log(f"❌ Failed to process file {filepath}: {e}")
                import traceback
                self.log(f"Traceback: {traceback.format_exc()}")
        self.log("")
        self.log("✅ Multi-sample test analysis complete.")
        self.log("Check outputs directory for per-sample results.")

    def input_voxel_sizes(self):
        io_input_voxel_sizes(self)

    # Labeling
    def generate_labels_from_thresholds(self):
        if self.training_data is None:
            self.log("❌ Please load training data first")
            return
        self.training_data = gen_labels_from_thresholds(
            training_data=self.training_data,
            expert_thresholds=self.expert_thresholds,
            voxel_sizes=self.voxel_sizes,
            sample_list=self.sample_list,
            log=self.log,
        )

    # Training
    def train_model(self):
        if self.training_data is None:
            self.log("❌ Please load training data first")
            return
        if not self.expert_thresholds:
            self.log("❌ Please enter expert thresholds first")
            return
        try:
            self.log("🔄 Training model...")
            self.root.update()
            self.generate_labels_from_thresholds()
            model, features, training_results = train_model_pipeline(
                training_data=self.training_data,
                voxel_sizes=self.voxel_sizes,
                resolution_aware_engineer=self.resolution_aware_engineer,
                lightgbm_available=LIGHTGBM_AVAILABLE,
            )
            self.model = model
            self.features = features
            self.training_results = training_results
            self.log("✅ Training complete!")
            self.log(f"   - Num features: {len(self.features.columns)}")
            self.log(f"   - Num samples: {len(self.training_results['X'])}")
            self.log(f"   - Train AUC: {self.training_results['train_auc']:.3f}")
            self.log(f"   - Train accuracy: {self.training_results['train_accuracy']:.3f}")
            self.log(f"   - Precision: {self.training_results['precision']:.3f}")
            self.log(f"   - Recall: {self.training_results['recall']:.3f}")
            self.log(f"   - F1 score: {self.training_results['f1']:.3f}")
            # Auto-save
            persist_auto_save(
                model=self.model,
                training_data=self.training_data,
                expert_thresholds=self.expert_thresholds,
                voxel_sizes=self.voxel_sizes,
                training_files=self.training_files,
                features=self.features,
                training_results=self.training_results,
                ellipsoid_analysis_results=self.ellipsoid_analysis_results,
                resolution_aware_engineer=self.resolution_aware_engineer,
                outputs_dir='trained model'
            )
            self.log("💾 Model automatically saved to 'trained model' folder for next session")
        except Exception as e:
            self.log(f"❌ Training failed: {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")

    # Prediction
    def predict_analysis(self):
        if self.test_data is None:
            self.log("❌ Please load test data first")
            return
        if self.model is None:
            self.log("❌ Please train the model first")
            return
        try:
            self.log("🔄 Starting prediction analysis...")
            if not self.voxel_sizes:
                self.log("❌ Please input voxel sizes first (mm)")
                return
            first_sample = self.sample_list[0] if self.sample_list else list(self.voxel_sizes.keys())[0]
            voxel_mm = float(self.voxel_sizes[first_sample])
            test_features = self.resolution_aware_engineer.extract(self.test_data, voxel_size_mm=voxel_mm, fit_scaler=False)
            self.log(f"🔧 Resolution-aware prediction features: {test_features.shape[1]}")
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(test_features.values)[:, 1]
            else:
                probabilities = self.model.predict(test_features.values)
            volumes = self.test_data['Volume3d (mm^3) '].values
            voxel_vol = voxel_mm ** 3
            voxels_cont = volumes / voxel_vol
            inflection_threshold, noise_removal_threshold = compute_dual_thresholds(
                voxels_cont, probabilities, self.strict_probability_threshold
            )
            self.log("✅ Prediction analysis complete!")
            self.log(f"   - Total particles: {len(volumes)}")
            if inflection_threshold is not None:
                self.log(f"   - Loose threshold (Inflection): {int(np.ceil(inflection_threshold))} vox | {(int(np.ceil(inflection_threshold))*voxel_vol):.2e} mm³")
            if noise_removal_threshold is not None:
                self.log(f"   - Strict threshold (P>{self.strict_probability_threshold}): {int(np.ceil(noise_removal_threshold))} vox | {(int(np.ceil(noise_removal_threshold))*voxel_vol):.2e} mm³")
            self.probabilities = probabilities
            self.loose_threshold_vox = int(np.ceil(inflection_threshold)) if inflection_threshold is not None else None
            self.strict_threshold_vox = int(np.ceil(noise_removal_threshold)) if noise_removal_threshold is not None else None
            self.test_voxel_size_mm = voxel_mm
        except Exception as e:
            self.log(f"❌ Prediction analysis failed: {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
            self.loose_threshold_vox = None
            self.strict_threshold_vox = None
            self.test_voxel_size_mm = None

    def ensure_voxel_size_for_sample(self, sample_id: str) -> Optional[float]:
        """Ensure voxel size exists for the given sample; prompt user if missing."""
        if sample_id in self.voxel_sizes:
            voxel_size = float(self.voxel_sizes[sample_id])
            self.log(f"ℹ️ Using existing voxel size for {sample_id}: {voxel_size} mm")
            return voxel_size

        # Prompt user for voxel size (blocking dialog)
        self.log("📏 Please input voxel size for test data (mm/voxel):")
        self.log("   Example: 0.03 means each voxel edge length is 0.03mm")
        self.log("   If unknown, you can use 0.03 as default value")

        voxel_window = tk.Toplevel(self.root)
        voxel_window.title(f"Input Voxel Size - {sample_id}")
        voxel_window.geometry("400x200")
        voxel_window.transient(self.root)
        voxel_window.grab_set()

        tk.Label(
            voxel_window,
            text=f"Voxel size for test data: {sample_id}",
            font=("Arial", 12, "bold"),
        ).pack(pady=10)
        tk.Label(
            voxel_window,
            text="Voxel size (mm/voxel):",
            font=("Arial", 10),
        ).pack(pady=5)
        voxel_entry = tk.Entry(voxel_window, font=("Arial", 10), width=20)
        voxel_entry.pack(pady=5)
        voxel_entry.insert(0, "0.03")

        result = {"value": None}

        def save_voxel_size():
            try:
                voxel_size = float(voxel_entry.get())
                if voxel_size <= 0:
                    messagebox.showerror("Error", "Voxel size must be greater than 0")
                    return
                self.voxel_sizes[sample_id] = voxel_size
                result["value"] = voxel_size
                self.log(f"✅ Test data voxel size: {sample_id} = {voxel_size} mm")
                voxel_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number")

        tk.Button(
            voxel_window,
            text="Save",
            command=save_voxel_size,
            font=("Arial", 10),
            width=10,
        ).pack(pady=10)

        self.root.wait_window(voxel_window)
        return result["value"]

    def configure_test_voxel_sizes(self, sample_ids):
        """Configure voxel sizes for a batch of test samples, with XLSX import/export."""
        from tkinter import filedialog

        # Unique, stable order
        unique_ids = []
        seen = set()
        for sid in sample_ids:
            if sid not in seen:
                seen.add(sid)
                unique_ids.append(sid)

        dialog = tk.Toplevel(self.root)
        dialog.title("Configure Voxel Sizes for Test Samples")
        dialog.geometry("700x550")
        dialog.transient(self.root)
        dialog.grab_set()

        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        title_label = ttk.Label(
            main_frame,
            text="Configure Voxel Sizes for Test Samples",
            font=("Helvetica", 15, "bold"),
        )
        title_label.pack(pady=(0, 8))

        info_label = ttk.Label(
            main_frame,
            text=(
                "Enter voxel size (mm/voxel) for each test sample.\n"
                "You can also Load from an XLSX file with columns:\n"
                "SampleID, Volcano, VoxelSize_mm."
            ),
            font=("Helvetica", 10),
            justify=tk.LEFT,
        )
        info_label.pack(pady=(0, 10), anchor=tk.W)

        table_frame = ttk.Frame(main_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("Sample ID", "Volcano", "Voxel Size (mm)")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)
        for col in columns:
            tree.heading(col, text=col)
        tree.column("Sample ID", width=220, anchor=tk.W)
        tree.column("Volcano", width=180, anchor=tk.W)
        tree.column("Voxel Size (mm)", width=140, anchor=tk.E)

        scrollbar_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar_y.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        # Populate initial rows
        for sid in unique_ids:
            existing_voxel = self.voxel_sizes.get(sid, 0.03)
            tree.insert("", "end", values=(sid, "", existing_voxel))

        def edit_cell(event):
            item = tree.identify_row(event.y)
            column = tree.identify_column(event.x)
            if not item or column != "#3":
                return
            x, y, width, height = tree.bbox(item, column)
            if width == 0:
                return
            value = tree.set(item, column)
            entry = ttk.Entry(tree)
            entry.insert(0, value)
            entry.select_range(0, tk.END)

            def save_edit(event=None):
                new_val = entry.get()
                vals = list(tree.item(item, "values"))
                vals[2] = new_val
                tree.item(item, values=vals)
                entry.destroy()

            def cancel_edit(event=None):
                entry.destroy()

            entry.bind("<Return>", save_edit)
            entry.bind("<Escape>", cancel_edit)
            entry.bind("<FocusOut>", save_edit)
            entry.place(x=x, y=y, width=width, height=height)
            entry.focus_set()

        tree.bind("<Double-1>", edit_cell)

        result = {"confirmed": False}

        def save_and_close():
            try:
                new_voxels = {}
                for item in tree.get_children():
                    sid, volcano, voxel_str = tree.item(item, "values")
                    voxel_str = str(voxel_str).strip()
                    if not voxel_str:
                        messagebox.showerror(
                            "Invalid voxel size",
                            f"Voxel size for sample '{sid}' cannot be empty.",
                            parent=dialog,
                        )
                        return
                    voxel_val = float(voxel_str)
                    if voxel_val <= 0:
                        messagebox.showerror(
                            "Invalid voxel size",
                            f"Voxel size for sample '{sid}' must be greater than 0.",
                            parent=dialog,
                        )
                        return
                    new_voxels[sid] = voxel_val
                # All valid, commit to self.voxel_sizes
                self.voxel_sizes.update(new_voxels)
                self.log(f"✅ Saved voxel sizes for {len(new_voxels)} test samples")
                for sid in unique_ids:
                    if sid in self.voxel_sizes:
                        self.log(f"   - {sid}: {self.voxel_sizes[sid]:.4f} mm")
                result["confirmed"] = True
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror(
                    "Invalid voxel size",
                    f"Please enter valid numbers for voxel sizes.\nError: {e}",
                    parent=dialog,
                )

        def export_to_xlsx():
            try:
                path = filedialog.asksaveasfilename(
                    parent=dialog,
                    title="Save voxel sizes to XLSX",
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                )
                if not path:
                    return
                rows = []
                for item in tree.get_children():
                    sid, volcano, voxel_str = tree.item(item, "values")
                    voxel_str = str(voxel_str).strip()
                    voxel_val = float(voxel_str) if voxel_str else None
                    rows.append(
                        {
                            "SampleID": sid,
                            "Volcano": volcano,
                            "VoxelSize_mm": voxel_val,
                        }
                    )
                df = pd.DataFrame(rows)
                df.to_excel(path, index=False)
                self.log(f"✅ Test voxel size template saved to: {path}")
            except Exception as e:
                self.log(f"❌ Failed to save voxel sizes XLSX: {e}")
                import traceback
                self.log(f"Traceback: {traceback.format_exc()}")
                messagebox.showerror(
                    "Save failed",
                    f"Failed to save voxel sizes XLSX.\n\n{e}",
                    parent=dialog,
                )

        def load_from_xlsx():
            try:
                path = filedialog.askopenfilename(
                    parent=dialog,
                    title="Load voxel sizes from XLSX",
                    filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
                )
                if not path:
                    return
                df = pd.read_excel(path)
                required_cols = {"SampleID", "Volcano", "VoxelSize_mm"}
                if not required_cols.issubset(set(df.columns)):
                    self.log(
                        f"❌ Invalid voxel size XLSX. Required columns: {sorted(list(required_cols))}"
                    )
                    messagebox.showerror(
                        "Invalid format",
                        "The selected XLSX must contain columns:\n"
                        "SampleID, Volcano, VoxelSize_mm",
                        parent=dialog,
                    )
                    return
                mapping = {
                    str(row["SampleID"]): (row["Volcano"], row["VoxelSize_mm"])
                    for _, row in df.iterrows()
                }
                updated = 0
                missing = []
                for item in tree.get_children():
                    sid, volcano, voxel_str = tree.item(item, "values")
                    if sid in mapping:
                        vol, vox = mapping[sid]
                        vals = [sid, vol, vox]
                        tree.item(item, values=vals)
                        updated += 1
                    else:
                        missing.append(sid)
                self.log(
                    f"✅ Loaded voxel sizes from XLSX for {updated} test samples (file: {os.path.basename(path)})"
                )
                if missing:
                    self.log(
                        f"⚠️ {len(missing)} samples not found in XLSX: {missing}"
                    )
            except Exception as e:
                self.log(f"❌ Failed to load voxel sizes XLSX: {e}")
                import traceback
                self.log(f"Traceback: {traceback.format_exc()}")
                messagebox.showerror(
                    "Load failed",
                    f"Failed to load voxel sizes from XLSX.\n\n{e}",
                    parent=dialog,
                )

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(12, 0))

        ttk.Button(
            button_frame, text="Save & Close", width=15, command=save_and_close
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame, text="Save to XLSX", width=15, command=export_to_xlsx
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame, text="Load from XLSX", width=15, command=load_from_xlsx
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame, text="Cancel", width=15, command=dialog.destroy
        ).pack(side=tk.RIGHT, padx=5)

        self.root.wait_window(dialog)
        return result["confirmed"]

    def run_full_pipeline_for_current_test_sample(self):
        """Run prediction, mean fabric, and export for the current test_data."""
        if self.test_data is None:
            self.log("❌ No test data loaded for current sample")
            return
        try:
            self.predict_analysis()
            if (
                not hasattr(self, "loose_threshold_vox")
                or not hasattr(self, "strict_threshold_vox")
                or self.loose_threshold_vox is None
                or self.strict_threshold_vox is None
            ):
                self.log("⚠️ Thresholds not available; skipping mean fabric and export")
                return
            self.calculate_mean_fabric()
            self.export_results()
        except Exception as e:
            self.log(f"❌ Full pipeline failed for current sample: {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")

    # Visualization
    def show_training_visualization(self):
        ui_show_training(self)

    def show_prediction_visualization(self):
        ui_show_prediction(self)

    def save_chart(self, fig, base_name, format_type):
        try:
            ui_save_chart(fig, base_name, format_type, self.log)
        except Exception as e:
            self.log(f"❌ Chart save failed: {e}")

    # Help / User Guide
    def open_user_guide(self):
        try:
            import webbrowser
            from pathlib import Path
            root_dir = Path(__file__).resolve().parents[2]
            candidates = [
                root_dir / 'docs' / 'USER_GUIDE_MODEL_AND_FEATURES_EN.md',
                root_dir / 'docs' / 'user_guide.md',
            ]
            guide = next((p for p in candidates if p.exists()), None)
            if guide:
                webbrowser.open(guide.as_uri())
                self.log(f"📖 Opened User Guide: {guide.name}")
            else:
                msg = "⚠️ User Guide not found under docs/. Please add USER_GUIDE_MODEL_AND_FEATURES_EN.md or user_guide.md."
                self.log(msg)
                messagebox.showinfo("User Guide", msg)
        except Exception as e:
            self.log(f"❌ Failed to open User Guide: {e}")

    def configure_strict_threshold(self):
        """Configure the strict probability threshold (P>threshold)"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configure Strict Threshold")
        dialog.geometry("450x250")
        dialog.grab_set()
        dialog.resizable(False, False)

        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = ttk.Label(main_frame, text="Configure Strict Threshold", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10))

        info_label = ttk.Label(main_frame, text="Set the probability threshold for strict filtering (P > threshold)", font=("Arial", 10))
        info_label.pack(pady=(0, 15))

        threshold_frame = ttk.Frame(main_frame)
        threshold_frame.pack(pady=(0, 20))
        
        ttk.Label(threshold_frame, text="Probability Threshold:").pack(side=tk.LEFT, padx=(0, 10))
        threshold_var = tk.StringVar(value=str(self.strict_probability_threshold))
        threshold_entry = ttk.Entry(threshold_frame, textvariable=threshold_var, width=15)
        threshold_entry.pack(side=tk.LEFT, padx=(0, 10))
        threshold_entry.focus()

        # Add some spacing
        ttk.Label(main_frame, text="").pack(pady=5)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(10, 0))
        
        def save_threshold():
            try:
                new_threshold = float(threshold_var.get())
                if 0.0 <= new_threshold <= 1.0:
                    self.strict_probability_threshold = new_threshold
                    self.log(f"✅ Strict probability threshold updated to: {new_threshold}")
                    dialog.destroy()
                else:
                    self.log("❌ Threshold must be between 0.0 and 1.0")
            except ValueError:
                self.log("❌ Invalid threshold value. Please enter a number between 0.0 and 1.0")

        ttk.Button(button_frame, text="Save", command=save_threshold, width=15).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy, width=15).pack(side=tk.LEFT, padx=(0, 10))
        
        # Add keyboard shortcuts
        dialog.bind('<Return>', lambda e: save_threshold())
        dialog.bind('<Escape>', lambda e: dialog.destroy())

    # Config loader for thresholds & voxel sizes
    def load_thresholds_config(self):
        try:
            from tkinter import filedialog
            import pandas as pd
            path = filedialog.askopenfilename(title='Select thresholds_voxels.csv', filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
            if not path:
                return
            df = pd.read_csv(path)
            required = {'SampleID', 'ExpertThreshold_mm3', 'VoxelSize_mm'}
            if not required.issubset(set(df.columns)):
                self.log(f"❌ Invalid config. Required columns: {sorted(list(required))}")
                return
            # Update in-memory config
            self.expert_thresholds = {str(r['SampleID']): float(r['ExpertThreshold_mm3']) for _, r in df.iterrows()}
            self.voxel_sizes = {str(r['SampleID']): float(r['VoxelSize_mm']) for _, r in df.iterrows()}
            self.sample_list = sorted(list(self.voxel_sizes.keys()))
            self.log(f"✅ Loaded thresholds & voxels for {len(self.sample_list)} samples")
            for sid in self.sample_list:
                self.log(f"   - {sid}: threshold={self.expert_thresholds.get(sid, 'NA'):.2e} mm³, voxel={self.voxel_sizes.get(sid, 'NA'):.4f} mm")
        except Exception as e:
            self.log(f"❌ Failed to load thresholds config: {e}")

    # Fabric boxplots
    def generate_fabric_boxplots(self):
        try:
            if self.test_data is None:
                self.log("❌ Please load test data first")
                return
            required_cols = [
                'EigenVal1', 'EigenVal2', 'EigenVal3',
                'EigenVec1X', 'EigenVec1Y', 'EigenVec1Z',
                'EigenVec2X', 'EigenVec2Y', 'EigenVec2Z',
                'EigenVec3X', 'EigenVec3Y', 'EigenVec3Z',
            ]
            for c in required_cols:
                if c not in self.test_data.columns:
                    self.log(f"❌ Missing required column for fabric analysis: {c}")
                    return
            if not hasattr(self, 'loose_threshold_vox') or not hasattr(self, 'strict_threshold_vox'):
                self.log("❌ Please perform prediction analysis first to get thresholds")
                return
            if self.loose_threshold_vox is None or self.strict_threshold_vox is None:
                self.log("❌ Thresholds are not ready. Run prediction analysis again.")
                return
            if not self.voxel_sizes:
                self.log("❌ Please input voxel sizes first (mm)")
                return
            first_sample = self.sample_list[0] if self.sample_list else list(self.voxel_sizes.keys())[0]
            voxel_mm = float(self.voxel_sizes[first_sample])
            logger = UILogger(self.log)
            run_fabric_boxplots(
                df=self.test_data,
                voxel_size_mm=voxel_mm,
                loose_threshold_vox=self.loose_threshold_vox,
                strict_threshold_vox=self.strict_threshold_vox,
                logger=logger,
                outputs_dir='outputs',
                n_bootstrap=1000,
                min_particles=50,
            )
        except Exception as e:
            self.log(f"❌ Fabric boxplots failed: {e}")
            import traceback
            self.log(f"Detailed error: {traceback.format_exc()}")

    # Export
    def export_results(self):
        if self.test_data is None or self.probabilities is None:
            self.log("❌ Please perform prediction analysis first")
            return
        try:
            if not hasattr(self, 'loose_threshold_vox') or not hasattr(self, 'strict_threshold_vox'):
                self.log("❌ Please perform prediction analysis to get thresholds first")
                return
            if self.loose_threshold_vox is None or self.strict_threshold_vox is None:
                self.log("❌ Thresholds not calculated properly. Please run prediction analysis again.")
                return
            if not hasattr(self, 'test_voxel_size_mm') or self.test_voxel_size_mm is None:
                self.log("❌ Test voxel size not available")
                return
            voxel_size_mm = self.test_voxel_size_mm
            # Derive sample ID for naming outputs
            sample_id = "TestSample"
            if hasattr(self, 'test_file_path') and self.test_file_path:
                sample_id = derive_test_sample_id(self.test_file_path)
            elif self.sample_list:
                sample_id = self.sample_list[0]
            loose_file, strict_file = export_filtered_results(
                results_df=self.test_data,
                probabilities=self.probabilities,
                loose_threshold_vox=self.loose_threshold_vox,
                strict_threshold_vox=self.strict_threshold_vox,
                voxel_size_mm=voxel_size_mm,
                outputs_dir='outputs',
                strict_probability_threshold=self.strict_probability_threshold,
                sample_id=sample_id,
            )
            voxel_vol = voxel_size_mm ** 3
            loose_threshold_mm = self.loose_threshold_vox * voxel_vol
            strict_threshold_mm = self.strict_threshold_vox * voxel_vol
            loose_kept = int((self.test_data['Volume3d (mm^3) '] >= loose_threshold_mm).sum())
            strict_kept = int((self.test_data['Volume3d (mm^3) '] >= strict_threshold_mm).sum())
            report_filename = f"outputs/{sample_id}_Threshold_Report_{self.loose_threshold_vox:.0f}vox_{self.strict_threshold_vox:.0f}vox.txt"
            export_threshold_report(
                out_path=report_filename,
                total_rows=len(self.test_data),
                voxel_size_mm=voxel_size_mm,
                loose_threshold_vox=self.loose_threshold_vox,
                strict_threshold_vox=self.strict_threshold_vox,
                loose_threshold_mm=loose_threshold_mm,
                strict_threshold_mm=strict_threshold_mm,
                loose_kept=loose_kept,
                strict_kept=strict_kept,
            )
            self.log(f"✅ Loose threshold results exported to: {loose_file}")
            self.log(f"✅ Strict threshold results exported to: {strict_file}")
            self.log(f"✅ Threshold report exported to: {report_filename}")
            self.log("📊 Export complete: 2 XLSX files + 1 TXT report generated")
        except Exception as e:
            self.log(f"❌ Export results failed: {e}")
            import traceback
            self.log(f"Detailed error: {traceback.format_exc()}")

    # Analysis
    def analyze_ellipsoid_features(self):
        from src.ml_threshold_selection.analysis_pipeline import run_feature_analysis
        run_feature_analysis(self)

    def display_ellipsoid_analysis_results(self):
        from src.ml_threshold_selection.analysis_pipeline import display_feature_analysis_results
        display_feature_analysis_results(self)

    def extract_simple_features(self, df):
        return util_extract_simple_features(df)

    # Mean Fabric Calculation
    def calculate_mean_fabric(self):
        """Calculate mean fabric tensor for both loose and strict thresholds"""
        if self.test_data is None:
            self.log("❌ Please load test data first")
            return
        if not hasattr(self, 'loose_threshold_vox') or not hasattr(self, 'strict_threshold_vox'):
            self.log("❌ Please perform prediction analysis first to get thresholds")
            return
        if self.loose_threshold_vox is None or self.strict_threshold_vox is None:
            self.log("❌ Thresholds not calculated. Please run prediction analysis first.")
            return
        if not hasattr(self, 'test_voxel_size_mm') or self.test_voxel_size_mm is None:
            self.log("❌ Test voxel size not available")
            return
        
        try:
            self.log("🔄 Calculating mean fabric tensors...")
            voxel_size_mm = self.test_voxel_size_mm
            voxel_vol = voxel_size_mm ** 3
            
            # Calculate thresholds in mm³
            loose_threshold_mm3 = self.loose_threshold_vox * voxel_vol
            strict_threshold_mm3 = self.strict_threshold_vox * voxel_vol
            
            # Get sample ID (from test data filename or first sample)
            sample_id = "TestSample"
            if hasattr(self, 'test_file_path') and self.test_file_path:
                sample_id = derive_test_sample_id(self.test_file_path)
            elif self.sample_list:
                sample_id = self.sample_list[0]
            
            # Calculate for Loose threshold
            self.log(f"   Calculating for Loose threshold ({loose_threshold_mm3:.6e} mm³)...")
            loose_mean, loose_evecs, loose_evals, loose_T, loose_P, loose_n = compute_mean_fabric_single(
                self.test_data, loose_threshold_mm3
            )
            
            if loose_mean is None:
                self.log(f"   ⚠️ Loose threshold: Insufficient particles ({loose_n})")
            else:
                loose_file = export_mean_fabric_txt(
                    sample_id=sample_id,
                    threshold_type="Loose",
                    volume_threshold_mm3=loose_threshold_mm3,
                    mean_ellipsoid_matrix=loose_mean,
                    eigenvectors=loose_evecs,
                    eigenvalues=loose_evals,
                    T=loose_T,
                    P_prime=loose_P,
                    n_particles=loose_n,
                    output_dir='outputs'
                )
                self.log(f"   ✅ Loose threshold mean fabric saved: {loose_file}")
                self.log(f"      - Particles: {loose_n}, T: {loose_T:.6f}, P': {loose_P:.6f}")
                
                # Display results in GUI
                self.log("")
                self.log("=" * 60)
                self.log("LOOSE THRESHOLD - Mean Fabric Results:")
                self.log("=" * 60)
                loose_display_text = format_mean_fabric_for_display(
                    sample_id=sample_id,
                    threshold_type="Loose",
                    volume_threshold_mm3=loose_threshold_mm3,
                    mean_ellipsoid_matrix=loose_mean,
                    eigenvectors=loose_evecs,
                    eigenvalues=loose_evals,
                    T=loose_T,
                    P_prime=loose_P,
                    n_particles=loose_n
                )
                self.log(loose_display_text)
                self.log("")
            
            # Calculate for Strict threshold
            self.log(f"   Calculating for Strict threshold ({strict_threshold_mm3:.6e} mm³)...")
            strict_mean, strict_evecs, strict_evals, strict_T, strict_P, strict_n = compute_mean_fabric_single(
                self.test_data, strict_threshold_mm3
            )
            
            if strict_mean is None:
                self.log(f"   ⚠️ Strict threshold: Insufficient particles ({strict_n})")
            else:
                strict_file = export_mean_fabric_txt(
                    sample_id=sample_id,
                    threshold_type="Strict",
                    volume_threshold_mm3=strict_threshold_mm3,
                    mean_ellipsoid_matrix=strict_mean,
                    eigenvectors=strict_evecs,
                    eigenvalues=strict_evals,
                    T=strict_T,
                    P_prime=strict_P,
                    n_particles=strict_n,
                    output_dir='outputs'
                )
                self.log(f"   ✅ Strict threshold mean fabric saved: {strict_file}")
                self.log(f"      - Particles: {strict_n}, T: {strict_T:.6f}, P': {strict_P:.6f}")
                
                # Display results in GUI
                self.log("")
                self.log("=" * 60)
                self.log("STRICT THRESHOLD - Mean Fabric Results:")
                self.log("=" * 60)
                strict_display_text = format_mean_fabric_for_display(
                    sample_id=sample_id,
                    threshold_type="Strict",
                    volume_threshold_mm3=strict_threshold_mm3,
                    mean_ellipsoid_matrix=strict_mean,
                    eigenvectors=strict_evecs,
                    eigenvalues=strict_evals,
                    T=strict_T,
                    P_prime=strict_P,
                    n_particles=strict_n
                )
                self.log(strict_display_text)
                self.log("")
            
            self.log("✅ Mean fabric calculation complete!")
            self.log("📄 Two TXT files generated with mean fabric tensors and eigenvectors")
            self.log("   - Use Eigen1 (column 0) and Eigen2 (column 1) for foliation plane in Avizo")
            self.log("   - Use Eigen1 (column 0) and Eigen3 (column 2) for kinematic plane in Avizo")
            
        except Exception as e:
            self.log(f"❌ Mean fabric calculation failed: {e}")
            import traceback
            self.log(f"Detailed error: {traceback.format_exc()}")

    # Persistence
    def load_last_time_model(self):
        try:
            model_data = persist_load_last('trained model')
            self.model = model_data['model']
            self.training_data = model_data['training_data']
            self.expert_thresholds = model_data['expert_thresholds']
            self.voxel_sizes = model_data['voxel_sizes']
            self.training_files = model_data['training_files']
            self.features = model_data.get('features', None)
            self.training_results = model_data.get('training_results', None)
            self.ellipsoid_analysis_results = model_data.get('ellipsoid_analysis_results', None) or model_data.get('joshua_analysis_results', None)
            if 'resolution_aware_engineer' in model_data:
                self.resolution_aware_engineer = model_data['resolution_aware_engineer']
                self.log("   - Resolution-aware engineer with fitted scaler restored")
            else:
                self.resolution_aware_engineer = ResolutionAwareFeatureEngineer()
                self.log("   - New resolution-aware engineer created (scaler not fitted)")
            self.log("✅ Last time model loaded successfully!")
            self.log(f"   - Training data: {len(self.training_data)} particles")
            self.log(f"   - Expert thresholds: {len(self.expert_thresholds)} samples")
            self.log(f"   - Voxel sizes: {len(self.voxel_sizes)} samples")
            self.log(f"   - Training files: {len(self.training_files)} files")
            self.log("   - Model ready for prediction and visualization")
            self.log("👉 Next step: Load Test Data (Step 6) and run Predict Analysis (Step 7); then Mean Fabric or Export/Reports.")
            try:
                msg_win = tk.Toplevel(self.root)
                msg_win.title("Next Step")
                msg_win.geometry("450x200")
                msg_win.transient(self.root)
                msg_win.grab_set()
                
                content = (
                    "Model loaded. You can proceed:\n\n"
                    "1) Click 'Load Single/Multi Test Data' to load test data\n"
                    "2) Click 'Predict Analysis' to run prediction\n"
                    "3) Click 'Prediction Visualization'\n"
                    "4) Click 'Fabric Boxplots' and click 'Mean Fabric'"
                )
                
                tk.Label(msg_win, text=content, justify=tk.LEFT, font=("Arial", 11)).pack(pady=20, padx=20)
                tk.Button(msg_win, text="OK", width=10, command=msg_win.destroy, font=("Arial", 10)).pack(pady=5)
            except Exception:
                pass
        except Exception as e:
            self.log(f"❌ Load last time model failed: {e}")
            import traceback
            self.log(f"Detailed error: {traceback.format_exc()}")


