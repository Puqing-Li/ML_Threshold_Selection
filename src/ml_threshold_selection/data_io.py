#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data I/O and small UI dialogs extracted from main.
All functions accept the app instance and operate via app.log / app fields.
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog


def load_file(app, filepath: str):
    try:
        file_ext = Path(filepath).suffix.lower()
        if file_ext == '.csv':
            df = pd.read_csv(filepath)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        app.log(f"✅ File loaded: {Path(filepath).name}")
        app.log(f"   - Type: {file_ext}")
        app.log(f"   - Rows: {len(df)}")
        app.log(f"   - Columns: {len(df.columns)}")
        return df
    except Exception as e:
        app.log(f"❌ File load failed: {e}")
        return None


def load_multiple_training_data(app):
    filepaths = filedialog.askopenfilenames(
        title="Select multiple training data files",
        filetypes=[
            ("Excel files", "*.xlsx *.xls"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )
    if not filepaths:
        return
    app.log(f"🔄 Loading {len(filepaths)} files...")
    all_data = []
    successful_files = []
    sample_names = set()
    for filepath in filepaths:
        df = load_file(app, filepath)
        if df is None:
            continue
        df['source_file'] = Path(filepath).name
        all_data.append(df)
        successful_files.append(filepath)
        if 'SampleID' in df.columns:
            sample_names.update(df['SampleID'].unique())
        else:
            sample_name = Path(filepath).stem
            sample_names.add(sample_name)
            df['SampleID'] = sample_name
    if not all_data:
        app.log("❌ No files loaded successfully")
        return
    app.training_data = pd.concat(all_data, ignore_index=True)
    app.training_files = successful_files
    app.sample_list = sorted(list(sample_names))
    app.log(f"✅ Batch load complete: {len(app.training_data)} grains")
    app.log(f"📁 Loaded files: {[Path(f).name for f in successful_files]}")
    app.log(f"🔍 Samples: {app.sample_list}")
    validate_training_data(app, app.training_data)


def validate_training_data(app, df: pd.DataFrame) -> bool:
    required_cols = ['Volume3d (mm^3) ', 'EigenVal1', 'EigenVal2', 'EigenVal3']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        app.log(f"⚠️ Missing required columns: {missing_cols}")
        return False
    app.log(f"📋 Columns: {list(df.columns)}")
    return True


def derive_test_sample_id(filepath: str) -> str:
    """
    Derive a sample ID from a test data filepath.

    Examples:
    - 'Quantity_LE01.xlsx' -> 'LE01'
    - 'Quantity_LEU9002.csv' -> 'LEU9002'
    - 'totalAKAN20.xlsx' -> 'totalAKAN20' (no underscore case)
    """
    stem = os.path.splitext(os.path.basename(filepath))[0]
    if "_" in stem:
        return stem.split("_")[-1]
    return stem


def load_test_data_multiple(app):
    """Select multiple test data files (xlsx/xls/csv) and return their paths."""
    filepaths = filedialog.askopenfilenames(
        title="Select Multiple Test Data Files",
        filetypes=[
            ("Excel files", "*.xlsx *.xls"),
            ("CSV files", "*.csv"),
            ("All files", "*.*"),
        ],
    )
    if not filepaths:
        app.log("ℹ️ No test data files selected")
        return []
    app.log(f"🔄 Selected {len(filepaths)} test data files")
    app.log(f"📁 Files: {[Path(f).name for f in filepaths]}")
    return list(filepaths)


def input_expert_thresholds(app):
    if not app.sample_list:
        app.log("❌ Please import data files first")
        return
    app.log("📝 Please input expert thresholds (format: SampleID:Threshold)")
    app.log("Example: Quantity_LE03:1.0e-06")
    create_simple_threshold_input(app)


def create_simple_threshold_input(app):
    dialog = tk.Toplevel(app.root)
    dialog.title("Input Expert Thresholds")
    dialog.geometry("600x400")
    dialog.grab_set()

    main_frame = ttk.Frame(dialog)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    title_label = ttk.Label(main_frame, text="Input Expert Thresholds", font=("Arial", 14, "bold"))
    title_label.pack(pady=10)

    info_label = ttk.Label(main_frame, text="Enter a volume threshold per sample, one per line (SampleID:Threshold)", font=("Arial", 10))
    info_label.pack(pady=5)

    text_frame = ttk.Frame(main_frame)
    text_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    app.threshold_text = tk.Text(text_frame, height=15, width=60, font=("Consolas", 10))
    app.threshold_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=app.threshold_text.yview)
    app.threshold_text.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Default thresholds for known samples
    default_thresholds = {
        'totalAKAN20': 3.9e-03,
        'totalANA16937': 8.0e-04,
        'totalHL19335': 1.0e-03,
        'totalLE03': 1.0e-03,
        'totalLE19': 1.8e-03
    }
    
    example_text = ""
    for sample_id in app.sample_list:
        if sample_id in default_thresholds:
            example_text += f"{sample_id}:{default_thresholds[sample_id]:.1e}\n"
        else:
            example_text += f"{sample_id}:1.0e-06\n"
    app.threshold_text.insert(tk.END, example_text)

    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=10)
    ttk.Button(button_frame, text="Save", command=lambda: save_simple_thresholds(app, dialog), width=15).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Cancel", command=dialog.destroy, width=15).pack(side=tk.LEFT, padx=5)


def save_simple_thresholds(app, dialog):
    text_content = app.threshold_text.get("1.0", tk.END).strip()
    lines = text_content.split('\n')
    app.expert_thresholds = {}
    valid_count = 0
    for line in lines:
        line = line.strip()
        if ':' in line:
            try:
                sample_id, threshold_str = line.split(':', 1)
                sample_id = sample_id.strip()
                threshold = float(threshold_str.strip())
                app.expert_thresholds[sample_id] = threshold
                valid_count += 1
            except ValueError:
                app.log(f"⚠️ Invalid format: {line}")
    app.log(f"✅ Saved expert thresholds for {valid_count} samples")
    dialog.destroy()


def load_test_data(app):
    filepath = filedialog.askopenfilename(
        title="Select Test Data File",
        filetypes=[
            ("Excel files", "*.xlsx *.xls"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )
    if not filepath:
        return
    df = load_file(app, filepath)
    if df is None:
        return
    app.test_data = df
    app.test_file_path = filepath  # Save file path for later use
    app.log(f"✅ Test data loaded successfully: {len(app.test_data)} particles")
    sample_id = derive_test_sample_id(filepath)
    app.log("📏 Please input voxel size for test data (mm/voxel):")
    app.log("   Example: 0.03 means each voxel edge length is 0.03mm")
    app.log("   If unknown, you can use 0.03 as default value")
    voxel_window = tk.Toplevel(app.root)
    voxel_window.title("Input Test Data Voxel Size")
    voxel_window.geometry("400x200")
    voxel_window.transient(app.root)
    voxel_window.grab_set()
    tk.Label(voxel_window, text=f"Voxel size for test data: {sample_id}", font=("Arial", 12, "bold")).pack(pady=10)
    tk.Label(voxel_window, text="Voxel size (mm/voxel):", font=("Arial", 10)).pack(pady=5)
    voxel_entry = tk.Entry(voxel_window, font=("Arial", 10), width=20)
    voxel_entry.pack(pady=5)
    voxel_entry.insert(0, "0.03")

    def save_voxel_size():
        try:
            voxel_size = float(voxel_entry.get())
            if sample_id not in app.voxel_sizes:
                app.voxel_sizes[sample_id] = voxel_size
            app.log(f"✅ Test data voxel size: {sample_id} = {voxel_size} mm")
            voxel_window.destroy()
        except ValueError:
            tk.messagebox.showerror("Error", "Please enter a valid number")

    tk.Button(voxel_window, text="Save", command=save_voxel_size, font=("Arial", 10), width=10).pack(pady=10)


def input_voxel_sizes(app):
    if not app.sample_list:
        app.log("❌ Please load training data first")
        return
    voxel_window = tk.Toplevel(app.root)
    voxel_window.title("Input Voxel Sizes")
    voxel_window.geometry("600x500")
    voxel_window.transient(app.root)
    voxel_window.grab_set()
    title_label = ttk.Label(voxel_window, text="Input Voxel Sizes for Each Sample", font=("Arial", 14, "bold"))
    title_label.pack(pady=10)
    info_label = ttk.Label(voxel_window, text="Enter voxel size in mm for each sample.\nThis is used for feature normalization.\nExample: 0.03 for 30μm resolution", font=("Arial", 10))
    info_label.pack(pady=5)
    table_frame = ttk.Frame(voxel_window)
    table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    columns = ('Sample ID', 'Voxel Size (mm)')
    tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
    tree.heading('Sample ID', text='Sample ID')
    tree.heading('Voxel Size (mm)', text='Voxel Size (mm)')
    tree.column('Sample ID', width=200)
    tree.column('Voxel Size (mm)', width=200)
    scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    for sample_id in app.sample_list:
        current_value = app.voxel_sizes.get(sample_id, "0.03")
        tree.insert('', 'end', values=(sample_id, current_value))

    def on_double_click(event):
        item = tree.selection()[0]
        column = tree.identify_column(event.x)
        if column == '#2':
            edit_voxel_cell(app, tree, item, column)
    tree.bind('<Double-1>', on_double_click)

    button_frame = ttk.Frame(voxel_window)
    button_frame.pack(fill=tk.X, padx=20, pady=10)

    def save_voxel_sizes():
        try:
            for item in tree.get_children():
                values = tree.item(item)['values']
                sample_id = values[0]
                voxel_size_str = str(values[1]) if values[1] is not None else ""
                if voxel_size_str.strip():
                    voxel_size = float(voxel_size_str)
                    if voxel_size > 0:
                        app.voxel_sizes[sample_id] = voxel_size
                    else:
                        app.log(f"⚠️ Voxel size for sample {sample_id} must be greater than 0")
                        return
                else:
                    app.log(f"⚠️ Voxel size for sample {sample_id} cannot be empty")
                    return
            app.log(f"✅ Saved voxel sizes for {len(app.voxel_sizes)} samples")
            for sid, size in app.voxel_sizes.items():
                app.log(f"   - {sid}: {size:.4f} mm")
            voxel_window.destroy()
        except Exception as e:
            app.log(f"❌ Failed to save voxel sizes: {e}")

    def clear_all_voxel_sizes():
        for item in tree.get_children():
            tree.item(item, values=(tree.item(item)['values'][0], ""))
        app.voxel_sizes.clear()
        app.log("🗑️ Cleared all voxel sizes")

    ttk.Button(button_frame, text="Save", command=save_voxel_sizes, width=15).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Clear All", command=clear_all_voxel_sizes, width=15).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Cancel", command=voxel_window.destroy, width=15).pack(side=tk.RIGHT, padx=5)


def edit_voxel_cell(app, tree, item, column):
    entry = ttk.Entry(tree)
    current_value = tree.item(item)['values'][1] if len(tree.item(item)['values']) > 1 else ""
    entry.insert(0, current_value)
    entry.select_range(0, 'end')

    def save_edit(event=None):
        new_value = entry.get()
        sample_id = tree.item(item)['values'][0]
        tree.item(item, values=(sample_id, new_value))
        entry.destroy()

    def cancel_edit(event=None):
        entry.destroy()

    entry.bind('<Return>', save_edit)
    entry.bind('<Escape>', cancel_edit)
    entry.bind('<FocusOut>', save_edit)
    bbox = tree.bbox(item, column)
    if bbox:
        entry.place(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
        entry.focus_set()


