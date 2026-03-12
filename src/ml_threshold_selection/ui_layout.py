#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI layout builder extracted from main.
"""

import tkinter as tk
from tkinter import ttk


def build_main_ui(app):
    # Window title
    app.root.title("ML Threshold Selection System")

    # Base styles
    base_font = ("Helvetica", 12)
    mono_font = ("Consolas", 11)
    title_font = ("Helvetica", 17, "bold")
    subtitle_font = ("Helvetica", 11)

    style = ttk.Style()
    style.configure("TLabel", font=base_font)
    style.configure("TButton", font=("Helvetica", 12, "bold"), padding=6)
    style.configure("Header.TLabel", font=title_font)
    style.configure("Section.TLabelframe.Label", font=("Helvetica", 12, "bold"))

    main_frame = ttk.Frame(app.root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

    # Header
    header_frame = ttk.Frame(main_frame)
    header_frame.pack(fill=tk.X, pady=(0, 10))

    title_label = ttk.Label(
        header_frame,
        text="ML Threshold Selection System",
        style="Header.TLabel",
    )
    title_label.pack(anchor=tk.W)

    status_text = []
    status_text.append("Model: LightGBM ✅" if app.LIGHTGBM_AVAILABLE else "Model: LightGBM ❌")
    status_text.append("Modules: Full ✅" if app.FULL_MODULES_AVAILABLE else "Modules: Full ❌")
    status_label = ttk.Label(
        header_frame,
        text=" | ".join(status_text),
        font=subtitle_font,
        foreground="#3a3a3a",
    )
    status_label.pack(anchor=tk.W, pady=(4, 0))

    # Layout container
    content_frame = ttk.Frame(main_frame)
    content_frame.pack(fill=tk.BOTH, expand=False)

    # Workflow (numbered steps)
    workflow_frame = ttk.LabelFrame(content_frame, text="Workflow (Step-by-Step)", padding=10, style="Section.TLabelframe")
    workflow_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))

    btn_w = 28
    ttk.Button(workflow_frame, text="1. Load Training Data", width=btn_w, command=app.load_multiple_training_data).grid(row=0, column=0, padx=6, pady=4)
    ttk.Button(workflow_frame, text="2. Input Expert Thresholds", width=btn_w, command=app.input_expert_thresholds).grid(row=0, column=1, padx=6, pady=4)
    ttk.Button(workflow_frame, text="3. Input Voxel Sizes", width=btn_w, command=app.input_voxel_sizes).grid(row=1, column=0, padx=6, pady=4)
    ttk.Button(workflow_frame, text="4. Feature Analysis", width=btn_w, command=app.analyze_ellipsoid_features).grid(row=1, column=1, padx=6, pady=4)
    ttk.Button(workflow_frame, text="5. Train Model", width=btn_w, command=app.train_model).grid(row=2, column=0, padx=6, pady=4)
    ttk.Button(workflow_frame, text="6a. Load Single Test Data", width=btn_w, command=app.load_test_data).grid(row=2, column=1, padx=6, pady=4)
    ttk.Button(workflow_frame, text="6b. Load Multi Test Data", width=btn_w, command=app.load_test_data_multiple).grid(row=3, column=0, padx=6, pady=4)
    ttk.Button(workflow_frame, text="7. Predict Analysis", width=btn_w, command=app.predict_analysis).grid(row=3, column=1, padx=6, pady=4)
    ttk.Button(workflow_frame, text="8. Export / Reports", width=btn_w, command=app.export_results).grid(row=4, column=0, padx=6, pady=4)

    # Analysis & Visualization
    analysis_frame = ttk.LabelFrame(content_frame, text="Analysis & Visualization", padding=10, style="Section.TLabelframe")
    analysis_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
    ttk.Button(analysis_frame, text="Training Visualization", width=btn_w, command=app.show_training_visualization).grid(row=0, column=0, padx=6, pady=4)
    ttk.Button(analysis_frame, text="Prediction Visualization", width=btn_w, command=app.show_prediction_visualization).grid(row=0, column=1, padx=6, pady=4)
    ttk.Button(analysis_frame, text="Fabric Boxplots", width=btn_w, command=app.generate_fabric_boxplots).grid(row=1, column=0, padx=6, pady=4)
    ttk.Button(analysis_frame, text="Mean Fabric", width=btn_w, command=app.calculate_mean_fabric).grid(row=1, column=1, padx=6, pady=4)

    # Models & Config
    model_frame = ttk.LabelFrame(content_frame, text="Model & Threshold Config", padding=10, style="Section.TLabelframe")
    model_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 0), pady=(0, 10))
    ttk.Button(model_frame, text="Load Last Model", width=btn_w, command=app.load_last_time_model).grid(row=0, column=0, padx=6, pady=4)
    ttk.Button(model_frame, text="Config Threshold", width=btn_w, command=app.configure_strict_threshold).grid(row=1, column=0, padx=6, pady=4)

    # Help
    help_frame = ttk.LabelFrame(content_frame, text="Help", padding=10, style="Section.TLabelframe")
    help_frame.grid(row=1, column=1, sticky="nsew", padx=(0, 0), pady=(0, 10))
    ttk.Button(help_frame, text="User Guide", width=btn_w, command=app.open_user_guide).grid(row=0, column=0, padx=6, pady=4)

    # Make columns responsive
    content_frame.columnconfigure(0, weight=1)
    content_frame.columnconfigure(1, weight=1)
    workflow_frame.columnconfigure((0, 1), weight=1)
    analysis_frame.columnconfigure((0, 1), weight=1)

    # Status bar
    app.status_label = ttk.Label(main_frame, text="Waiting for operation...", font=("Helvetica", 12, "bold"))
    app.status_label.pack(fill=tk.X, pady=(0, 8))

    # Results / log panel
    log_frame = ttk.Frame(main_frame)
    log_frame.pack(fill=tk.BOTH, expand=True)

    app.results_text = tk.Text(
        log_frame,
        height=22,
        wrap=tk.NONE,
        font=mono_font,
        background="#f7f7f7",
        foreground="#1f1f1f",
    )
    app.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar_y = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=app.results_text.yview)
    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
    app.results_text.configure(yscrollcommand=scrollbar_y.set)

    scrollbar_x = ttk.Scrollbar(log_frame, orient=tk.HORIZONTAL, command=app.results_text.xview)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
    app.results_text.configure(xscrollcommand=scrollbar_x.set)
