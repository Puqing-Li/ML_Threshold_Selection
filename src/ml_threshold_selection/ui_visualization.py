#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tkinter visualization helpers extracted from main.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def save_chart(fig, base_name, format_type, log_func):
    from datetime import datetime
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.{format_type}"
    filepath = os.path.join(output_dir, filename)
    if format_type == "png":
        fig.savefig(filepath, dpi=300, bbox_inches='tight', format='png')
    elif format_type == "svg":
        fig.savefig(filepath, bbox_inches='tight', format='svg')
    log_func(f"✅ Chart saved as {format_type.upper()}: {filepath}")


def export_publication_fig3(app):
    """
    Export the exact 1:1 replica of PLOS ONE Fig 3: Artifact probability vs volume.
    Enforces Arial font, 300 DPI, specified colors, and exact structure.
    """
    import os
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    # 1. Enforce PLOS ONE Styles
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    # Create single figure
    fig, ax1 = plt.subplots(figsize=(10, 7), dpi=300)
    
    volumes = app.test_data['Volume3d (mm^3) '].values
    first_sample = app.sample_list[0] if app.sample_list else list(app.voxel_sizes.keys())[0]
    voxel_mm = float(app.voxel_sizes[first_sample])
    voxel_vol = voxel_mm ** 3
    
    # Convert from volume to exact V_min in mm³
    v_min_mm3 = volumes
    
    # 2. Scatter plot with plasma colormap (yellow highest, purple lowest)
    scatter = ax1.scatter(v_min_mm3, app.probabilities, c=app.probabilities, cmap='plasma_r', alpha=0.3, s=12, edgecolors='none')
    
    # Setup ax1
    ax1.set_xscale('log')
    ax1.set_xlabel(r'Minimum Volume ($V_{\mathrm{min}}$, mm³)', fontweight='bold')
    ax1.set_ylabel('Artifact Probability', fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(-0.05, 1.05)
    
    # Setup ax2 (twinx) for the mean curve
    ax2 = ax1.twinx()
    ax2.set_ylabel('Artifact Probability', fontweight='bold', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.set_ylim(-0.05, 1.05)
    
    # Mean curve using cumulative algorithm (retained particles)
    thresholds = np.logspace(np.log10(max(v_min_mm3.min(), 1e-4)), np.log10(v_min_mm3.max()), 50)
    artifact_rates = []
    for t in thresholds:
        retained = v_min_mm3 >= t
        artifact_rates.append(float(np.mean(app.probabilities[retained])) if np.sum(retained) > 0 else 0.0)
    
    ax2.plot(thresholds, artifact_rates, color='darkred', linewidth=2.5, label='Mean Artifact Probability')
    
    # PLOS ONE Colors for lines
    COLOR_LOOSE = "#0072B2"  # Blue/Teal dotted
    COLOR_STRICT = "#D55E00" # Orange dotted
    
    # Threshold Lines
    lines = []
    lines.append(Line2D([0], [0], color='darkred', lw=1.5, label='Mean Artifact Probability'))
    
    if hasattr(app, 'loose_threshold_vox') and app.loose_threshold_vox is not None:
        loose_mm3 = app.loose_threshold_vox * voxel_vol
        ax1.axvline(x=loose_mm3, color=COLOR_LOOSE, linestyle=':', linewidth=2)
        lines.append(Line2D([0], [0], color=COLOR_LOOSE, lw=2, linestyle=':', label='Loose Threshold'))
        
    if hasattr(app, 'strict_threshold_vox') and app.strict_threshold_vox is not None:
        strict_mm3 = app.strict_threshold_vox * voxel_vol
        ax1.axvline(x=strict_mm3, color=COLOR_STRICT, linestyle=':', linewidth=2)
        lines.append(Line2D([0], [0], color=COLOR_STRICT, lw=2, linestyle=':', label='Strict Threshold'))
        
    ax1.legend(handles=lines, loc='lower left', frameon=False)
    
    # Colorbar on ax1
    cbar = fig.colorbar(scatter, ax=ax1, pad=0.08, shrink=0.7)
    cbar.set_label('Artifact Probability', rotation=90, labelpad=15, color='darkred')
    
    # Frame styling
    for spine in ['bottom', 'left']:
         ax1.spines[spine].set_linewidth(1.5)
    for spine in ['right']:
         ax2.spines[spine].set_linewidth(1.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
         
    plt.tight_layout()
    
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"Publication_Fig3_{timestamp}.pdf")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    filepath_png = os.path.join(output_dir, f"Publication_Fig3_{timestamp}.png")
    fig.savefig(filepath_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    app.log(f"✅ Publication Fig3 saved: {filepath_png}")


def show_training_visualization(app):
    if app.training_results is None:
        app.log("❌ Please train model first")
        return
    if app.visualization_window is not None:
        app.visualization_window.destroy()
    import tkinter as tk
    app.visualization_window = tk.Toplevel(app.root)
    app.visualization_window.title("Training Results Visualization")
    app.visualization_window.geometry("1200x800")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Model Training Results Analysis', fontsize=16, fontweight='bold')

    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(app.training_results['y'], app.training_results['train_proba'])
    axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f"AUC = {app.training_results['train_auc']:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=1)
    axes[0, 0].set_xlabel('False Positive Rate (FPR)')
    axes[0, 0].set_ylabel('True Positive Rate (TPR)')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)

    feature_names = app.training_results['features'].columns
    if hasattr(app.model, 'feature_importances_'):
        importance = app.model.feature_importances_
    elif hasattr(app.model, 'feature_importance'):
        importance = app.model.feature_importance(importance_type='gain')
    else:
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(app.model, app.training_results['X'], app.training_results['y'], n_repeats=5, random_state=42)
        importance = perm_importance.importances_mean
    top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:10]
    if top_features:
        names, values = zip(*top_features)
        axes[0, 1].barh(range(len(names)), values)
        axes[0, 1].set_yticks(range(len(names)))
        axes[0, 1].set_yticklabels(names)
        axes[0, 1].set_xlabel('Importance')
        axes[0, 1].set_title('Feature Importance (Top 10)')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No feature importance available', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Feature Importance')
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)

    normal_proba = app.training_results['train_proba'][app.training_results['y'] == 0]
    artifact_proba = app.training_results['train_proba'][app.training_results['y'] == 1]
    axes[1, 0].hist(normal_proba, bins=30, alpha=0.7, label='Normal Particles', color='blue')
    axes[1, 0].hist(artifact_proba, bins=30, alpha=0.7, label='Artifact Particles', color='red')
    axes[1, 0].axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
    axes[1, 0].set_xlabel('Prediction Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Prediction Probability Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [app.training_results['train_accuracy'], app.training_results['precision'], app.training_results['recall'], app.training_results['f1']]
    bars = axes[1, 1].bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)
    for bar, value in zip(bars, values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, app.visualization_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    import tkinter as tk
    save_frame = tk.Frame(app.visualization_window)
    save_frame.pack(pady=10)
    tk.Button(save_frame, text="Save as PNG", command=lambda: save_chart(fig, "training_visualization", "png", app.log)).pack(side=tk.LEFT, padx=5)
    tk.Button(save_frame, text="Save as SVG", command=lambda: save_chart(fig, "training_visualization", "svg", app.log)).pack(side=tk.LEFT, padx=5)
    app.log("✅ Training visualization displayed")


def show_prediction_visualization(app):
    if app.test_data is None or app.probabilities is None:
        app.log("❌ Please perform prediction analysis first")
        return
    if app.visualization_window is not None:
        app.visualization_window.destroy()
    import tkinter as tk
    app.visualization_window = tk.Toplevel(app.root)
    app.visualization_window.title("Prediction Results Visualization")
    app.visualization_window.geometry("1400x850")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Prediction Results Analysis', fontsize=16, fontweight='bold')

    volumes = app.test_data['Volume3d (mm^3) '].values
    if not app.voxel_sizes:
        app.log("❌ Please input voxel sizes first (mm)")
        return
    first_sample = app.sample_list[0] if app.sample_list else list(app.voxel_sizes.keys())[0]
    voxel_mm = float(app.voxel_sizes[first_sample])
    voxel_vol = voxel_mm ** 3
    voxels_cont = np.clip(volumes / voxel_vol, a_min=1e-12, a_max=None)

    axes[0, 0].hist(np.log10(voxels_cont), bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('log10(Voxel Count)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Voxel Count Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    xticks = axes[0, 0].get_xticks()
    mm3_labels = [f"{(10**t)*voxel_vol:.1e}" for t in xticks]
    ax_top = axes[0, 0].secondary_xaxis('top')
    ax_top.set_xticks(xticks)
    ax_top.set_xticklabels(mm3_labels, rotation=0)
    ax_top.set_xlabel('Equivalent Volume (mm³)')
    ax_top.set_xlabel('Equivalent Volume (mm³)')
    
    # Hide the spines on axes[0, 0] and ax_top
    axes[0, 0].spines['right'].set_visible(False)
    axes[0, 0].spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['top'].set_visible(False)

    # 2. Replicate Fig 3 styling for the top-right subplot (axes[0, 1])
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.0
    })
    v_min_mm3 = volumes
    scatter = axes[0, 1].scatter(v_min_mm3, app.probabilities, c=app.probabilities, cmap='plasma_r', alpha=0.3, s=15, edgecolors='none')
    
    axes[0, 1].set_xscale('log')
    # Set explicit font sizes for the top-right subplot
    axes[0, 1].set_xlabel(r'Minimum Volume ($V_{\mathrm{min}}$, mm³)', fontweight='bold', fontsize=11)
    axes[0, 1].set_ylabel('Artifact Probability', fontweight='bold', color='black', fontsize=11)
    # Set explicit tick label sizes
    axes[0, 1].tick_params(axis='both', labelsize=9)
    axes[0, 1].tick_params(axis='y', labelcolor='black')
    
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].set_title('Prediction Probability vs Minimum Volume', fontweight='bold', fontsize=12)
    
    ax01_twin = axes[0, 1].twinx()
    ax01_twin.set_ylabel('Artifact Probability', fontweight='bold', color='darkred')
    ax01_twin.tick_params(axis='y', labelcolor='darkred')
    ax01_twin.set_ylim(-0.05, 1.05)
    
    thresholds_gui = np.logspace(np.log10(max(v_min_mm3.min(), 1e-4)), np.log10(v_min_mm3.max()), 50)
    artifact_rates_gui = []
    for t in thresholds_gui:
        retained = v_min_mm3 >= t
        artifact_rates_gui.append(float(np.mean(app.probabilities[retained])) if np.sum(retained) > 0 else 0.0)
    
    ax01_twin.plot(thresholds_gui, artifact_rates_gui, color='darkred', linewidth=2.5, label='Mean Artifact Probability')
    
    from matplotlib.lines import Line2D
    lines_01 = [Line2D([0], [0], color='darkred', lw=1.5, label='Mean Artifact Probability')]
    
    COLOR_LOOSE = "#0072B2"
    COLOR_STRICT = "#D55E00"
    if hasattr(app, 'loose_threshold_vox') and app.loose_threshold_vox is not None:
        loose_mm3 = app.loose_threshold_vox * voxel_vol
        axes[0, 1].axvline(x=loose_mm3, color=COLOR_LOOSE, linestyle=':', linewidth=2)
        lines_01.append(Line2D([0], [0], color=COLOR_LOOSE, lw=2, linestyle=':', label='Loose Threshold'))
        
    if hasattr(app, 'strict_threshold_vox') and app.strict_threshold_vox is not None:
        strict_mm3 = app.strict_threshold_vox * voxel_vol
        axes[0, 1].axvline(x=strict_mm3, color=COLOR_STRICT, linestyle=':', linewidth=2)
        lines_01.append(Line2D([0], [0], color=COLOR_STRICT, lw=2, linestyle=':', label='Strict Threshold'))
        
    axes[0, 1].legend(handles=lines_01, loc='lower left', frameon=False, fontsize=8)
    
    cbar = fig.colorbar(scatter, ax=axes[0, 1], pad=0.08, shrink=0.7)
    cbar.set_label('Artifact Probability', rotation=90, labelpad=15, color='darkred')

    for spine in ['bottom', 'left']:
         axes[0, 1].spines[spine].set_linewidth(1.5)
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)
    
    # Hide all traces of the right Y-axis for the top-right subplot
    ax01_twin.spines['right'].set_visible(False)
    ax01_twin.spines['top'].set_visible(False)
    ax01_twin.spines['left'].set_visible(False)
    ax01_twin.yaxis.set_visible(False)
    
    axes[1, 0].hist(app.probabilities, bins=30, alpha=0.7, color='lightgreen')
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)
    axes[1, 0].axvline(x=0.5, color='red', linestyle='--', label='Threshold=0.5')
    axes[1, 0].set_xlabel('Prediction Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Prediction Probability Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    thresholds = np.logspace(np.log10(max(voxels_cont.min(), 1e-12)), np.log10(voxels_cont.max()), 50)
    retention_rates, artifact_rates = [], []
    for t in thresholds:
        retained = voxels_cont >= t
        retention_rates.append(float(np.mean(retained)))
        artifact_rates.append(float(np.mean(app.probabilities[retained])) if np.sum(retained) > 0 else 0.0)
    ax2 = axes[1, 1].twinx()
    line1 = axes[1, 1].plot(np.log10(thresholds), retention_rates, 'b-', label='Retention Rate')
    line2 = ax2.plot(np.log10(thresholds), artifact_rates, 'r-', label='Artifact Rate')
    
    # Add threshold lines if available
    if hasattr(app, 'loose_threshold_vox') and app.loose_threshold_vox is not None:
        loose_log = np.log10(app.loose_threshold_vox)
        axes[1, 1].axvline(x=loose_log, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Loose Threshold')
        ax2.axvline(x=loose_log, color='green', linestyle='--', linewidth=2, alpha=0.8)
    
    if hasattr(app, 'strict_threshold_vox') and app.strict_threshold_vox is not None:
        strict_log = np.log10(app.strict_threshold_vox)
        strict_label = f'Strict Threshold (P>{app.strict_probability_threshold})'
        axes[1, 1].axvline(x=strict_log, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=strict_label)
        ax2.axvline(x=strict_log, color='orange', linestyle='--', linewidth=2, alpha=0.8)
    
    axes[1, 1].set_xlabel('log10(Voxel Threshold)')
    axes[1, 1].set_ylabel('Retention Rate', color='b')
    ax2.set_ylabel('Artifact Rate', color='r')
    axes[1, 1].set_title('Dual Threshold Analysis (Voxel Domain)')
    axes[1, 1].set_title('Dual Threshold Analysis (Voxel Domain)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes[1, 1].legend(lines, labels, loc='center right')

    plt.tight_layout()
    import tkinter as tk
    
    # Pack Right frame for export buttons FIRST so it doesn't get squashed by the expanding canvas
    right_panel = tk.Frame(app.visualization_window, width=280, bg="#f0f0f0")
    right_panel.pack(side=tk.RIGHT, fill='y', padx=10, pady=10)
    
    tk.Label(right_panel, text="Export HQ Subplots", font=('Arial', 12, 'bold'), bg="#f0f0f0").pack(pady=(0,5))
    tk.Label(right_panel, text="(300 DPI, Publication Ready)", font=('Arial', 9), bg="#f0f0f0").pack(pady=(0,15))
    
    # Left frame for canvas goes SECOND
    plot_frame = tk.Frame(app.visualization_window)
    plot_frame.pack(side=tk.LEFT, fill='both', expand=True)
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def export_single_subplot(target_ax, ax_twins, title_prefix):
        import os
        from datetime import datetime
        from matplotlib.transforms import Bbox
        
        # Hide all axes except target and its twins
        states = {a: a.get_visible() for a in fig.axes}
        for a in fig.axes:
            a.set_visible(False)
        target_ax.set_visible(True)
        for t in ax_twins:
            t.set_visible(True)
            
        suptit = fig._suptitle.get_text() if fig._suptitle else ""
        if suptit: fig.suptitle("")
        
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bboxes = [target_ax.get_tightbbox(renderer)]
        for t in ax_twins: bboxes.append(t.get_tightbbox(renderer))
        
        merged_bbox = Bbox.union(bboxes)
        extent = merged_bbox.transformed(fig.dpi_scale_trans.inverted())
        
        out_dir = "outputs"
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fpath_png = os.path.join(out_dir, f"{title_prefix}_{ts}.png")
        fpath_svg = os.path.join(out_dir, f"{title_prefix}_{ts}.svg")
        fig.savefig(fpath_png, dpi=300, bbox_inches=extent.expanded(1.05, 1.15), facecolor='white')
        fig.savefig(fpath_svg, dpi=300, bbox_inches=extent.expanded(1.05, 1.15), facecolor='white')
        
        # Restore
        for a, s in states.items(): a.set_visible(s)
        if suptit: fig.suptitle(suptit, fontsize=16, fontweight='bold')
        canvas.draw()
        app.log(f"✅ Exported {title_prefix} to .png and .svg")

    tk.Button(right_panel, text="1. Voxel Count Dist. (PNG+SVG)", width=28,
              command=lambda: export_single_subplot(axes[0,0], [ax_top], "VoxelCount")).pack(pady=5)
    tk.Button(right_panel, text="2. Prob. vs Vol. Curve (PNG+SVG)", width=28,
              command=lambda: export_single_subplot(axes[0,1], [ax01_twin], "Prob_vs_Vol")).pack(pady=5)
    tk.Button(right_panel, text="3. Prob. Dist. (PNG+SVG)", width=28,
              command=lambda: export_single_subplot(axes[1,0], [], "Prob_Dist")).pack(pady=5)
    tk.Button(right_panel, text="4. Dual Threshold (PNG+SVG)", width=28,
              command=lambda: export_single_subplot(axes[1,1], [ax2], "Dual_Threshold")).pack(pady=5)
              
    tk.Label(right_panel, text="Whole GUI Image Actions", font=('Arial', 10, 'bold'), bg="#f0f0f0").pack(pady=(25, 5))
    tk.Button(right_panel, text="Save FULL Grid as PNG", width=28, command=lambda: save_chart(fig, "prediction_visualization", "png", app.log)).pack(pady=5)
    tk.Button(right_panel, text="Export HQ Prob Curve", width=28, command=lambda: export_publication_fig3(app)).pack(pady=5)
    app.log("✅ Prediction visualization displayed")


