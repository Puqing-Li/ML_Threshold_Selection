from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Enforce publication-ready fonts and sizes
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.default': 'regular',
    'font.size': 14,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 12,
    'axes.linewidth': 2.0,
    'lines.linewidth': 2.0
})



def compute_fabric_params(eigenvals: Sequence[float]) -> Tuple[float, float]:
    """
    Compute T and P' (Jelínek 1981) from eigenvalues [λ1, λ2, λ3].
    Returns:
        (T, P_prime)
    """
    vals = np.asarray(eigenvals, dtype=float)
    if vals.size != 3 or np.any(vals <= 0):
        return np.nan, np.nan

    l1, l2, l3 = vals
    ln1, ln2, ln3 = np.log(l1), np.log(l2), np.log(l3)

    # T (Jelínek 1981)
    if abs(ln1 - ln3) < 1e-12:
        T = 0.0
    else:
        T = (ln2 - ln3 - ln1 + ln2) / (ln2 - ln3 + ln1 - ln2)

    # P' (Jelínek 1981)
    lm = (l1 + l2 + l3) / 3.0
    ln_m = np.log(lm)
    P_prime = float(np.exp(np.sqrt(2.0 * ((ln1 - ln_m) ** 2 + (ln2 - ln_m) ** 2 + (ln3 - ln_m) ** 2))))
    return T, P_prime


def plot_param_boxplot_by_volume_thresholds(
    bootstrap_samples: Mapping[float, Sequence[float]],
    param: str = "T",  # "T" or "P'"
    inflection_threshold: Optional[float] = None,
    zero_artifact_threshold: Optional[float] = None,
    particle_counts: Optional[Mapping[float, int]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    dpi: int = 300,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw T or P' parameter boxplots across volume thresholds,
    matching the PLOS ONE 1:1 format style.
    """
    if param not in ("T", "P'"):
        raise ValueError("param must be either 'T' or \"P'\"")

    if not isinstance(bootstrap_samples, Mapping) or len(bootstrap_samples) == 0:
        raise ValueError("bootstrap_samples must be a non-empty mapping of {threshold: samples}.")

    # 1) Filter and order thresholds
    thresholds: List[float] = []
    box_data: List[np.ndarray] = []
    counts: List[int] = []

    for vt in sorted(bootstrap_samples.keys()):
        arr = np.asarray(bootstrap_samples[vt], dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            continue
        thresholds.append(float(vt))
        arr = arr[~np.isnan(arr)]
        box_data.append(arr)
        if particle_counts and vt in particle_counts:
            counts.append(int(particle_counts[vt]))
        else:
            counts.append(int(arr.size))

    if not thresholds:
        raise ValueError("No valid samples after filtering.")

    # 2) Figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_dpi(dpi)

    # 3) Boxplot at categorical positions
    positions = np.arange(1, len(thresholds) + 1, dtype=float)
    widths = 0.6
    bp = ax.boxplot(
        box_data,
        positions=positions,
        patch_artist=True,
        showfliers=False,
        widths=widths,
        meanline=True,
        showmeans=True,
        meanprops=dict(color="#1F3545", linewidth=1.5, linestyle="--"),
        medianprops=dict(color="#1F3545", linewidth=1.5, linestyle="-"),
        whiskerprops=dict(color="#1F3545", linewidth=1.5),
        capprops=dict(color="#1F3545", linewidth=1.5),
    )

    # 4) PLOS ONE Colors (Colorblind Safe)
    COLOR_LOOSE = "#1B9E77"  # Teal (ML Loose)
    COLOR_STRICT = "#D95F02" # Orange (ML Strict)
    COLOR_BELOW = "#C0C0C0"  # Grey (Below loose)
    COLOR_ABOVE = "#8DA0CB"  # Light Blue (Above loose)
    EDGE_COLOR = "#1F3545"

    def _box_colors(v_threshold: float) -> Tuple[str, str]:
        if inflection_threshold is not None and math.isclose(v_threshold, inflection_threshold, abs_tol=1e-12):
            return COLOR_LOOSE, EDGE_COLOR
        if zero_artifact_threshold is not None and math.isclose(v_threshold, zero_artifact_threshold, abs_tol=1e-12):
            return COLOR_STRICT, EDGE_COLOR
        if inflection_threshold is not None and v_threshold < inflection_threshold:
            return COLOR_BELOW, EDGE_COLOR
        return COLOR_ABOVE, EDGE_COLOR

    for i, patch in enumerate(bp["boxes"]):
        v = thresholds[i]
        fill, edge = _box_colors(v)
        patch.set_facecolor(fill)
        patch.set_edgecolor(edge)
        patch.set_alpha(0.9)
        patch.set_linewidth(1.5)

    # 5) n labels above upper whiskers
    for i, (box, n) in enumerate(zip(bp["boxes"], counts)):
        verts = box.get_path().vertices
        # Find the top of the upper whisker
        whisker_top = float(np.max(bp["caps"][2*i+1].get_ydata()))
        y_pos = whisker_top + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0] + 1e-6)
        
        ax.text(
            positions[i],
            y_pos,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color="#222222",
        )

    # 6) Reference line, labels
    if param == "T":
        ax.axhline(y=0.0, color="red", linestyle="--", linewidth=1.2)
        ax.set_ylabel("T Parameter", fontweight="bold")
    else:
        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.2)
        ax.set_ylabel("P' Parameter", fontweight="bold")

    ax.set_xlabel("Minimum Volume Threshold (mm³)", fontweight="bold")

    # Scientific notation for xticks: 1.1 × 10⁻⁴ using MathText
    def _sci_notation(v: float) -> str:
        if v == 0:
            return "$0$"
        exponent = int(math.floor(math.log10(abs(v))))
        coeff = v / (10 ** exponent)
        return rf"${coeff:.1f} \times 10^{{{exponent}}}$"

    ax.set_xticks(positions)
    labels = [_sci_notation(v) for v in thresholds]
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=16, fontweight="bold")
    
    # Explicitly enlarge and bold the Y-axis tick labels to match the X-axis
    ax.tick_params(axis='y', labelsize=16)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")


    # Y-limits
    all_vals = np.concatenate(box_data) if len(box_data) > 0 else np.array([0.0])
    if param == "T" and all_vals.size:
        y_min = float(np.nanmin(all_vals) - 0.2)
        y_max = float(np.nanmax(all_vals) + 0.2)
        ax.set_ylim(y_min, y_max)
    elif param == "P'" and all_vals.size:
        y_min = max(0.9, float(np.nanmin(all_vals) - 0.2))
        y_max = float(np.nanmax(all_vals) + 0.5)
        ax.set_ylim(y_min, y_max)

    # Styling: border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color("#111111")

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", loc='left')

    # Legend perfectly tailored to 1:1 image style
    legend_elements = [
        Patch(facecolor=COLOR_LOOSE, edgecolor=EDGE_COLOR, label="ML Loose Threshold (inflection)"),
        Patch(facecolor=COLOR_STRICT, edgecolor=EDGE_COLOR, label="ML Strict Threshold (Zero artifact)"),
        Patch(facecolor=COLOR_BELOW, edgecolor=EDGE_COLOR, label="Below loose threshold"),
        Patch(facecolor=COLOR_ABOVE, edgecolor=EDGE_COLOR, label="Above loose threshold (excluding strict)"),
        Line2D([0], [0], color=EDGE_COLOR, lw=1.5, linestyle="-", label="Median (solid)"),
        Line2D([0], [0], color=EDGE_COLOR, lw=1.5, linestyle="--", label="Mean (dashed, bootstrap)"),
        # Line2D([0], [0], color="red", lw=1.2, linestyle="--", label="Single-pass value (no bootstrap)"),
        Patch(facecolor="white", edgecolor=EDGE_COLOR, label="Box = IQR (Q1–Q3)"),
        Line2D([0], [0], color=EDGE_COLOR, lw=1.5, label="Whiskers = 1.5×IQR"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, edgecolor='#A0A0A0', 
              fontsize=10, borderpad=0.8, labelspacing=0.5)

    ax.grid(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    if show:
        plt.show()

    return fig, ax
