#!/usr/bin/env python3
"""Generate corrected manuscript Figs 3 and 4 from a released LE01 run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml_threshold_selection.fabric_bootstrap import (
    bootstrap_tp_samples,
    build_spinel_block,
    calculate_T_Pprime_from_vals,
    eigvals_from_logMean,
    precompute_logE_block,
)
from ml_threshold_selection.fabric_thresholds import generate_logstep_thresholds
from ml_threshold_selection.prediction_analysis import compute_dual_thresholds

STEREONET_SCRIPT = ROOT / "scripts" / "generate_fig1_stereonets.py"
STEREONET_SPEC = importlib.util.spec_from_file_location(
    "generate_fig1_stereonets_for_fig4",
    STEREONET_SCRIPT,
)
if STEREONET_SPEC is None or STEREONET_SPEC.loader is None:
    raise ImportError(f"Cannot load stereonet source from {STEREONET_SCRIPT}")
STEREONETS = importlib.util.module_from_spec(STEREONET_SPEC)
STEREONET_SPEC.loader.exec_module(STEREONETS)


BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#1B9E77"
DARK = "#253746"
GREY = "#8C8C8C"
LIGHT_GREY = "#D0D0D0"
LIGHT_BLUE = "#9ECAE1"
FIG4_STEREONET_THRESHOLDS_MM3 = (0.0, 0.0006, 0.002, 0.004, 0.01, 0.06)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "svg.fonttype": "none",
        "svg.hashsalt": "ml-threshold-selection-main-figures",
        "pdf.fonttype": 42,
    })


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"Creator": "ML Threshold Selection", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
        metadata={"Creator": "ML Threshold Selection", "Date": None},
    )
    png_path = stem.with_suffix(".png")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    tif_path = stem.with_suffix(".tif")
    with Image.open(png_path) as source:
        rgba = source.convert("RGBA")
        white = Image.new("RGBA", rgba.size, "white")
        flattened = Image.alpha_composite(white, rgba).convert("RGB")
        flattened.thumbnail((4500, 5250), Image.Resampling.LANCZOS)
        tifffile.imwrite(
            tif_path,
            np.asarray(flattened),
            photometric="rgb",
            compression="deflate",
            resolution=(600, 600),
            resolutionunit="INCH",
            metadata=None,
        )


def _retained_probability_curve(
    voxels: np.ndarray,
    probabilities: np.ndarray,
    voxel_size_mm: float,
) -> pd.DataFrame:
    thresholds = np.logspace(np.log10(voxels.min()), np.log10(voxels.max()), 50)
    rows = []
    for threshold in thresholds:
        retained = voxels >= threshold
        rows.append({
            "candidate_threshold_vox": float(threshold),
            "candidate_threshold_mm3": float(threshold * voxel_size_mm ** 3),
            "retained_n": int(retained.sum()),
            "mean_predicted_below_threshold_probability": float(probabilities[retained].mean()),
        })
    return pd.DataFrame(rows)


def _figure3(
    data: pd.DataFrame,
    curve: pd.DataFrame,
    loose_vox: int,
    strict_vox: int,
    voxel_size_mm: float,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.scatter(
        data["VoxelCount"],
        data["predicted_below_threshold_probability"],
        s=6,
        c=GREY,
        alpha=0.26,
        edgecolors="none",
        rasterized=True,
        label="Segmented objects",
    )
    ax.plot(
        curve["candidate_threshold_vox"],
        curve["mean_predicted_below_threshold_probability"],
        color="#A50026",
        lw=2.0,
        label=r"Retained-population mean, $A(V_{min})$",
    )
    ax.axvline(loose_vox, color=BLUE, lw=1.8, ls="--")
    ax.axvline(strict_vox, color=VERMILLION, lw=1.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(data["VoxelCount"].min() * 0.9, data["VoxelCount"].max() * 1.12)
    ax.set_ylim(-0.025, 1.025)
    ax.set_xlabel("Object volume (voxels)")
    ax.set_ylabel("Predicted probability of the below-threshold class")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", lw=0.6)
    handles = [
        Line2D([], [], marker="o", ls="", ms=4, color=GREY, alpha=0.55, label="Segmented objects"),
        Line2D([], [], color="#A50026", lw=2.0, label=r"Retained-population mean, $A(V_{min})$"),
        Line2D([], [], color=BLUE, lw=1.8, ls="--", label=f"Loose candidate ({loose_vox} voxels)"),
        Line2D([], [], color=VERMILLION, lw=1.8, ls="--", label=f"Strict candidate ({strict_vox} voxels)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    secondary = ax.secondary_xaxis(
        "top",
        functions=(lambda value: value * voxel_size_mm ** 3, lambda value: value / voxel_size_mm ** 3),
    )
    secondary.set_xscale("log")
    secondary.set_xlabel(r"Object volume (mm$^3$)", labelpad=6)
    _save_figure(fig, out_dir / "Fig3")
    plt.close(fig)


def _bootstrap_source(
    data: pd.DataFrame,
    voxel_size_mm: float,
    loose_vox: int,
    strict_vox: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    volumes = data["Volume3d (mm^3) "].astype(float).to_numpy()
    thresholds_mm3 = generate_logstep_thresholds(
        volumes,
        loose_vox * voxel_size_mm ** 3,
        strict_vox * voxel_size_mm ** 3,
        min_particles=50,
        log10_step=0.25,
    )
    log_e = precompute_logE_block(build_spinel_block(data))
    rng = np.random.default_rng(seed)
    long_rows = []
    summary_rows = []
    for threshold_mm3 in thresholds_mm3:
        retained = volumes >= threshold_mm3
        retained_log_e = log_e[retained]
        n_retained = int(retained.sum())
        t_samples, p_samples = bootstrap_tp_samples(retained_log_e, n_bootstrap, rng=rng)
        full_values = eigvals_from_logMean(retained_log_e.mean(axis=0))
        full_t, full_p = calculate_T_Pprime_from_vals(full_values)
        threshold_vox = threshold_mm3 / voxel_size_mm ** 3
        if np.isclose(threshold_vox, loose_vox):
            threshold_class = "loose"
        elif np.isclose(threshold_vox, strict_vox):
            threshold_class = "strict"
        elif threshold_vox < loose_vox:
            threshold_class = "below_loose"
        else:
            threshold_class = "above_loose"
        summary_rows.append({
            "threshold_mm3": float(threshold_mm3),
            "threshold_vox": float(threshold_vox),
            "threshold_class": threshold_class,
            "retained_n": n_retained,
            "full_population_P_prime": float(full_p),
            "full_population_T": float(full_t),
            "bootstrap_P_prime_median": float(np.median(p_samples)),
            "bootstrap_P_prime_q025": float(np.quantile(p_samples, 0.025)),
            "bootstrap_P_prime_q975": float(np.quantile(p_samples, 0.975)),
            "bootstrap_T_median": float(np.median(t_samples)),
            "bootstrap_T_q025": float(np.quantile(t_samples, 0.025)),
            "bootstrap_T_q975": float(np.quantile(t_samples, 0.975)),
        })
        for iteration, (p_prime, t_value) in enumerate(zip(p_samples, t_samples), start=1):
            long_rows.append({
                "threshold_mm3": float(threshold_mm3),
                "threshold_vox": float(threshold_vox),
                "threshold_class": threshold_class,
                "retained_n": n_retained,
                "bootstrap_iteration": iteration,
                "P_prime": float(p_prime),
                "T": float(t_value),
            })
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def _figure4(
    data: pd.DataFrame,
    long_data: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    thresholds = summary["threshold_mm3"].to_numpy()
    positions = np.arange(len(summary), dtype=float)
    p_samples = [
        long_data.loc[np.isclose(long_data["threshold_mm3"], threshold), "P_prime"].to_numpy()
        for threshold in thresholds
    ]
    t_samples = [
        long_data.loc[np.isclose(long_data["threshold_mm3"], threshold), "T"].to_numpy()
        for threshold in thresholds
    ]
    classes = summary["threshold_class"].tolist()
    colors = {
        "loose": GREEN,
        "strict": VERMILLION,
        "below_loose": LIGHT_GREY,
        "above_loose": LIGHT_BLUE,
    }

    fig = plt.figure(figsize=(7.5, 8.2), constrained_layout=True)
    layout = fig.add_gridspec(3, 1, height_ratios=(0.52, 1.0, 1.0))
    stereonet_layout = layout[0].subgridspec(1, len(FIG4_STEREONET_THRESHOLDS_MM3))
    stereonet_axes = [
        fig.add_subplot(stereonet_layout[0, index])
        for index in range(len(FIG4_STEREONET_THRESHOLDS_MM3))
    ]
    volumes = data["Volume3d (mm^3) "].astype(float).to_numpy()
    for ax, threshold_mm3 in zip(stereonet_axes, FIG4_STEREONET_THRESHOLDS_MM3):
        retained = data.loc[volumes >= threshold_mm3]
        vectors = STEREONETS._unit_axial_vectors(retained, 3)
        x_base, _, base_grid = STEREONETS.modified_kamb_mud_grid(vectors)
        x_grid, y_grid, density = STEREONETS._interpolated_net(x_base, base_grid)
        finite = density[np.isfinite(density)]
        levels = np.linspace(
            float(finite.min()),
            float(finite.max()),
            STEREONETS.CONTOUR_INTERVALS + 1,
        )
        fill = ax.contourf(
            x_grid,
            y_grid,
            density,
            levels=levels,
            cmap=STEREONETS._axis_colormap(3),
            antialiased=True,
        )
        ax.contour(
            x_grid,
            y_grid,
            density,
            levels=levels[1:-1],
            colors="#333333",
            linewidths=0.25,
            alpha=0.75,
        )
        STEREONETS._draw_net(ax)
        threshold_label = "0" if threshold_mm3 == 0 else f"{threshold_mm3:g}"
        ax.set_title(
            rf"$V_{{min}}$ = {threshold_label} mm$^3$" + f"\n$n$ = {len(retained):,}",
            fontsize=6.5,
            pad=2,
        )
        colorbar = fig.colorbar(fill, ax=ax, fraction=0.05, pad=0.01)
        colorbar.set_label("m.u.d.", fontsize=5.5)
        colorbar.ax.tick_params(labelsize=5)
    stereonet_axes[0].text(
        -0.24,
        1.16,
        "(a)",
        transform=stereonet_axes[0].transAxes,
        fontsize=11,
        fontweight="bold",
    )

    axes = [fig.add_subplot(layout[1]), fig.add_subplot(layout[2])]
    for ax, samples, full_values, ylabel, reference in [
        (axes[0], p_samples, summary["full_population_P_prime"], "Corrected degree of anisotropy, P'", 1.0),
        (axes[1], t_samples, summary["full_population_T"], "Jelinek shape parameter, T", 0.0),
    ]:
        bp = ax.boxplot(
            samples,
            positions=positions,
            widths=0.62,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": DARK, "lw": 1.1},
            whiskerprops={"color": DARK, "lw": 0.9},
            capprops={"color": DARK, "lw": 0.9},
            boxprops={"edgecolor": DARK, "lw": 0.9},
        )
        for patch, threshold_class in zip(bp["boxes"], classes):
            patch.set_facecolor(colors[threshold_class])
            patch.set_alpha(0.88)
        ax.scatter(positions, full_values, marker="D", s=18, color="black", zorder=4)
        ax.axhline(reference, color="#666666", lw=0.8, ls=":")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E5E5E5", lw=0.6)

    for panel, ax in zip(("(b)", "(c)"), axes):
        ax.text(-0.09, 1.02, panel, transform=ax.transAxes, fontsize=11, fontweight="bold")

    labels = []
    for index, threshold_vox in enumerate(summary["threshold_vox"]):
        if index % 2 == 0 or classes[index] in {"loose", "strict"}:
            labels.append(f"{threshold_vox:.0f}")
        else:
            labels.append("")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, rotation=0)
    axes[1].set_xlabel("Minimum retained volume (voxels)")
    axes[0].set_xticks(positions)
    axes[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axes[0].set_xlim(-0.5, len(positions) - 0.5)
    axes[1].set_xlim(-0.5, len(positions) - 0.5)
    legend = [
        Patch(facecolor=GREEN, edgecolor=DARK, label="Loose candidate"),
        Patch(facecolor=VERMILLION, edgecolor=DARK, label="Strict candidate"),
        Patch(facecolor=LIGHT_GREY, edgecolor=DARK, label="Below loose candidate"),
        Patch(facecolor=LIGHT_BLUE, edgecolor=DARK, label="Above loose candidate"),
        Line2D([], [], marker="D", color="black", ls="", ms=4, label="Full retained-population estimate"),
    ]
    axes[0].legend(
        handles=legend,
        frameon=False,
        loc="upper left",
        ncol=2,
        fontsize=7,
    )
    _save_figure(fig, out_dir / "Fig4")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    _configure_style()
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.predictions, compression="infer")
    required = {"Volume3d (mm^3) ", "VoxelCount", "predicted_below_threshold_probability"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")
    voxels = data["VoxelCount"].astype(float).to_numpy()
    probabilities = data["predicted_below_threshold_probability"].astype(float).to_numpy()
    loose_raw, strict_raw = compute_dual_thresholds(voxels, probabilities, args.strict_probability)
    loose_vox = int(np.ceil(loose_raw))
    strict_vox = int(np.ceil(strict_raw))
    if (loose_vox, strict_vox) != (args.expected_loose, args.expected_strict):
        raise RuntimeError(
            f"Released-model thresholds changed: got {(loose_vox, strict_vox)}, "
            f"expected {(args.expected_loose, args.expected_strict)}"
        )

    objects = data[["index", "Volume3d (mm^3) ", "VoxelCount", "predicted_below_threshold_probability"]].copy()
    objects.to_csv(args.output / "Fig3_objects.csv", index=False)
    curve = _retained_probability_curve(voxels, probabilities, args.voxel_size_mm)
    curve.to_csv(args.output / "Fig3_retained_mean_curve.csv", index=False)
    _figure3(data, curve, loose_vox, strict_vox, args.voxel_size_mm, args.output)

    bootstrap, fabric_summary = _bootstrap_source(
        data,
        args.voxel_size_mm,
        loose_vox,
        strict_vox,
        args.bootstrap,
        args.seed,
    )
    bootstrap.to_csv(
        args.output / "Fig4_bootstrap_values.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    fabric_summary.to_csv(args.output / "Fig4_threshold_summary.csv", index=False)
    _figure4(data, bootstrap, fabric_summary, args.output)

    metadata = {
        "input_predictions": str(args.predictions.resolve()),
        "input_sha256": _sha256(args.predictions),
        "voxel_size_mm": args.voxel_size_mm,
        "seed": args.seed,
        "bootstrap_resamples_per_threshold": args.bootstrap,
        "strict_probability_tolerance": args.strict_probability,
        "loose_continuous_vox": float(loose_raw),
        "loose_applied_vox": loose_vox,
        "strict_applied_vox": strict_vox,
        "figure_4_stereonet_thresholds_mm3": list(FIG4_STEREONET_THRESHOLDS_MM3),
        "figure_4_stereonet_axis": "minimum principal ellipsoid axis (Phi3)",
        "figure_4_stereonet_density_method": "TomoFab modified Kamb m.u.d.",
        "figure_4_outliers_displayed": False,
        "figure_4_outliers_available_in": "Fig4_bootstrap_values.csv.gz",
    }
    (args.output / "Fig3_Fig4_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-size-mm", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--strict-probability", type=float, default=0.01)
    parser.add_argument("--expected-loose", type=int, default=50)
    parser.add_argument("--expected-strict", type=int, default=154)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
