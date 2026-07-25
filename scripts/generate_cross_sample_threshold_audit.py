#!/usr/bin/env python3
"""Generate a reproducible five-sample threshold and fabric audit figure."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from axis_locking_validation import _axis_distance_degrees, _summary
from cross_validation import VOL, _read_table, load_config
from loso_threshold_validation import run as run_loso_validation
from src.ml_threshold_selection.fabric_bootstrap import (
    bootstrap_tp_samples,
    build_spinel_block,
    calculate_T_Pprime_from_vals,
    eigvals_from_logMean,
    precompute_logE_block,
)


COLORS = {
    "expert": "#555555",
    "loso_loose": "#2F6B9A",
    "loso_strict": "#D97706",
}
LABELS = {
    "expert": "Expert",
    "loso_loose": "LOSO loose",
    "loso_strict": "LOSO strict",
}


def _fabric_summary(
    log_tensors: np.ndarray,
    retained: np.ndarray,
    seed: np.random.SeedSequence,
    n_bootstrap: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    selected = log_tensors[retained]
    axes = eigvals_from_logMean(selected.mean(axis=0))
    t_mean, p_mean = calculate_T_Pprime_from_vals(axes)
    t_samples, p_samples = bootstrap_tp_samples(
        selected,
        n_bootstrap,
        np.random.default_rng(seed),
    )
    t_values = np.asarray(t_samples)
    p_values = np.asarray(p_samples)
    return (
        {
            "T": t_mean,
            "T_CI95_low": float(np.percentile(t_values, 2.5)),
            "T_CI95_high": float(np.percentile(t_values, 97.5)),
            "Pprime": p_mean,
            "Pprime_CI95_low": float(np.percentile(p_values, 2.5)),
            "Pprime_CI95_high": float(np.percentile(p_values, 97.5)),
        },
        t_values,
        p_values,
    )


def build(repo: Path, output_stem: Path, n_bootstrap: int = 1000) -> pd.DataFrame:
    config = load_config(repo / "examples" / "expert_thresholds.csv")
    loso = run_loso_validation(repo).set_index("Sample")
    sample_ids = list(config)
    rows = []
    bootstrap_values: dict[tuple[str, str, str], np.ndarray] = {}

    for sample_index, sample_id in enumerate(sample_ids):
        metadata = config[sample_id]
        table = _read_table(repo / "training_data" / f"total{sample_id}.xlsx")
        voxel_volume = metadata["vox"] ** 3
        voxel_count = table[VOL].to_numpy(float) / voxel_volume
        aligned = _axis_distance_degrees(table, 3) <= 5.0
        log_tensors = precompute_logE_block(build_spinel_block(table))
        threshold_mm3 = {
            "expert": float(metadata["thr"]),
            "loso_loose": float(loso.loc[sample_id, "LooseVmin_mm3"]),
            "loso_strict": float(loso.loc[sample_id, "StrictVmin_mm3"]),
        }
        continuous_candidate_voxels = {
            "expert": np.nan,
            "loso_loose": float(loso.loc[sample_id, "LooseVmin_continuous_vox"]),
            "loso_strict": float(loso.loc[sample_id, "StrictVmin_continuous_vox"]),
        }

        for threshold_index, (threshold_type, physical_threshold) in enumerate(threshold_mm3.items()):
            voxel_equivalent = physical_threshold / voxel_volume
            retained = table[VOL].to_numpy(float) >= physical_threshold
            alignment = _summary(retained, aligned)
            fabric, t_values, p_values = _fabric_summary(
                log_tensors,
                retained,
                np.random.SeedSequence([42, sample_index, threshold_index]),
                n_bootstrap,
            )
            bootstrap_values[(sample_id, threshold_type, "T")] = t_values
            bootstrap_values[(sample_id, threshold_type, "Pprime")] = p_values
            rows.append(
                {
                    "Sample": sample_id,
                    "ThresholdType": threshold_type,
                    "VoxelSize_mm": metadata["vox"],
                    "ExpertThreshold_mm3": metadata["thr"],
                    "ExpertEquivalentVoxelCount": metadata["thr"] / voxel_volume,
                    "ExpertMinimumRetainedIntegerVoxelCount": int(
                        np.ceil(metadata["thr"] / voxel_volume)
                    ),
                    "CandidateContinuousVoxelThreshold": continuous_candidate_voxels[
                        threshold_type
                    ],
                    "Threshold_vox_equivalent": voxel_equivalent,
                    "MinimumRetainedIntegerVoxelCount": int(np.ceil(voxel_equivalent)),
                    "Threshold_mm3": physical_threshold,
                    "RelativeError_vsExpert": physical_threshold / metadata["thr"] - 1.0,
                    "Retained_n": int(retained.sum()),
                    "MinimumAxisAligned": alignment["aligned"],
                    "MinimumAxisAlignedFraction": alignment["fraction"],
                    "MinimumAxisAlignedWilson95Low": alignment["wilson_95_low"],
                    "MinimumAxisAlignedWilson95High": alignment["wilson_95_high"],
                    **fabric,
                    "BootstrapResamples": n_bootstrap,
                }
            )

    results = pd.DataFrame(rows)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_stem.with_suffix(".csv"), index=False)
    _plot(results, bootstrap_values, sample_ids, output_stem)
    return results


def _plot(
    results: pd.DataFrame,
    bootstrap_values: dict[tuple[str, str, str], np.ndarray],
    sample_ids: list[str],
    output_stem: Path,
) -> None:
    threshold_types = list(COLORS)
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.0))

    ax = axes[0, 0]
    expert = results.loc[results["ThresholdType"] == "expert"].set_index("Sample")
    values = [results.loc[results["ThresholdType"] == kind].set_index("Sample") for kind in threshold_types[1:]]
    for kind, frame, marker in zip(threshold_types[1:], values, ("o", "s")):
        ax.scatter(
            expert.loc[sample_ids, "Threshold_mm3"],
            frame.loc[sample_ids, "Threshold_mm3"],
            color=COLORS[kind],
            marker=marker,
            s=55,
            label=LABELS[kind],
            zorder=3,
        )
        label_offsets = {
            ("loso_loose", "AKAN20"): (4, -12),
            ("loso_loose", "ANA16937"): (4, -12),
            ("loso_loose", "HL19335"): (4, 5),
            ("loso_loose", "LE03"): (4, 5),
            ("loso_loose", "LE19"): (4, -12),
            ("loso_strict", "AKAN20"): (4, 5),
            ("loso_strict", "ANA16937"): (4, 5),
            ("loso_strict", "HL19335"): (4, 5),
            ("loso_strict", "LE03"): (4, 5),
            ("loso_strict", "LE19"): (4, -12),
        }
        for sample_id in sample_ids:
            ax.annotate(
                sample_id,
                (
                    expert.loc[sample_id, "Threshold_mm3"],
                    frame.loc[sample_id, "Threshold_mm3"],
                ),
                xytext=label_offsets[(kind, sample_id)],
                textcoords="offset points",
                fontsize=8,
            )
    all_thresholds = results["Threshold_mm3"].to_numpy(float)
    limits = [all_thresholds.min() * 0.7, all_thresholds.max() * 1.4]
    ax.plot(limits, limits, linestyle="--", color="#777777", linewidth=1.2, label="1:1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(r"Configured expert Vmin (mm$^3$)")
    ax.set_ylabel(r"LOSO-derived applied Vmin (mm$^3$)")
    ax.set_title("(A) Sample-level threshold recovery")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    x = np.arange(len(sample_ids))
    offsets = np.linspace(-0.24, 0.24, len(threshold_types))
    for offset, kind in zip(offsets, threshold_types):
        frame = results.loc[results["ThresholdType"] == kind].set_index("Sample").loc[sample_ids]
        y = 100.0 * frame["MinimumAxisAlignedFraction"].to_numpy()
        low = 100.0 * frame["MinimumAxisAlignedWilson95Low"].to_numpy()
        high = 100.0 * frame["MinimumAxisAlignedWilson95High"].to_numpy()
        ax.errorbar(
            x + offset,
            y,
            yerr=np.vstack([y - low, high - y]),
            fmt="o",
            color=COLORS[kind],
            capsize=2.5,
            markersize=5,
            label=LABELS[kind],
        )
    isotropic = 100.0 * 3.0 * (1.0 - np.cos(np.deg2rad(5.0)))
    ax.axhline(isotropic, color="#222222", linestyle="--", linewidth=1.2, label="Isotropic expectation")
    ax.set_xticks(x, sample_ids)
    ax.set_ylabel("Minimum axes within 5 degrees of a scan axis (%)")
    ax.set_title("(B) Residual scan-axis locking after filtering")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    positions = []
    tick_positions = []
    tick_labels = []
    for sample_index, sample_id in enumerate(sample_ids):
        base = sample_index * 4.0
        tick_positions.append(base + 1.0)
        tick_labels.append(sample_id)
        positions.extend([base, base + 1.0, base + 2.0])

    for ax, metric, title, ylabel in [
        (axes[1, 0], "Pprime", "(C) Fabric-strength sensitivity", "P prime"),
        (axes[1, 1], "T", "(D) Fabric-shape sensitivity", "T"),
    ]:
        arrays = [
            bootstrap_values[(sample_id, kind, metric)]
            for sample_id in sample_ids
            for kind in threshold_types
        ]
        box = ax.boxplot(
            arrays,
            positions=positions,
            widths=0.72,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
        )
        for patch, kind in zip(box["boxes"], threshold_types * len(sample_ids)):
            patch.set_facecolor(COLORS[kind])
            patch.set_alpha(0.82)
            patch.set_edgecolor("#333333")
        for median in box["medians"]:
            median.set_color("white")
            median.set_linewidth(1.4)
        ax.set_xticks(tick_positions, tick_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    axes[1, 0].axhline(1.0, color="#555555", linestyle=":", linewidth=1.0)
    axes[1, 1].axhline(0.0, color="#555555", linestyle=":", linewidth=1.0)

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=COLORS[kind], label=LABELS[kind])
        for kind in threshold_types
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    stem = repository / "outputs" / "cross_sample_threshold_audit"
    frame = build(repository, stem)
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(frame.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"Wrote {stem.with_suffix('.csv')}, .png, and .pdf")
