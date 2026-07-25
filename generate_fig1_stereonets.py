#!/usr/bin/env python3
"""Generate corrected LE01 principal-axis stereonets for manuscript Fig 1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import RectBivariateSpline


TOMOFAB_COMMIT = "2697865623c3afa34626abdd765183825180a069"
TOMOFAB_URL = "https://github.com/benpetri/tomofab"
SIGMA = 3.0
BASE_GRID_N = 50
INTERPOLATION_STEP = 0.2
CONTOUR_INTERVALS = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, float_precision="round_trip")


def _unit_axial_vectors(data: pd.DataFrame, axis_number: int) -> np.ndarray:
    columns = [
        f"EigenVec{axis_number}X",
        f"EigenVec{axis_number}Y",
        f"EigenVec{axis_number}Z",
    ]
    vectors = data[columns].to_numpy(dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(vectors)) or np.any(~np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError(f"EigenVec{axis_number} contains invalid vectors")
    return vectors / norms[:, None]


def modified_kamb_mud_grid(
    vectors: np.ndarray,
    grid_n: int = BASE_GRID_N,
    sigma: float = SIGMA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the modified-Kamb m.u.d. formula used by TomoFab DataDens.m.

    This independent Python implementation is regression-checked against the
    published TomoFab output. Unoriented axes are handled with
    ``abs(dot(u, v))``. The evaluation grid is the lower-hemisphere Schmidt
    net used by TomoFab. Dividing its three-sigma density grid by ``sigma``
    gives multiples of uniform density.
    """
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or len(vectors) == 0:
        raise ValueError("vectors must be a non-empty n by 3 array")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(vectors)) or np.any(norms == 0):
        raise ValueError("vectors contain non-finite or zero-length rows")
    vectors = vectors / norms[:, None]

    n_data = len(vectors)
    kamb_f = 2.0 * (1.0 + n_data / (sigma * sigma))
    z_unit = np.sqrt(n_data * (kamb_f * 0.5 - 1.0) / (kamb_f * kamb_f))
    coordinates = np.linspace(-1.0, 1.0, grid_n)
    grid = np.empty((grid_n, grid_n), dtype=float)
    for i, x_coord in enumerate(coordinates):
        for j, y_coord in enumerate(coordinates):
            radius_sq = x_coord * x_coord + y_coord * y_coord
            projection_scale = np.sqrt(abs(2.0 - radius_sq))
            direction = np.array([
                projection_scale * x_coord,
                projection_scale * y_coord,
                -(1.0 - radius_sq),
            ])
            weights = np.exp(kamb_f * (np.abs(vectors @ direction) - 1.0))
            grid[i, j] = weights.sum() / z_unit / sigma
    return coordinates, coordinates, grid


def _interpolated_net(
    coordinates: np.ndarray,
    grid: np.ndarray,
    step: float = INTERPOLATION_STEP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = int(round((len(coordinates) - 1) / step)) + 1
    fine = np.linspace(-1.0, 1.0, points)
    interpolator = RectBivariateSpline(coordinates, coordinates, grid, kx=1, ky=1)
    density = interpolator(fine, fine).T
    x_grid, y_grid = np.meshgrid(fine, fine)
    density[x_grid * x_grid + y_grid * y_grid > 1.0 + 1e-12] = np.nan
    return x_grid, y_grid, density


def _axis_colormap(axis_number: int) -> LinearSegmentedColormap:
    colors = {
        1: ("#FFFFFF", "#F5A3A3", "#B2182B"),
        2: ("#FFFFFF", "#9ED7A3", "#1B7837"),
        3: ("#FFFFFF", "#AFC6E9", "#21468B"),
    }
    return LinearSegmentedColormap.from_list(f"axis_{axis_number}", colors[axis_number])


def _draw_net(ax: plt.Axes) -> None:
    boundary = plt.Circle((0.0, 0.0), 1.0, fill=False, color="#222222", lw=0.8)
    ax.add_patch(boundary)
    ax.text(-1.08, 0.0, "X", ha="right", va="center", fontsize=8)
    ax.text(0.0, -1.08, "Y", ha="center", va="top", fontsize=8)
    ax.text(0.0, 0.0, "Z", ha="center", va="center", fontsize=7, color="#333333")
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")


def generate(args: argparse.Namespace) -> dict[str, object]:
    data = _read_table(args.input)
    volume_column = "Volume3d (mm^3) "
    if volume_column not in data:
        raise ValueError(f"Missing required column: {volume_column!r}")
    voxel_counts = data[volume_column].to_numpy(dtype=float) / args.voxel_size_mm ** 3
    stages = [
        ("Prefiltered", 0, np.ones(len(data), dtype=bool)),
        ("Loose", args.loose_vox, voxel_counts >= args.loose_vox),
        ("Strict", args.strict_vox, voxel_counts >= args.strict_vox),
    ]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "le01-fig1-modified-kamb",
    })
    fig, axes = plt.subplots(3, 3, figsize=(7.4, 7.2), constrained_layout=True)
    grid_rows: list[dict[str, object]] = []
    panel_records: list[dict[str, object]] = []

    for row, (stage_name, threshold, mask) in enumerate(stages):
        retained = data.loc[mask]
        for column, axis_number in enumerate((1, 2, 3)):
            vectors = _unit_axial_vectors(retained, axis_number)
            x_base, y_base, base_grid = modified_kamb_mud_grid(vectors)
            x_grid, y_grid, density = _interpolated_net(x_base, base_grid)
            finite = density[np.isfinite(density)]
            level_min = float(finite.min())
            level_max = float(finite.max())
            levels = np.linspace(level_min, level_max, CONTOUR_INTERVALS + 1)

            ax = axes[row, column]
            fill = ax.contourf(
                x_grid,
                y_grid,
                density,
                levels=levels,
                cmap=_axis_colormap(axis_number),
                antialiased=True,
            )
            ax.contour(
                x_grid,
                y_grid,
                density,
                levels=levels[1:-1],
                colors="#333333",
                linewidths=0.35,
                alpha=0.75,
            )
            _draw_net(ax)
            if row == 0:
                ax.set_title(rf"$\Phi_{axis_number}$", fontsize=11, pad=4)
            if column == 0:
                ax.text(
                    -0.23,
                    0.5,
                    f"({chr(97 + row)})  {stage_name}\n$n$ = {len(retained):,}",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=9,
                )
            colorbar = fig.colorbar(fill, ax=ax, fraction=0.046, pad=0.02)
            colorbar.set_label("m.u.d.", fontsize=7)
            colorbar.ax.tick_params(labelsize=6)

            panel_records.append({
                "stage": stage_name,
                "threshold_vox": threshold,
                "retained_n": int(len(retained)),
                "principal_axis": axis_number,
                "base_grid_n": BASE_GRID_N,
                "sigma": SIGMA,
                "raw_grid_min_mud": float(base_grid.min()),
                "raw_grid_max_mud": float(base_grid.max()),
                "plotted_min_mud": level_min,
                "plotted_max_mud": level_max,
                "contour_levels_mud": [float(value) for value in levels],
            })
            for i, x_coord in enumerate(x_base):
                for j, y_coord in enumerate(y_base):
                    grid_rows.append({
                        "stage": stage_name,
                        "threshold_vox": threshold,
                        "retained_n": int(len(retained)),
                        "principal_axis": axis_number,
                        "grid_x": float(x_coord),
                        "grid_y": float(y_coord),
                        "mud": float(base_grid[i, j]),
                        "inside_equal_area_net": bool(x_coord * x_coord + y_coord * y_coord <= 1.0),
                    })

    args.output.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".pdf": {"metadata": {"Creator": "ML Threshold Selection", "CreationDate": None, "ModDate": None}},
        ".svg": {"metadata": {"Creator": "ML Threshold Selection", "Date": None}},
        ".png": {"dpi": 600},
        ".tif": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        fig.savefig(args.output / f"Fig1{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    tif_path = args.output / "Fig1.tif"
    with Image.open(tif_path) as source:
        rgba = source.convert("RGBA")
        white = Image.new("RGBA", rgba.size, "white")
        flattened = Image.alpha_composite(white, rgba).convert("RGB")
        flattened.thumbnail((4500, 5250), Image.Resampling.LANCZOS)
        flattened.save(tif_path, compression="tiff_lzw", dpi=(600, 600))
    plt.close(fig)

    grid_path = args.output / "Fig1_modified_kamb_grid.csv.gz"
    pd.DataFrame(grid_rows).to_csv(
        grid_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    metadata = {
        "input_table": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "input_status": "legacy derivative; raw exclusions cannot be reconstructed",
        "voxel_edge_mm": args.voxel_size_mm,
        "voxel_geometry_status": "scalar legacy input; physical volumes conditional on isotropic reconstruction",
        "loose_threshold_vox": args.loose_vox,
        "strict_threshold_vox": args.strict_vox,
        "retention_rule": "VoxelCount >= Vmin",
        "projection": "lower-hemisphere equal-area Schmidt net",
        "axis_treatment": "unoriented axial data; antipodes combined by abs(dot)",
        "density_method": "TomoFab modified Kamb method after Vollmer (1995)",
        "tomofab_repository": TOMOFAB_URL,
        "tomofab_commit": TOMOFAB_COMMIT,
        "tomofab_source_function": "DataDens.m",
        "sigma": SIGMA,
        "base_grid": f"{BASE_GRID_N} x {BASE_GRID_N}",
        "display_interpolation_step_in_base_grid_cells": INTERPOLATION_STEP,
        "contour_rule": f"{CONTOUR_INTERVALS} equal intervals from panel minimum to maximum m.u.d.",
        "mud_normalization": "TomoFab three-sigma grid divided by sigma=3",
        "panels": panel_records,
        "source_grid": grid_path.name,
    }
    (args.output / "Fig1_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8", newline="\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-size-mm", type=float, default=0.03)
    parser.add_argument("--loose-vox", type=int, default=50)
    parser.add_argument("--strict-vox", type=int, default=154)
    metadata = generate(parser.parse_args())
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
