#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-sample agreement with expert volume-derived pseudo-labels.

Uses the same seven resolution-aware features as the GUI app: VoxelCount plus six
log-ellipsoid tensor components, with a LightGBM classifier (RandomForest fallback).
The evaluation mirrors the training pipeline. For every validation split,
StandardScaler is fitted on the training samples only and applied unchanged to the
held-out sample, LightGBM parameters are imported from the same canonical
configuration used by the analysis pipeline, and class weighting is balanced.

Pseudo-label = below the expert volume threshold (expert-threshold labelling,
identical to the app's ExpertThresholdProcessor). Because the pseudo-label is
defined from volume and VoxelCount is a predictor, these results quantify transfer
of the expert decision rule; they are not independent validation of physical
artifact identity.

  * Panel A  Cross-sample ROC (leave-one-sample-out) -> AUC per held-out sample
  * Panel B  Per held-out sample accuracy + AUC (LOSO)

Usage (from the repository root):
    python scripts/analysis/cross_validation.py \
        --data data/training \
        --config data/training/training_config.csv \
        --out outputs/S3_validation

The per-sample object tables (total<SampleID>.xlsx or .csv.gz) ship in the
``data/training`` folder; --config columns are SampleID,
ExpertThreshold_mm3, VoxelSize_mm and files
are matched by SampleID (xlsx, csv, or csv.gz).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from src.ml_threshold_selection.training_pipeline import lightgbm_parameters

VOL = "Volume3d (mm^3) "  # trailing space, as exported by Avizo / used by the app
SEED = 42
REFERENCE_LENGTH_MM = 1.0
FEATURE_NAMES = [
    'VoxelCount', 'L11', 'L22', 'L33', 'sqrt2_L12', 'sqrt2_L13', 'sqrt2_L23'
]


def make_classifier():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            **lightgbm_parameters(SEED), n_estimators=100, class_weight="balanced"
        ), "LightGBM"
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=SEED, class_weight="balanced"), "RandomForest"


def res_aware_features(df: pd.DataFrame, voxel_mm: float) -> np.ndarray:
    """7 resolution-aware features (mirrors res_aware_feature_engineering.py)."""
    vol = df[VOL].astype(float).to_numpy()
    voxel_count = vol / (voxel_mm ** 3)
    ev = df[["EigenVal1", "EigenVal2", "EigenVal3"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(ev)) or np.any(ev <= 0):
        raise ValueError("EigenVal1-3 must be finite and strictly positive")
    l = np.log(np.sqrt(5.0 * ev) / REFERENCE_LENGTH_MM)
    Q = np.stack([
        df[["EigenVec1X", "EigenVec1Y", "EigenVec1Z"]].to_numpy(),
        df[["EigenVec2X", "EigenVec2Y", "EigenVec2Z"]].to_numpy(),
        df[["EigenVec3X", "EigenVec3Y", "EigenVec3Z"]].to_numpy(),
    ], axis=1)
    if not np.all(np.isfinite(Q)):
        raise ValueError("EigenVec1-3 must be finite")
    u, _, vh = np.linalg.svd(Q)
    Q = u @ vh
    logE = np.zeros((len(df), 3, 3))
    logE[:, 0, 0], logE[:, 1, 1], logE[:, 2, 2] = -2 * l[:, 0], -2 * l[:, 1], -2 * l[:, 2]
    L = Q.transpose(0, 2, 1) @ logE @ Q
    L = 0.5 * (L + L.transpose(0, 2, 1))
    return np.column_stack([
        voxel_count, L[:, 0, 0], L[:, 1, 1], L[:, 2, 2],
        np.sqrt(2.0) * L[:, 0, 1], np.sqrt(2.0) * L[:, 0, 2], np.sqrt(2.0) * L[:, 1, 2],
    ])


def _read_table(path: Path) -> pd.DataFrame:
    df = (
        pd.read_excel(path)
        if path.suffix.lower() in (".xlsx", ".xls")
        else pd.read_csv(path, float_precision="round_trip")
    )
    if VOL not in df.columns:
        for c in df.columns:
            if str(c).strip() == VOL.strip():
                df = df.rename(columns={c: VOL}); break
    return df


def load_config(path: Path) -> dict[str, dict]:
    df = pd.read_csv(path)
    c = {x.lower(): x for x in df.columns}
    sid = c.get("sampleid", df.columns[0])
    thr = c.get("expertthreshold_mm3", c.get("threshold", df.columns[1]))
    vox = c.get("voxelsize_mm", c.get("voxel"))
    if vox is None:
        raise ValueError("Config must contain a VoxelSize_mm column")
    out = {}
    for _, r in df.iterrows():
        sample_id = str(r[sid])
        threshold = float(r[thr])
        voxel_size = float(r[vox])
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"Invalid expert threshold for {sample_id}")
        if not np.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError(f"Invalid voxel size for {sample_id}")
        out[sample_id] = {"thr": threshold, "vox": voxel_size}
    return out


def load_samples(data_dir: Path, cfg: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sid, meta in cfg.items():
        hits = (
            sorted(data_dir.glob(f"{sid}.xlsx"))
            or sorted(data_dir.glob(f"{sid}.csv"))
            or sorted(data_dir.glob(f"{sid}.csv.gz"))
            or sorted(data_dir.glob(f"*{sid}*.xlsx"))
            or sorted(data_dir.glob(f"*{sid}*.csv"))
            or sorted(data_dir.glob(f"*{sid}*.csv.gz"))
        )
        if not hits:
            print(f"  [skip] no table for {sid}"); continue
        df = _read_table(hits[0])
        vol = df[VOL].astype(float).to_numpy()
        y = (vol < meta["thr"]).astype(int)
        X = res_aware_features(df, meta["vox"])
        out[sid] = {"X": X, "y": y}
        print(f"  [ok]  {sid}: {len(df)} objects, {int(y.sum())} below-threshold pseudo-labels ({y.mean()*100:.1f}%)  [{hits[0].name}]")
    return out


def run(data_dir: Path, config: Path, out: Path) -> None:
    cfg = load_config(config)
    data = load_samples(data_dir, cfg)
    if len(data) < 2:
        sys.exit("need >=2 samples with data")
    clf_proto, clf_name = make_classifier()
    print(f"classifier: {clf_name}")
    ids = list(data)
    n_total = sum(len(d["y"]) for d in data.values())

    # ---- leave-one-sample-out ----
    roc, bars = {}, {}
    for held in ids:
        tr = [s for s in ids if s != held]
        Xtr = np.vstack([data[s]["X"] for s in tr]); ytr = np.concatenate([data[s]["y"] for s in tr])
        if ytr.min() == ytr.max():
            continue
        scaler = StandardScaler().fit(Xtr)
        Xtr_scaled = scaler.transform(Xtr)
        clf = clone(clf_proto); clf.fit(pd.DataFrame(Xtr_scaled, columns=FEATURE_NAMES), ytr)
        Xte, yte = data[held]["X"], data[held]["y"]
        Xte_scaled = scaler.transform(Xte)
        proba = clf.predict_proba(pd.DataFrame(Xte_scaled, columns=FEATURE_NAMES))[:, 1]
        auc = roc_auc_score(yte, proba) if yte.min() != yte.max() else float("nan")
        fpr, tpr, _ = roc_curve(yte, proba)
        roc[held] = (fpr, tpr, auc)
        bars[held] = (accuracy_score(yte, proba > 0.5) * 100, auc * 100)

    # ---- pooled stratified 5-fold ----
    Xall = np.vstack([data[s]["X"] for s in ids]); yall = np.concatenate([data[s]["y"] for s in ids])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_auc = []
    for tr_i, te_i in skf.split(Xall, yall):
        scaler = StandardScaler().fit(Xall[tr_i])
        clf = clone(clf_proto)
        clf.fit(pd.DataFrame(scaler.transform(Xall[tr_i]), columns=FEATURE_NAMES), yall[tr_i])
        p = clf.predict_proba(pd.DataFrame(scaler.transform(Xall[te_i]), columns=FEATURE_NAMES))[:, 1]
        fold_auc.append(roc_auc_score(yall[te_i], p))
    cv5 = float(np.mean(fold_auc))

    _plot(roc, bars, n_total, out)
    pd.DataFrame([
        {
            'HeldOutSample': sample,
            'AUC': roc[sample][2],
            'Accuracy': bars[sample][0] / 100.0,
            'LabelDefinition': 'Volume3d < expert threshold',
        }
        for sample in roc
    ]).to_csv(out.with_suffix('.csv'), index=False)
    print("\nLOSO summary:")
    for s in roc:
        print(f"  {s}: AUC={roc[s][2]:.3f}  acc={bars[s][0]:.1f}%")
    aucs = [v[2] for v in roc.values() if v[2] == v[2]]
    if aucs:
        print(f"LOSO AUC range {min(aucs):.3f}-{max(aucs):.3f}")
    print(f"pooled object-level 5-fold AUC (non-independent diagnostic) = {cv5:.3f}")


def _plot(roc, bars, n_total, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 6.0))
    fig.suptitle(f"Cross-sample agreement with expert volume-derived pseudo-labels "
                 f"(5 samples, n = {n_total:,} objects)",
                 fontsize=14, y=0.99)
    for s, (fpr, tpr, auc) in roc.items():
        ax[0].plot(fpr, tpr, lw=1.7, label=f"{s} (AUC={auc:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate",
              title="(A) Pseudo-label ROC (leave-one-sample-out)")
    ax[0].legend(loc="lower right", fontsize=9)
    xs = list(bars); x = np.arange(len(xs)); w = 0.38
    ax[1].bar(x - w/2, [bars[s][0] for s in xs], w, label="Accuracy", color="#4C72B0")
    ax[1].bar(x + w/2, [bars[s][1] for s in xs], w, label="AUC", color="#DD8452")
    ax[1].set(xticks=x, ylim=(60, 100), ylabel="%", title="(B) Per held-out sample (LOSO)")
    ax[1].set_xticklabels(xs, rotation=20); ax[1].legend(loc="lower left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.with_suffix('.png')} and .pdf")


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-sample evaluation of expert-derived pseudo-label agreement (AUC)")
    p.add_argument("--data", type=Path, required=True, help="dir with per-sample training tables")
    p.add_argument("--config", type=Path, required=True, help="CSV: SampleID,ExpertThreshold_mm3,VoxelSize_mm")
    p.add_argument("--out", type=Path, default=Path("S3_validation"))
    a = p.parse_args()
    run(a.data, a.config, a.out)


if __name__ == "__main__":
    main()
