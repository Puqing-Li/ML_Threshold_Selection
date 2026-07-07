#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stand-alone cross-validation of the artifact classifier (reviewer M2 / R1-7 / R2-3).

Reproduces the reported discrimination performance on the five training samples:
under leave-one-sample-out and stratified five-fold cross-validation the classifier
attains an AUC of ~0.96-0.99. Uses the SAME seven resolution-aware features as the
GUI app (src/features/res_aware_feature_engineering.py): VoxelCount + six
log-ellipsoid tensor components, with a LightGBM classifier (RandomForest fallback).
Tree models are invariant to the per-feature StandardScaler used in the GUI, so the
scaler is omitted here without changing AUC.

Ground-truth label = artifact if  Volume3d < expert threshold  (semi-supervised
labelling, identical to the app's ExpertThresholdProcessor).

  * Panel A  Cross-sample ROC (leave-one-sample-out) -> AUC per held-out sample
  * Panel B  Per held-out sample accuracy + AUC (LOSO) and pooled 5-fold AUC

The per-sample ML-vs-expert minimum-volume threshold (S3 Fig panel C) is produced
by the GUI app's own threshold module (src/ml_threshold_selection/prediction_analysis.py,
compute_dual_thresholds) and is not recomputed here.

Usage (from the repository root):
    python cross_validation.py \
        --data "trained model" \
        --config examples/expert_thresholds.csv \
        --out S3_validation

The per-sample grain tables (total<SampleID>.xlsx) ship in the "trained model"
folder; --config columns are SampleID, ExpertThreshold_mm3, VoxelSize_mm and files
are matched by SampleID (xlsx or csv).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

VOL = "Volume3d (mm^3) "  # trailing space, as exported by Avizo / used by the app
SEED = 42


def make_classifier():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            objective="binary", metric="auc", boosting_type="gbdt", num_leaves=31,
            learning_rate=0.05, feature_fraction=0.9, bagging_fraction=0.8,
            bagging_freq=5, verbose=-1, random_state=SEED, class_weight="balanced"), "LightGBM"
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=SEED, class_weight="balanced"), "RandomForest"


def res_aware_features(df: pd.DataFrame, voxel_mm: float) -> np.ndarray:
    """7 resolution-aware features (mirrors res_aware_feature_engineering.py)."""
    vol = df[VOL].astype(float).to_numpy()
    voxel_count = vol / (voxel_mm ** 3)
    ev = df[["EigenVal1", "EigenVal2", "EigenVal3"]].to_numpy()
    l = np.log(np.sqrt(np.clip(ev, 1e-30, None)))
    Q = np.stack([
        df[["EigenVec1X", "EigenVec1Y", "EigenVec1Z"]].to_numpy(),
        df[["EigenVec2X", "EigenVec2Y", "EigenVec2Z"]].to_numpy(),
        df[["EigenVec3X", "EigenVec3Y", "EigenVec3Z"]].to_numpy(),
    ], axis=1)
    norms = np.linalg.norm(Q, axis=2, keepdims=True)
    norms[norms == 0] = 1.0
    Q = Q / norms
    logE = np.zeros((len(df), 3, 3))
    logE[:, 0, 0], logE[:, 1, 1], logE[:, 2, 2] = -2 * l[:, 0], -2 * l[:, 1], -2 * l[:, 2]
    L = np.einsum("nij,njk,nlk->nil", Q.transpose(0, 2, 1), logE, Q)
    return np.column_stack([
        voxel_count, L[:, 0, 0], L[:, 1, 1], L[:, 2, 2],
        np.sqrt(2.0) * L[:, 0, 1], np.sqrt(2.0) * L[:, 0, 2], np.sqrt(2.0) * L[:, 1, 2],
    ])


def _read_table(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
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
    out = {}
    for _, r in df.iterrows():
        out[str(r[sid])] = {"thr": float(r[thr]), "vox": float(r[vox]) if vox else 0.03}
    return out


def load_samples(data_dir: Path, cfg: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sid, meta in cfg.items():
        hits = (sorted(data_dir.glob(f"{sid}.xlsx")) or sorted(data_dir.glob(f"{sid}.csv"))
                or sorted(data_dir.glob(f"*{sid}*.xlsx")) or sorted(data_dir.glob(f"*{sid}*.csv")))
        if not hits:
            print(f"  [skip] no table for {sid}"); continue
        df = _read_table(hits[0])
        vol = df[VOL].astype(float).to_numpy()
        y = (vol < meta["thr"]).astype(int)
        X = res_aware_features(df, meta["vox"])
        out[sid] = {"X": X, "y": y}
        print(f"  [ok]  {sid}: {len(df)} grains, {int(y.sum())} artifacts ({y.mean()*100:.1f}%)  [{hits[0].name}]")
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
        clf = clone(clf_proto); clf.fit(Xtr, ytr)
        Xte, yte = data[held]["X"], data[held]["y"]
        proba = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, proba) if yte.min() != yte.max() else float("nan")
        fpr, tpr, _ = roc_curve(yte, proba)
        roc[held] = (fpr, tpr, auc)
        bars[held] = (accuracy_score(yte, proba > 0.5) * 100, auc * 100)

    # ---- pooled stratified 5-fold ----
    Xall = np.vstack([data[s]["X"] for s in ids]); yall = np.concatenate([data[s]["y"] for s in ids])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_auc = []
    for tr_i, te_i in skf.split(Xall, yall):
        clf = clone(clf_proto); clf.fit(Xall[tr_i], yall[tr_i])
        p = clf.predict_proba(Xall[te_i])[:, 1]
        fold_auc.append(roc_auc_score(yall[te_i], p))
    cv5 = float(np.mean(fold_auc))

    _plot(roc, bars, cv5, n_total, out)
    print("\nLOSO summary:")
    for s in roc:
        print(f"  {s}: AUC={roc[s][2]:.3f}  acc={bars[s][0]:.1f}%")
    aucs = [v[2] for v in roc.values() if v[2] == v[2]]
    if aucs:
        print(f"LOSO AUC range {min(aucs):.3f}-{max(aucs):.3f}")
    print(f"pooled 5-fold AUC = {cv5:.3f}")


def _plot(roc, bars, cv5, n_total, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 6.0))
    fig.suptitle(f"S3. Artifact-classifier cross-validation (5 training samples, n = {n_total:,} grains)",
                 fontsize=14, y=0.99)
    for s, (fpr, tpr, auc) in roc.items():
        ax[0].plot(fpr, tpr, lw=1.7, label=f"{s} (AUC={auc:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate",
              title="(A) Cross-sample ROC (leave-one-sample-out)")
    ax[0].legend(loc="lower right", fontsize=9)
    xs = list(bars); x = np.arange(len(xs)); w = 0.38
    ax[1].bar(x - w/2, [bars[s][0] for s in xs], w, label="Accuracy", color="#4C72B0")
    ax[1].bar(x + w/2, [bars[s][1] for s in xs], w, label="AUC", color="#DD8452")
    ax[1].axhline(cv5 * 100, ls="--", color="k", lw=1, label=f"pooled 5-fold AUC={cv5:.3f}")
    ax[1].set(xticks=x, ylim=(60, 100), ylabel="%", title="(B) Per held-out sample (LOSO)")
    ax[1].set_xticklabels(xs, rotation=20); ax[1].legend(loc="lower left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.with_suffix('.png')} and .pdf")


def main() -> None:
    p = argparse.ArgumentParser(description="Artifact-classifier cross-validation (AUC)")
    p.add_argument("--data", type=Path, required=True, help="dir with per-sample training tables")
    p.add_argument("--config", type=Path, required=True, help="CSV: SampleID,ExpertThreshold_mm3,VoxelSize_mm")
    p.add_argument("--out", type=Path, default=Path("S3_validation"))
    a = p.parse_args()
    run(a.data, a.config, a.out)


if __name__ == "__main__":
    main()
