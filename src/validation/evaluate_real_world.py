"""Zero-shot real-world validation of the digital twin on the Shanghai T2DM cohort.

The fused regressor and the adverse-event classifier were trained ONLY on
synthetic (Bergman-ODE) data. Here we run them, with no retraining and no
fine-tuning, on 100 real Type 2 diabetes patients from the Shanghai dataset
(Zhao et al., Sci Data 2023) and report honest transfer performance.

Why this matters: it directly addresses the central weakness of any synthetic
digital-twin project — "does it work on real patients?" — and quantifies the
sim-to-real gap instead of hiding it. We also report a persistence baseline
computed on the SAME real data, which is the fair bar: a useful model must beat
"assume glucose stays where it is" on real patients, not just on its own sim.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, precision_score, recall_score, f1_score,
)

PROCESSED = "data/processed"
REAL_CSV = os.path.join(PROCESSED, "features_real_shanghai_t2dm.csv")


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "median_ae": round(float(np.median(abs_err)), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def evaluate() -> Dict:
    feature_names = json.load(open(os.path.join(PROCESSED, "feature_names.json")))
    all_cols = feature_names["all"]

    df = pd.read_csv(REAL_CSV)
    X = df[all_cols].values.astype(np.float32)
    y = df["target_glucose_2h"].values.astype(np.float32)
    y_cls = df["target_adverse"].values.astype(np.int32)
    cgm_idx = all_cols.index("cgm_current")

    # --- Load models trained on synthetic data only ---
    reg = xgb.XGBRegressor()
    reg.load_model(os.path.join(PROCESSED, "xgb_fused_regressor.json"))
    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(PROCESSED, "xgb_fused_classifier.json"))

    # --- Zero-shot regression on real data ---
    y_pred = reg.predict(X)
    reg_metrics = _reg_metrics(y, y_pred)

    # --- Persistence baseline on the SAME real data (the fair bar) ---
    y_persist = X[:, cgm_idx]
    persist_metrics = _reg_metrics(y, y_persist)

    # --- Adverse-event classifier on real data ---
    cls_block: Dict[str, float] = {}
    if len(np.unique(y_cls)) > 1:
        proba = clf.predict_proba(X)[:, 1]
        pred_cls = clf.predict(X)
        cls_block = {
            "auroc": round(float(roc_auc_score(y_cls, proba)), 4),
            "precision": round(float(precision_score(y_cls, pred_cls, zero_division=0)), 4),
            "recall": round(float(recall_score(y_cls, pred_cls, zero_division=0)), 4),
            "f1": round(float(f1_score(y_cls, pred_cls, zero_division=0)), 4),
            "prevalence_pct": round(float(y_cls.mean() * 100), 2),
        }

    # --- Reference: synthetic held-out performance (for the sim-to-real gap) ---
    synth = json.load(open(os.path.join(PROCESSED, "evaluation_results.json")))
    synth_fused_rmse = synth["fused"]["rmse"]

    improvement_vs_persist = round(
        (persist_metrics["rmse"] - reg_metrics["rmse"]) / persist_metrics["rmse"] * 100, 1
    )

    results = {
        "dataset": "ShanghaiT2DM (Zhao et al., Scientific Data 2023)",
        "evaluation": "zero-shot transfer (trained on synthetic only, no fine-tuning)",
        "n_patients": int(df["patient_id"].nunique()),
        "n_windows": int(len(df)),
        "real_target_mean_mg_dl": round(float(y.mean()), 1),
        "fused_model_real_world": reg_metrics,
        "persistence_baseline_real_world": persist_metrics,
        "fused_improvement_vs_persistence_pct": improvement_vs_persist,
        "adverse_classifier_real_world": cls_block,
        "sim_to_real_gap": {
            "synthetic_test_rmse": synth_fused_rmse,
            "real_world_rmse": reg_metrics["rmse"],
            "absolute_gap_mg_dl": round(reg_metrics["rmse"] - synth_fused_rmse, 2),
        },
        "notes": [
            "Models were trained exclusively on synthetic Bergman-ODE data and "
            "evaluated on real patients with no retraining.",
            "Shanghai provides real CGM + EHR but no consumer-wearable streams; "
            "absent wearable features were neutralized to training medians, so this "
            "validates the CGM + EHR pathway of the fused model.",
            "The persistence baseline is computed on the same real windows and is "
            "the fair bar a useful 2-hour forecaster must beat.",
        ],
    }
    return results


if __name__ == "__main__":
    res = evaluate()
    out = os.path.join(PROCESSED, "real_world_results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    print("=" * 64)
    print("  REAL-WORLD ZERO-SHOT VALIDATION — ShanghaiT2DM (100 patients)")
    print("=" * 64)
    print(f"Windows evaluated      : {res['n_windows']:,}")
    print(f"Real 2h-target mean    : {res['real_target_mean_mg_dl']} mg/dL")
    print("-" * 64)
    rm = res["fused_model_real_world"]
    pb = res["persistence_baseline_real_world"]
    print(f"{'Model':<26}{'RMSE':>8}{'MAE':>8}{'medAE':>8}{'R2':>9}")
    print(f"{'Persistence (real)':<26}{pb['rmse']:>8}{pb['mae']:>8}{pb['median_ae']:>8}{pb['r2']:>9}")
    print(f"{'Fused twin (real)':<26}{rm['rmse']:>8}{rm['mae']:>8}{rm['median_ae']:>8}{rm['r2']:>9}")
    print("-" * 64)
    print(f"Fused beats persistence by {res['fused_improvement_vs_persistence_pct']}% RMSE on REAL data")
    g = res["sim_to_real_gap"]
    print(f"Sim-to-real gap        : synthetic {g['synthetic_test_rmse']} -> real {g['real_world_rmse']} mg/dL "
          f"(+{g['absolute_gap_mg_dl']})")
    if res["adverse_classifier_real_world"]:
        c = res["adverse_classifier_real_world"]
        print(f"Adverse classifier     : AUROC {c['auroc']} | recall {c['recall']} | "
              f"precision {c['precision']} | prevalence {c['prevalence_pct']}%")
    print("=" * 64)
    print(f"Saved {out}")
