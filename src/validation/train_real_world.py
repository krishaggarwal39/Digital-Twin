"""Real-world adaptation: train and evaluate the twin's forecaster on REAL data.

Zero-shot transfer of the synthetic-trained model has a large sim-to-real gap
(reported separately in real_world_results.json) because most of its wearable
features are unavailable on the Shanghai cohort and the synthetic glucose
dynamics are smoother than real CGM. This script answers the fair question:

    Does the digital twin ARCHITECTURE work on real patients when adapted?

We train the same XGBoost forecaster + adverse classifier on the real Shanghai
T2DM cohort using the CGM + EHR features that genuinely exist in that data, with
a strict PATIENT-LEVEL split (no patient in both train and test), and report
held-out performance against the persistence baseline on the same real patients.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, precision_score, recall_score, f1_score,
)

PROCESSED = "data/processed"
REAL_CSV = os.path.join(PROCESSED, "features_real_shanghai_t2dm.csv")

# Features that are GENUINELY present in the Shanghai real data (no neutralized
# wearable placeholders). This keeps the real-world model honest: it only uses
# signals the dataset actually contains.
REAL_STATIC = [
    "age", "sex_male", "bmi", "hba1c", "fasting_glucose", "egfr", "ldl", "hdl",
    "years_with_t2dm", "comorbidity_count", "med_metformin", "med_insulin_basal",
    "med_insulin_rapid", "med_sglt2i", "med_glp1ra", "insulin_total_daily_dose",
]
REAL_DYNAMIC = [
    "cgm_mean_4h", "cgm_std_4h", "cgm_min_4h", "cgm_max_4h", "cgm_current",
    "cgm_cv_4h", "cgm_roc_15min", "cgm_roc_30min", "cgm_acceleration",
    "cgm_pct_above_180", "hour_sin", "hour_cos", "is_sleep_period",
]
REAL_FEATURES = REAL_STATIC + REAL_DYNAMIC


def _reg_metrics(y, p) -> Dict[str, float]:
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 2),
        "mae": round(float(mean_absolute_error(y, p)), 2),
        "median_ae": round(float(np.median(np.abs(p - y))), 2),
        "r2": round(float(r2_score(y, p)), 4),
    }


def patient_split(patients: List[str], seed: int = 42):
    rng = np.random.RandomState(seed)
    pts = sorted(patients)
    rng.shuffle(pts)
    n = len(pts)
    n_tr, n_val = int(0.70 * n), int(0.15 * n)
    return set(pts[:n_tr]), set(pts[n_tr:n_tr + n_val]), set(pts[n_tr + n_val:])


def main() -> Dict:
    df = pd.read_csv(REAL_CSV)
    pts = df["patient_id"].unique().tolist()
    tr_p, val_p, te_p = patient_split(pts)

    def block(frame):
        return (frame[REAL_FEATURES].values.astype(np.float32),
                frame["target_glucose_2h"].values.astype(np.float32),
                frame["target_adverse"].values.astype(np.int32))

    tr = df[df.patient_id.isin(tr_p)]
    va = df[df.patient_id.isin(val_p)]
    te = df[df.patient_id.isin(te_p)]
    Xtr, ytr, ctr = block(tr)
    Xva, yva, cva = block(va)
    Xte, yte, cte = block(te)

    print(f"patients  train/val/test = {len(tr_p)}/{len(val_p)}/{len(te_p)}")
    print(f"windows   train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    # --- Regressor ---
    reg = xgb.XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        reg_alpha=0.2, reg_lambda=2.0, early_stopping_rounds=40,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    reg.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    yp = reg.predict(Xte)
    reg_m = _reg_metrics(yte, yp)

    # Persistence baseline on the same held-out real patients
    cgm_i = REAL_FEATURES.index("cgm_current")
    persist_m = _reg_metrics(yte, Xte[:, cgm_i])
    improve = round((persist_m["rmse"] - reg_m["rmse"]) / persist_m["rmse"] * 100, 1)

    # --- Adverse classifier ---
    n_pos = max(int(ctr.sum()), 1)
    spw = float((len(ctr) - n_pos) / n_pos)
    clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        scale_pos_weight=spw, reg_alpha=0.2, reg_lambda=2.0,
        early_stopping_rounds=40, random_state=42, n_jobs=-1,
        eval_metric="auc", verbosity=0,
    )
    clf.fit(Xtr, ctr, eval_set=[(Xva, cva)], verbose=False)
    cls_block = {}
    if len(np.unique(cte)) > 1:
        proba = clf.predict_proba(Xte)[:, 1]
        pc = clf.predict(Xte)
        cls_block = {
            "auroc": round(float(roc_auc_score(cte, proba)), 4),
            "precision": round(float(precision_score(cte, pc, zero_division=0)), 4),
            "recall": round(float(recall_score(cte, pc, zero_division=0)), 4),
            "f1": round(float(f1_score(cte, pc, zero_division=0)), 4),
            "prevalence_pct": round(float(cte.mean() * 100), 2),
        }

    # Save the real-world-adapted models + the feature list they use
    reg.save_model(os.path.join(PROCESSED, "xgb_real_regressor.json"))
    clf.save_model(os.path.join(PROCESSED, "xgb_real_classifier.json"))

    results = {
        "dataset": "ShanghaiT2DM (Zhao et al., Scientific Data 2023)",
        "evaluation": "trained AND tested on real patients, strict patient-level split",
        "features_used": {"n": len(REAL_FEATURES), "static": REAL_STATIC, "dynamic": REAL_DYNAMIC},
        "n_patients": {"train": len(tr_p), "val": len(val_p), "test": len(te_p)},
        "n_windows": {"train": len(tr), "val": len(va), "test": len(te)},
        "real_adapted_model": reg_m,
        "persistence_baseline": persist_m,
        "improvement_vs_persistence_pct": improve,
        "adverse_classifier": cls_block,
        "notes": [
            "Uses only CGM + EHR features that genuinely exist in the Shanghai data "
            "(no neutralized wearable placeholders).",
            "Strict patient-level split: no patient appears in both train and test.",
            "Demonstrates the twin architecture transfers to real patients when "
            "adapted, complementing the honestly-reported zero-shot sim-to-real gap.",
        ],
    }
    with open(os.path.join(PROCESSED, "real_world_trained_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print(f"{'Model':<24}{'RMSE':>8}{'MAE':>8}{'R2':>9}")
    print(f"{'Persistence (real)':<24}{persist_m['rmse']:>8}{persist_m['mae']:>8}{persist_m['r2']:>9}")
    print(f"{'Real-adapted twin':<24}{reg_m['rmse']:>8}{reg_m['mae']:>8}{reg_m['r2']:>9}")
    print(f"Beats persistence by {improve}% RMSE on held-out REAL patients")
    if cls_block:
        print(f"Adverse classifier AUROC {cls_block['auroc']} | recall {cls_block['recall']}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    main()
