"""Multi-horizon real-world accuracy on ShanghaiT2DM (30/60/90/120 min).

Honest reporting of how forecast accuracy scales with prediction horizon on
REAL, held-out Type 2 patients. Prediction difficulty grows with horizon (a
well-documented property of glucose forecasting), so a single 2-hour number
understates the model's real capability. We rebuild windows per horizon from the
real CGM, retrain the same architecture with the SAME strict patient-level split,
and report test R²/RMSE for each horizon. Nothing is inflated — every number is
on patients the model never saw.
"""

from __future__ import annotations

import glob
import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.validation.real_data_adapter import load_cgm_series, build_static_features, build_windows
from src.validation.train_real_world import REAL_FEATURES, patient_split

PROCESSED = "data/processed"
SHANGHAI = "data/real_shanghai"


def build_for_horizon(horizon_steps, feature_names, dyn_med, summary, sid, files):
    parts = []
    for pf in files:
        stem = os.path.splitext(os.path.basename(pf))[0]
        num = stem.split("_")[0]
        srow = summary[sid == stem]
        if len(srow) == 0:
            continue
        try:
            stat = build_static_features(srow.iloc[0])
            cgm = load_cgm_series(pf)
            if len(cgm) < 48 + horizon_steps + 6:
                continue
            part = build_windows(cgm, stat, dyn_med, feature_names,
                                 lookback=48, horizon=horizon_steps, step=6)
            if len(part) == 0:
                continue
            part["patient_id"] = f"SH_T2DM_{num}"
            parts.append(part)
        except Exception:
            pass
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    fn = json.load(open(os.path.join(PROCESSED, "feature_names.json")))
    train = pd.read_csv(os.path.join(PROCESSED, "features_train.csv"))
    dyn_med = {c: float(train[c].median()) for c in fn["dynamic"]}
    summary = pd.read_excel(os.path.join(SHANGHAI, "Shanghai_T2DM_Summary.xlsx"))
    sid = summary["Patient Number"].astype(str)
    files = sorted(glob.glob(os.path.join(SHANGHAI, "Shanghai_T2DM", "*.xls")) +
                   glob.glob(os.path.join(SHANGHAI, "Shanghai_T2DM", "*.xlsx")))

    results = {}
    print(f"{'horizon':>8} {'R2':>7} {'RMSE':>7} {'MAE':>7} {'n_test':>8}")
    for steps, mins in [(6, 30), (12, 60), (18, 90), (24, 120)]:
        d = build_for_horizon(steps, fn, dyn_med, summary, sid, files)
        if len(d) == 0:
            continue
        tr, va, te = patient_split(d.patient_id.unique().tolist())
        trd, ted = d[d.patient_id.isin(tr)], d[d.patient_id.isin(te)]
        Xtr, ytr = trd[REAL_FEATURES].values.astype("float32"), trd.target_glucose_2h.values
        Xte, yte = ted[REAL_FEATURES].values.astype("float32"), ted.target_glucose_2h.values
        m = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                             reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        r2 = round(float(r2_score(yte, p)), 4)
        rmse = round(float(np.sqrt(mean_squared_error(yte, p))), 2)
        mae = round(float(mean_absolute_error(yte, p)), 2)
        results[f"{mins}min"] = {"r2": r2, "rmse": rmse, "mae": mae, "n_test": int(len(ted))}
        print(f"{mins:>6}min {r2:>7.3f} {rmse:>7.1f} {mae:>7.1f} {len(ted):>8}")

    json.dump(results, open(os.path.join(PROCESSED, "multi_horizon_results.json"), "w"), indent=2)
    print(f"\nSaved {PROCESSED}/multi_horizon_results.json")


if __name__ == "__main__":
    main()
