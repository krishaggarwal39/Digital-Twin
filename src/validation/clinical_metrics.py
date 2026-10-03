"""Clinical safety metrics for the real-world-adapted glucose forecaster.

Beyond RMSE/MAE/R2, a clinical audience evaluates glucose predictions with
metrics tied to patient safety:

  * Clarke Error Grid Analysis (EGA) — the standard introduced by Clarke et al.
    (1987) for blood-glucose estimation. Each (reference, prediction) pair falls
    into a zone:
        A  clinically accurate (no effect on treatment)
        B  benign error (would not lead to inappropriate treatment)
        C  overcorrection errors
        D  dangerous failure to detect / treat
        E  erroneous treatment (opposite direction)
    A high A+B percentage is what clinicians require of a usable system.

  * Adverse-event lead time — for true hyperglycemic excursions the model flags,
    how far in advance (minutes) does the warning come? The whole premise of the
    twin is proactive warning, so we quantify it.

Run after train_real_world.py (uses xgb_real_regressor/classifier on the real
held-out test patients).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from src.validation.train_real_world import REAL_FEATURES, patient_split

PROCESSED = "data/processed"
REAL_CSV = os.path.join(PROCESSED, "features_real_shanghai_t2dm.csv")


def clarke_zone(ref: float, pred: float) -> str:
    """Return the Clarke Error Grid zone ('A'..'E') for one (ref, pred) pair (mg/dL)."""
    # Zone A: within 20% of reference, or both < 70
    if (ref <= 70 and pred <= 70) or (abs(pred - ref) <= 0.2 * ref):
        return "A"
    # Zone E: erroneous opposite treatment
    if (ref <= 70 and pred >= 180) or (ref >= 180 and pred <= 70):
        return "E"
    # Zone C: overcorrection
    if ((70 <= ref <= 290) and pred >= ref + 110) or \
       ((130 <= ref <= 180) and pred <= (7.0 / 5.0) * ref - 182):
        return "C"
    # Zone D: dangerous failure to detect
    if (ref >= 240 and (70 <= pred <= 180)) or \
       (ref <= 70 and (70 <= pred <= 180)):
        return "D"
    # Everything else is benign zone B
    return "B"


def clarke_grid(ref: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    zones = {z: 0 for z in "ABCDE"}
    for r, p in zip(ref, pred):
        zones[clarke_zone(float(r), float(p))] += 1
    n = len(ref)
    pct = {z: round(zones[z] / n * 100, 2) for z in "ABCDE"}
    pct["A_plus_B"] = round(pct["A"] + pct["B"], 2)
    return pct


def plot_clarke(ref: np.ndarray, pred: np.ndarray, out_png: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7), dpi=140)
    # scatter (subsample for legibility)
    if len(ref) > 6000:
        idx = np.random.RandomState(0).choice(len(ref), 6000, replace=False)
        rp, pp = ref[idx], pred[idx]
    else:
        rp, pp = ref, pred
    ax.scatter(rp, pp, s=6, alpha=0.25, color="#2563eb", edgecolors="none")
    lim = 400
    ax.plot([0, lim], [0, lim], "k-", lw=1)
    # +/-20% reference lines
    ax.plot([0, lim], [0, lim * 1.2], "k--", lw=0.6, alpha=0.5)
    ax.plot([0, lim], [0, lim * 0.8], "k--", lw=0.6, alpha=0.5)
    # clinical thresholds
    for v in (70, 180):
        ax.axhline(v, color="#64748b", lw=0.5, ls=":")
        ax.axvline(v, color="#64748b", lw=0.5, ls=":")
    # Zone letter labels (standard Clarke EGA positions)
    zone_labels = [
        (240, 230, "A"), (370, 260, "B"), (120, 370, "B"),
        (160, 20, "C"), (25, 125, "C"),
        (370, 120, "D"), (25, 300, "E"), (370, 25, "E"),
    ]
    for zx, zy, zl in zone_labels:
        ax.text(zx, zy, zl, fontsize=15, fontweight="bold", color="#334155",
                ha="center", va="center", alpha=0.75)

    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Reference glucose (mg/dL)", fontweight="bold")
    ax.set_ylabel("Predicted glucose (mg/dL)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close(fig)


def lead_time_analysis(
    df_test: pd.DataFrame,
    proba: np.ndarray,
    threshold: float = 0.5,
    step_minutes: int = 5,
    horizon_steps: int = 24,
) -> Dict[str, float]:
    """Among true adverse windows that the model flags, the forecast horizon is
    the warning lead time (the target is `horizon_steps` ahead of the window end).
    Reports detection rate and the effective lead time of correct early warnings.
    """
    y = df_test["target_adverse"].values.astype(int)
    flagged = (proba >= threshold).astype(int)
    true_adv = y == 1
    n_true = int(true_adv.sum())
    caught = int(((flagged == 1) & true_adv).sum())
    horizon_min = horizon_steps * step_minutes
    return {
        "n_true_adverse_windows": n_true,
        "detected": caught,
        "detection_rate_pct": round(caught / n_true * 100, 1) if n_true else 0.0,
        # The prediction horizon is fixed by design: the model forecasts the
        # adverse event `horizon_min` minutes ahead, so every true-positive flag
        # is by construction a warning that far in advance. This is the forecast
        # horizon, not an empirically-varying lead-time distribution.
        "forecast_horizon_min": horizon_min,
        "operating_threshold": threshold,
    }


def main() -> Dict:
    df = pd.read_csv(REAL_CSV)
    pts = df["patient_id"].unique().tolist()
    _, _, te_p = patient_split(pts)
    te = df[df.patient_id.isin(te_p)].reset_index(drop=True)

    X = te[REAL_FEATURES].values.astype(np.float32)
    y = te["target_glucose_2h"].values.astype(np.float32)

    reg = xgb.XGBRegressor(); reg.load_model(os.path.join(PROCESSED, "xgb_real_regressor.json"))
    clf = xgb.XGBClassifier(); clf.load_model(os.path.join(PROCESSED, "xgb_real_classifier.json"))

    pred = reg.predict(X)
    proba = clf.predict_proba(X)[:, 1]

    grid = clarke_grid(y, pred)
    out_png = os.path.join(PROCESSED, "clarke_error_grid_real.png")
    plot_clarke(y, pred, out_png,
                "Clarke Error Grid — Real-World Held-Out T2DM Patients\n(2-hour glucose forecast)")

    lead = lead_time_analysis(te, proba)

    results = {
        "dataset": "ShanghaiT2DM held-out real patients",
        "clarke_error_grid_pct": grid,
        "clinically_acceptable_A_plus_B_pct": grid["A_plus_B"],
        "lead_time_analysis": lead,
        "interpretation": (
            f"{grid['A_plus_B']}% of 2-hour forecasts fall in Clarke zones A+B "
            f"(clinically acceptable). The adverse-event classifier flags "
            f"{lead['detection_rate_pct']}% of true excursions a full "
            f"{lead['forecast_horizon_min']} minutes in advance."
        ),
    }
    with open(os.path.join(PROCESSED, "clinical_metrics_real.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("  CLINICAL SAFETY METRICS — Real held-out T2DM patients")
    print("=" * 60)
    print("Clarke Error Grid zones:")
    for z in "ABCDE":
        print(f"   Zone {z}: {grid[z]:>5.2f}%")
    print(f"   A+B (clinically acceptable): {grid['A_plus_B']}%")
    print("-" * 60)
    print(f"True adverse windows: {lead['n_true_adverse_windows']} | "
          f"detected: {lead['detected']} ({lead['detection_rate_pct']}%)")
    print(f"Forecast horizon: {lead['forecast_horizon_min']} minutes ahead")
    print(f"Error grid plot: {out_png}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    main()
