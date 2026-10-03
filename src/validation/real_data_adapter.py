"""Adapter: map the real-world Shanghai T2DM dataset into the digital twin's
47-feature schema so the trained model can be validated out-of-distribution.

Source: Zhao et al., "Chinese diabetes datasets for data-driven machine learning",
Scientific Data 10, 35 (2023). ShanghaiT2DM = 100+ real Type 2 patients with
continuous glucose monitoring (15-min interval) plus a static clinical summary.

Honesty notes (surfaced in the validation report, never hidden):
  * The Shanghai dataset provides REAL CGM + REAL EHR but NO consumer-wearable
    streams (heart rate, HRV, steps, sleep). We therefore validate the
    CGM + EHR pathway of the fused model and neutralize the absent dynamic
    wearable features by setting them to the training-set median (an unbiased,
    information-free default). This is a conservative test: the model gets no
    wearable signal it was partly trained to use.
  * CGM is resampled from 15-min to the model's native 5-min grid by linear
    interpolation, matching the 48-step (4h) lookback and 24-step (2h) horizon.
  * A handful of static fields absent from Shanghai (systolic BP, polygenic risk
    score) are imputed with neutral/median values; comorbidity count is derived
    from the recorded complication/comorbidity fields.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# --- unit conversions ---
MMOL_L_TO_MG_DL_CHOL = 38.67        # cholesterol mmol/L -> mg/dL


def hba1c_mmol_mol_to_percent(ifcc: float) -> float:
    """IFCC (mmol/mol) -> NGSP (%) :  % = IFCC/10.929 + 2.15."""
    return float(ifcc) / 10.929 + 2.15


def _text(x: Any) -> str:
    return "" if pd.isna(x) else str(x).lower()


def _num(x: Any, default: float) -> Optional[float]:
    """Parse a possibly-dirty numeric cell ('/', '-', '', text) -> float or None."""
    if x is None or pd.isna(x):
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        s = str(x).strip()
        if s in {"/", "-", "", "nan", "NA"}:
            return None
        m = re.search(r"-?\d+\.?\d*", s)
        return float(m.group()) if m else None


def parse_medications(agents: str) -> Dict[str, Any]:
    """Parse the free-text 'Hypoglycemic Agents' field into medication flags.

    Returns the five medication-class flags used by the model plus an estimated
    total daily insulin dose (coarse; the dataset does not give a single TDD, so
    the presence of insulin maps to a cohort-typical dose).
    """
    t = _text(agents)

    metformin = int("metformin" in t)
    # Insulin brand/analog names present in the Shanghai free text
    insulin_terms = [
        "insulin", "novolin", "humulin", "gansulin", "aspart", "lispro",
        "glargine", "detemir", "degludec", "70/30", "30r", "50/50", "70r", "40r",
    ]
    has_insulin = int(any(k in t for k in insulin_terms))
    # Rapid/prandial analogs vs. basal/premix: aspart/lispro/R -> rapid component
    rapid_terms = ["aspart", "lispro", "novolin r", "humulin r", "insulin r"]
    med_insulin_rapid = int(any(k in t for k in rapid_terms))
    # Any insulin regimen implies at least basal coverage.
    med_insulin_basal = int(has_insulin)

    sglt2 = int(any(k in t for k in ["flozin", "dapagliflozin", "empagliflozin", "canagliflozin"]))
    glp1 = int(any(k in t for k in ["glutide", "liraglutide", "dulaglutide", "semaglutide", "exenatide"]))

    # Coarse total daily insulin dose: 0 if none, else a cohort-typical value.
    insulin_tdd = 30.0 if has_insulin else 0.0

    return {
        "med_metformin": float(metformin),
        "med_insulin_basal": float(med_insulin_basal),
        "med_insulin_rapid": float(med_insulin_rapid),
        "med_sglt2i": float(sglt2),
        "med_glp1ra": float(glp1),
        "insulin_total_daily_dose": float(insulin_tdd),
    }


def _count_comorbidities(row: pd.Series) -> int:
    """Derive a 0-5 comorbidity count from the complication/comorbidity fields."""
    fields = [
        "Acute Diabetic Complications",
        "Diabetic Macrovascular  Complications",
        "Diabetic Microvascular Complications",
        "Comorbidities",
    ]
    n = 0
    for f in fields:
        if f in row and not pd.isna(row[f]):
            txt = str(row[f]).strip().lower()
            if txt and txt not in {"/", "none", "no", "nan", "-"}:
                # count comma/semicolon separated items, cap contribution
                n += min(2, len([p for p in re.split(r"[,;、]", txt) if p.strip()]))
    return int(np.clip(n, 0, 5))


def build_static_features(summary_row: pd.Series) -> Dict[str, float]:
    """Map one Shanghai summary row to the 18 static EHR features."""
    g = _num(summary_row.get("Gender (Female=1, Male=2)", 2), 2.0)
    sex_male = 1.0 if (g is not None and g == 2) else 0.0

    ifcc = _num(summary_row.get("HbA1c (mmol/mol)", None), None)
    hba1c = hba1c_mmol_mol_to_percent(ifcc) if ifcc is not None else 7.5

    ldl_mmol = _num(summary_row.get("Low-Density Lipoprotein Cholesterol (mmol/L)", None), None)
    hdl_mmol = _num(summary_row.get("High-Density Lipoprotein Cholesterol (mmol/L)", None), None)
    ldl = ldl_mmol * MMOL_L_TO_MG_DL_CHOL if ldl_mmol is not None else 110.0
    hdl = hdl_mmol * MMOL_L_TO_MG_DL_CHOL if hdl_mmol is not None else 45.0

    egfr_raw = None
    for c in summary_row.index:
        if "Glomerular" in str(c):
            egfr_raw = _num(summary_row[c], None)
            break
    egfr = egfr_raw if egfr_raw is not None else 90.0

    fpg = _num(summary_row.get("Fasting Plasma Glucose (mg/dl)", None), None)
    fasting_glucose = fpg if fpg is not None else 130.0

    dur = _num(summary_row.get("Duration of diabetes (years)", None), None)
    years_t2dm = dur if dur is not None else 5.0

    age = _num(summary_row.get("Age (years)", 55), 55.0) or 55.0
    bmi = _num(summary_row.get("BMI (kg/m2)", 25), 25.0) or 25.0

    meds = parse_medications(summary_row.get("Hypoglycemic Agents", ""))

    feats = {
        "age": float(np.clip(age, 18, 95)),
        "sex_male": sex_male,
        "bmi": float(np.clip(bmi, 15, 55)),
        "hba1c": float(np.clip(hba1c, 4.0, 15.0)),
        "fasting_glucose": float(np.clip(fasting_glucose, 60, 350)),
        "egfr": float(np.clip(egfr, 10, 150)),
        "ldl": float(np.clip(ldl, 30, 300)),
        "hdl": float(np.clip(hdl, 15, 120)),
        "systolic_bp": 138.0,                 # imputed (absent in dataset)
        "years_with_t2dm": float(np.clip(years_t2dm, 0, 40)),
        "comorbidity_count": float(_count_comorbidities(summary_row)),
        **meds,
        "genetic_risk_score": 0.5,            # imputed neutral (no genetics in dataset)
    }
    return feats


def load_cgm_series(patient_file: str) -> pd.DataFrame:
    """Load one patient's CGM file, resampled to a regular 5-minute grid."""
    df = pd.read_excel(patient_file)
    # Normalize the key columns (names carry odd spacing in the source)
    date_col = next((c for c in df.columns if str(c).strip().lower() == "date"), None)
    cgm_col = next((c for c in df.columns if "cgm" in str(c).lower()), None)
    if date_col is None or cgm_col is None:
        raise ValueError(f"{patient_file}: missing Date/CGM columns")

    s = df[[date_col, cgm_col]].copy()
    s.columns = ["timestamp", "cgm_glucose"]
    s["timestamp"] = pd.to_datetime(s["timestamp"], errors="coerce")
    s = s.dropna(subset=["timestamp", "cgm_glucose"]).sort_values("timestamp")
    s = s.drop_duplicates(subset="timestamp").set_index("timestamp")

    # Resample onto a continuous 5-min grid. Interpolate only across SMALL gaps
    # (<=30 min); larger gaps remain NaN so that build_windows never forms a
    # window or 2-hour target that silently spans a real data gap (which would
    # make the "2 hours ahead" target untrue in wall-clock time).
    s = s.resample("5min").mean().interpolate(method="linear", limit=6, limit_area="inside")
    s = s.reset_index()  # keep NaNs; build_windows enforces contiguity
    return s


def build_windows(
    cgm: pd.DataFrame,
    static: Dict[str, float],
    dynamic_medians: Dict[str, float],
    feature_names: Dict[str, List[str]],
    lookback: int = 48,
    horizon: int = 24,
    step: int = 6,
) -> pd.DataFrame:
    """Build 47-feature windows + 2h-ahead targets from one patient's CGM series.

    Dynamic CGM-derived features are computed from the real CGM window. All other
    dynamic (wearable) features are set to the training-set median (neutralized),
    and static features are the patient's real EHR values.
    """
    static_names = feature_names["static"]
    dynamic_names = feature_names["dynamic"]
    all_names = feature_names["all"]

    g = cgm["cgm_glucose"].values
    ts = cgm["timestamp"].values
    n = len(g)
    rows: List[Dict[str, float]] = []

    for end in range(lookback, n - horizon, step):
        w = g[end - lookback:end]
        target = float(g[end + horizon])

        # Contiguity guard: skip any window or 2-hour target that spans a data
        # gap (NaN introduced by resampling across a real recording gap). This
        # keeps every window continuous and every target a true 2h-ahead value.
        if np.isnan(w).any() or np.isnan(target):
            continue

        # Real CGM-derived dynamic features (match feature_pipeline exactly)
        cgm_mean = float(np.mean(w))
        cgm_std = float(np.std(w))
        last_ts = pd.Timestamp(ts[end - 1])
        hour = last_ts.hour + last_ts.minute / 60.0

        feat = dict(dynamic_medians)  # start from neutral medians for ALL dynamic
        feat.update({
            "cgm_mean_4h": cgm_mean,
            "cgm_std_4h": cgm_std,
            "cgm_min_4h": float(np.min(w)),
            "cgm_max_4h": float(np.max(w)),
            "cgm_current": float(w[-1]),
            "cgm_cv_4h": float(cgm_std / cgm_mean * 100.0) if cgm_mean > 0 else 0.0,
            "cgm_roc_15min": float((w[-1] - w[-4]) / 15.0) if lookback >= 4 else 0.0,
            "cgm_roc_30min": float((w[-1] - w[-7]) / 30.0) if lookback >= 7 else 0.0,
            "cgm_acceleration": 0.0,
            "cgm_pct_above_180": float(np.mean(w > 180.0) * 100.0),
            # circadian features are real (derived from the real timestamp)
            "hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
            "is_sleep_period": 1.0 if (hour >= 23 or hour < 6.5) else 0.0,
        })
        feat["cgm_acceleration"] = feat["cgm_roc_15min"] - feat["cgm_roc_30min"]

        row = {name: static[name] for name in static_names}
        row.update({name: feat[name] for name in dynamic_names})
        row["target_glucose_2h"] = target
        row["target_adverse"] = 1 if (target > 250.0 or target < 54.0) else 0
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=all_names + ["target_glucose_2h", "target_adverse"])
    df = pd.DataFrame(rows)
    # Sanitize any inf/NaN the same way the training pipeline does, so the real
    # feature table is guaranteed clean for inputs dirtier than this cohort.
    df[all_names] = df[all_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df[all_names + ["target_glucose_2h", "target_adverse"]]


def build_real_dataset(
    shanghai_dir: str = "data/real_shanghai",
    processed_dir: str = "data/processed",
    cohort: str = "T2DM",
) -> pd.DataFrame:
    """Build the full real-world validation feature table across all patients."""
    feature_names = json.load(open(os.path.join(processed_dir, "feature_names.json")))

    # Training medians for neutralizing absent wearable features
    train = pd.read_csv(os.path.join(processed_dir, "features_train.csv"))
    dynamic_medians = {c: float(train[c].median()) for c in feature_names["dynamic"]}

    summary = pd.read_excel(os.path.join(shanghai_dir, f"Shanghai_{cohort}_Summary.xlsx"))
    patient_dir = os.path.join(shanghai_dir, f"Shanghai_{cohort}")
    files = sorted(glob.glob(os.path.join(patient_dir, "*.xls")) +
                   glob.glob(os.path.join(patient_dir, "*.xlsx")))

    all_parts: List[pd.DataFrame] = []
    n_ok = 0
    summary_ids = summary["Patient Number"].astype(str)
    for pf in files:
        # 'Patient Number' is the full file stem, e.g. 2000_0_20201230
        stem = os.path.splitext(os.path.basename(pf))[0]
        pid_num = stem.split("_")[0]  # short id for labelling
        srow = summary[summary_ids == stem]
        if len(srow) == 0:
            continue
        try:
            static = build_static_features(srow.iloc[0])
            cgm = load_cgm_series(pf)
            if len(cgm) < 48 + 24 + 6:
                continue
            part = build_windows(cgm, static, dynamic_medians, feature_names)
            if len(part) == 0:
                continue
            part["patient_id"] = f"SH_{cohort}_{pid_num}"
            all_parts.append(part)
            n_ok += 1
        except Exception as e:  # noqa: BLE001 - skip unreadable patient, keep going
            print(f"  skip {os.path.basename(pf)}: {e}")

    out = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    print(f"Built real-world dataset: {n_ok} patients, {len(out)} windows")
    return out


if __name__ == "__main__":
    df = build_real_dataset()
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/features_real_shanghai_t2dm.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} rows, {len(df.columns)} cols)")
    if len(df):
        adv = df["target_adverse"].mean() * 100
        print(f"Target glucose mean: {df['target_glucose_2h'].mean():.1f} mg/dL | "
              f"adverse rate: {adv:.2f}% | patients: {df['patient_id'].nunique()}")
