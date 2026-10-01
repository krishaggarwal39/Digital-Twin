import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd


static_feature_names = [
    "age", "sex_male", "bmi", "hba1c", "fasting_glucose", "egfr",
    "ldl", "hdl", "systolic_bp", "years_with_t2dm", "comorbidity_count",
    "med_metformin", "med_insulin_basal", "med_insulin_rapid",
    "med_sglt2i", "med_glp1ra", "insulin_total_daily_dose",
    "genetic_risk_score"
]

dynamic_feature_names = [
    # CGM (10)
    "cgm_mean_4h", "cgm_std_4h", "cgm_min_4h", "cgm_max_4h",
    "cgm_current", "cgm_cv_4h",
    "cgm_roc_15min", "cgm_roc_30min", "cgm_acceleration",
    "cgm_pct_above_180",
    # Heart Rate (4)
    "hr_mean_4h", "hr_std_4h", "hr_current", "hr_resting_estimate",
    # HRV (3)
    "hrv_rmssd_mean_4h", "hrv_rmssd_current", "hrv_trend_4h",
    # Activity (4)
    "steps_last_1h", "steps_last_4h", "active_minutes_1h",
    "minutes_since_last_activity",
    # Meal (3)
    "minutes_since_last_meal", "carbs_last_meal", "total_carbs_4h",
    # Insulin (2)
    "active_insulin_on_board", "minutes_since_last_insulin",
    # Temporal (3)
    "hour_sin", "hour_cos", "is_sleep_period"
]


def extract_static_features(profile: pd.Series) -> dict:
    """Extract 18 static features from a patient EHR profile."""
    sex_val = str(profile.get("sex", "")).strip().upper()
    sex_male = 1.0 if sex_val == "M" else 0.0

    return {
        "age": float(profile["age"]),
        "sex_male": sex_male,
        "bmi": float(profile["bmi"]),
        "hba1c": float(profile["hba1c"]),
        "fasting_glucose": float(profile["fasting_glucose"]),
        "egfr": float(profile["egfr"]),
        "ldl": float(profile["ldl"]),
        "hdl": float(profile["hdl"]),
        "systolic_bp": float(profile["systolic_bp"]),
        "years_with_t2dm": float(profile["years_with_t2dm"]),
        "comorbidity_count": float(profile["comorbidity_count"]),
        "med_metformin": float(profile["med_metformin"]),
        "med_insulin_basal": float(profile["med_insulin_basal"]),
        "med_insulin_rapid": float(profile["med_insulin_rapid"]),
        "med_sglt2i": float(profile["med_sglt2i"]),
        "med_glp1ra": float(profile["med_glp1ra"]),
        "insulin_total_daily_dose": float(profile["insulin_total_daily_dose"]),
        "genetic_risk_score": float(profile["genetic_risk_score"]),
    }


def extract_dynamic_features(window: pd.DataFrame) -> dict:
    """Extract 30 dynamic features from a 4-hour (48 readings) lookback window."""
    # CGM features
    cgm = window["cgm_glucose"].values
    cgm_mean_4h = float(np.mean(cgm))
    cgm_std_4h = float(np.std(cgm)) if len(cgm) > 1 else 0.0
    cgm_min_4h = float(np.min(cgm))
    cgm_max_4h = float(np.max(cgm))
    cgm_current = float(cgm[-1])
    cgm_cv_4h = float(cgm_std_4h / cgm_mean_4h * 100.0) if cgm_mean_4h > 0 else 0.0
    cgm_roc_15min = float((cgm[-1] - cgm[-4]) / 15.0) if len(cgm) >= 4 else 0.0  # mg/dL per minute
    cgm_roc_30min = float((cgm[-1] - cgm[-7]) / 30.0) if len(cgm) >= 7 else 0.0
    cgm_acceleration = float(cgm_roc_15min - cgm_roc_30min)
    cgm_pct_above_180 = float(np.mean(cgm > 180.0) * 100.0)

    # Heart Rate features
    hr = window["heart_rate"].values
    hr_mean_4h = float(np.mean(hr))
    hr_std_4h = float(np.std(hr)) if len(hr) > 1 else 0.0
    hr_current = float(hr[-1])
    hr_resting_estimate = float(np.percentile(hr, 10))

    # HRV features
    hrv = window["hrv_rmssd"].values
    hrv_rmssd_mean_4h = float(np.mean(hrv))
    hrv_rmssd_current = float(hrv[-1])
    # Trend: slope of linear regression over window
    if len(hrv) > 1:
        x = np.arange(len(hrv))
        hrv_trend_4h = float(np.polyfit(x, hrv, 1)[0])  # slope
    else:
        hrv_trend_4h = 0.0

    # Activity features
    steps = window["steps_5min"].values
    activity = window["activity_intensity"].values
    steps_last_1h = float(np.sum(steps[-12:]))          # Last 12 readings = 1 hour
    steps_last_4h = float(np.sum(steps))
    active_minutes_1h = float(np.sum(activity[-12:] > 0.1) * 5.0)   # Each reading = 5 min
    # Minutes since last activity > 0.1 intensity
    active_indices = np.where(activity > 0.1)[0]
    if len(active_indices) > 0:
        minutes_since_last_activity = float((len(activity) - 1 - active_indices[-1]) * 5)
    else:
        minutes_since_last_activity = 240.0  # Cap at 4 hours

    # Meal features
    meal_flags = window["meal_flag"].values
    meal_carbs_col = window["meal_carbs"].values
    meal_indices = np.where(meal_flags == 1)[0]
    if len(meal_indices) > 0:
        minutes_since_last_meal = float((len(meal_flags) - 1 - meal_indices[-1]) * 5)
        carbs_last_meal = float(meal_carbs_col[meal_indices[-1]])
    else:
        minutes_since_last_meal = 240.0  # Cap at 4 hours
        carbs_last_meal = 0.0
    total_carbs_4h = float(np.sum(meal_carbs_col))

    # Insulin features
    insulin_flags = window["insulin_flag"].values
    insulin_doses = window["insulin_dose"].values
    insulin_indices = np.where(insulin_flags == 1)[0]
    if len(insulin_indices) > 0:
        minutes_since_last_insulin = float((len(insulin_flags) - 1 - insulin_indices[-1]) * 5)
        # Active insulin on board: sum of doses with exponential decay (half-life ~60 min)
        active_insulin_on_board = 0.0
        for idx in insulin_indices:
            minutes_ago = (len(insulin_flags) - 1 - idx) * 5
            active_insulin_on_board += float(insulin_doses[idx]) * np.exp(-0.693 * minutes_ago / 60.0)
    else:
        minutes_since_last_insulin = 240.0  # Cap
        active_insulin_on_board = 0.0

    # Temporal features
    last_ts = pd.to_datetime(window["timestamp"].iloc[-1])
    hour = last_ts.hour + last_ts.minute / 60.0
    hour_sin = float(np.sin(2.0 * np.pi * hour / 24.0))
    hour_cos = float(np.cos(2.0 * np.pi * hour / 24.0))
    is_sleep_period = float(int(window["is_sleeping"].iloc[-1]))

    return {
        # CGM (10)
        "cgm_mean_4h": cgm_mean_4h,
        "cgm_std_4h": cgm_std_4h,
        "cgm_min_4h": cgm_min_4h,
        "cgm_max_4h": cgm_max_4h,
        "cgm_current": cgm_current,
        "cgm_cv_4h": cgm_cv_4h,
        "cgm_roc_15min": cgm_roc_15min,
        "cgm_roc_30min": cgm_roc_30min,
        "cgm_acceleration": cgm_acceleration,
        "cgm_pct_above_180": cgm_pct_above_180,
        # Heart Rate (4)
        "hr_mean_4h": hr_mean_4h,
        "hr_std_4h": hr_std_4h,
        "hr_current": hr_current,
        "hr_resting_estimate": hr_resting_estimate,
        # HRV (3)
        "hrv_rmssd_mean_4h": hrv_rmssd_mean_4h,
        "hrv_rmssd_current": hrv_rmssd_current,
        "hrv_trend_4h": hrv_trend_4h,
        # Activity (4)
        "steps_last_1h": steps_last_1h,
        "steps_last_4h": steps_last_4h,
        "active_minutes_1h": active_minutes_1h,
        "minutes_since_last_activity": minutes_since_last_activity,
        # Meal (3)
        "minutes_since_last_meal": minutes_since_last_meal,
        "carbs_last_meal": carbs_last_meal,
        "total_carbs_4h": total_carbs_4h,
        # Insulin (2)
        "active_insulin_on_board": active_insulin_on_board,
        "minutes_since_last_insulin": minutes_since_last_insulin,
        # Temporal (3)
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "is_sleep_period": is_sleep_period,
    }


def build_dataset(
    ehr_path: str = "data/synthea/patient_profiles.csv",
    wearable_dir: str = "data/wearable",
    output_dir: str = "data/processed"
):
    """Build train, validation, and test datasets with static, dynamic features and targets."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Read patient profiles
    ehr_df = pd.read_csv(ehr_path)
    patient_ids = ehr_df["patient_id"].tolist()

    # 2. Split patients at patient level
    train_patients = patient_ids[:210]   # P001-P210 (210 patients, 70%)
    val_patients = patient_ids[210:255]   # P211-P255 (45 patients, 15%)
    test_patients = patient_ids[255:300]  # P256-P300 (45 patients, 15%)

    splits = {
        "train": train_patients,
        "val": val_patients,
        "test": test_patients
    }

    all_features = static_feature_names + dynamic_feature_names
    target_names = ["target_glucose_2h", "target_adverse"]

    split_counts = {}

    for split_name, p_list in splits.items():
        print(f"\nBuilding {split_name} split with {len(p_list)} patients...")
        split_rows = []

        total_patients = len(p_list)
        for i, pid in enumerate(p_list, 1):
            if i % 30 == 0 or i == total_patients:
                print(f"Processing patient {i}/{total_patients} for {split_name} split")

            # Static features for patient
            profile_row = ehr_df.loc[ehr_df["patient_id"] == pid].iloc[0]
            static_dict = extract_static_features(profile_row)

            # Wearable time series
            wearable_path = os.path.join(wearable_dir, f"{pid}.csv")
            wearable_data = pd.read_csv(wearable_path)
            # Pre-parse timestamps for fast slice handling
            wearable_data["timestamp"] = pd.to_datetime(wearable_data["timestamp"])

            n_rows = len(wearable_data)
            # Slide a 48-row window from row index 48 to (len - 24), stepping by 6
            for window_end_idx in range(48, n_rows - 24, 6):
                window = wearable_data.iloc[window_end_idx - 48:window_end_idx]
                dynamic_dict = extract_dynamic_features(window)

                target_glucose_2h = float(wearable_data.iloc[window_end_idx + 24]["cgm_glucose"])
                target_adverse = 1 if (target_glucose_2h > 250.0 or target_glucose_2h < 54.0) else 0

                row_dict = {}
                row_dict.update(static_dict)
                row_dict.update(dynamic_dict)
                row_dict["target_glucose_2h"] = target_glucose_2h
                row_dict["target_adverse"] = target_adverse
                row_dict["patient_id"] = pid

                split_rows.append(row_dict)

        df_split = pd.DataFrame(split_rows)

        # Handle missing/NaN values: fill with 0.0
        df_split = df_split.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Round numeric feature values to 4 decimal places
        numeric_cols = all_features + ["target_glucose_2h"]
        for col in numeric_cols:
            df_split[col] = df_split[col].round(4).astype(np.float32)

        df_split["target_adverse"] = df_split["target_adverse"].astype(np.int32)

        # Ensure correct column ordering
        final_cols = all_features + ["target_glucose_2h", "target_adverse", "patient_id"]
        df_split = df_split[final_cols]

        # Save to CSV
        out_csv = os.path.join(output_dir, f"features_{split_name}.csv")
        df_split.to_csv(out_csv, index=False, float_format="%.4f")
        split_counts[f"{split_name}_rows"] = len(df_split)
        print(f"Saved {out_csv} ({len(df_split)} rows)")

    # Save feature_names.json
    feature_names_dict = {
        "static": static_feature_names,
        "dynamic": dynamic_feature_names,
        "all": all_features,
        "targets": target_names
    }
    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names_dict, f, indent=2)
    print(f"Saved {output_dir}/feature_names.json")

    # Save split_info.json
    split_info_dict = {
        "train_patients": train_patients,
        "val_patients": val_patients,
        "test_patients": test_patients,
        "train_rows": split_counts["train_rows"],
        "val_rows": split_counts["val_rows"],
        "test_rows": split_counts["test_rows"]
    }
    with open(os.path.join(output_dir, "split_info.json"), "w") as f:
        json.dump(split_info_dict, f, indent=2)
    print(f"Saved {output_dir}/split_info.json")


if __name__ == "__main__":
    build_dataset(
        ehr_path="data/synthea/patient_profiles.csv",
        wearable_dir="data/wearable",
        output_dir="data/processed"
    )
    # Load and print summary
    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"data/processed/features_{split}.csv")
        print(f"{split}: {len(df)} rows, {len(df.columns)} columns")
        print(f"  Adverse event rate: {df['target_adverse'].mean()*100:.1f}%")
        print(f"  Target glucose mean: {df['target_glucose_2h'].mean():.1f} mg/dL")
