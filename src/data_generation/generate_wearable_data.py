"""Synthetic Wearable and IoT Continuous Sensor Data Generator for T2DM Cohort.

This module generates 14 days of realistic multi-signal wearable/IoT time-series
data (at 5-minute sampling intervals) for each patient in the synthetic EHR cohort:
- Continuous Glucose Monitoring (CGM) via the Bergman Minimal Model
- Continuous Photoplethysmography (PPG) Heart Rate (HR)
- Heart Rate Variability (HRV RMSSD)
- Tri-axial Accelerometer Physical Activity (5-minute step counts, activity intensity)
- Sleep Architecture (is_sleeping flag, sleep stages: awake, light, deep, rem)
- Behavioral & Treatment Flags (meal intake carbs/GI, exercise sessions, insulin doses)
"""

from datetime import datetime, timedelta
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

# Enable relative project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from src.data_generation.glucose_dynamics_model import PatientParams, simulate_glucose


def generate_patient_wearable_data(
    profile: Dict[str, Any],
    patient_index: int,
    n_days: int = 14,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate 14 days of 5-minute wearable multi-signal time-series for a single patient.

    Parameters
    ----------
    profile : dict
        EHR clinical profile containing patient demographic, metabolic, and medication fields.
    patient_index : int
        Index of the patient in the cohort for reproducible PRNG seeding.
    n_days : int, default=14
        Number of monitoring days to simulate.
    seed : int, default=42
        Base seed for random number generation.

    Returns
    -------
    pd.DataFrame
        DataFrame of 4032 rows (14 days * 288 5-min intervals) with 14 exact columns.
    """
    # Deterministic yet unique seed per patient
    np.random.seed(seed + patient_index)

    patient_id: str = str(profile.get("patient_id", f"P{patient_index + 1:03d}"))
    age: float = float(profile.get("age", 55.0))
    bmi: float = float(profile.get("bmi", 30.0))
    med_basal: int = int(profile.get("med_insulin_basal", 0))
    med_rapid: int = int(profile.get("med_insulin_rapid", 0))
    insulin_tdd: float = float(profile.get("insulin_total_daily_dose", 0.0))

    # Initialize ODE model parameters from patient EHR
    params: PatientParams = PatientParams.from_ehr_profile(profile)

    n_intervals: int = n_days * 288  # 14 * 288 = 4032 intervals of 5 minutes
    total_duration_minutes: float = float(n_days * 1440)

    # Activity level scaled inversely by excess adiposity
    activity_level: float = max(0.2, 1.0 - 0.03 * (bmi - 22.0))

    # Sleep architecture proportions (aging reduces slow wave and REM sleep)
    deep_pct: float = max(0.10, 0.25 - 0.003 * age)
    rem_pct: float = max(0.15, 0.25 - 0.002 * age)
    awake_pct: float = 0.03 + 0.01 * max(0.0, bmi - 28.0) / 10.0
    light_pct: float = max(0.10, 1.0 - deep_pct - rem_pct - awake_pct)

    meals: List[Dict[str, Any]] = []
    exercises: List[Dict[str, Any]] = []
    insulin_events: List[Dict[str, Any]] = []
    daily_schedules: List[Dict[str, Any]] = []

    # 1. Generate Daily Schedules across 14 Days
    for day in range(n_days):
        day_offset_min: float = float(day * 1440)

        # Sleep times: bedtime ~10:30 PM (22.5h), wake ~6:30 AM (6.5h)
        bedtime_hour: float = float(np.clip(np.random.normal(22.5, 0.5), 21.0, 24.0))
        wake_hour: float = float(np.clip(np.random.normal(6.5, 0.5), 5.0, 9.0))

        # Main meals (Breakfast, Lunch, Dinner)
        d_meals: List[Dict[str, Any]] = [
            {
                "name": "breakfast",
                "hour": float(np.clip(np.random.normal(7.5, 0.5), 6.0, 9.5)),
                "carbs": float(np.clip(np.random.normal(45.0, 10.0), 10.0, 100.0)),
                "gi": float(np.clip(np.random.uniform(0.5, 0.8), 0.3, 0.95)),
            },
            {
                "name": "lunch",
                "hour": float(np.clip(np.random.normal(12.5, 0.5), 11.5, 14.5)),
                "carbs": float(np.clip(np.random.normal(60.0, 15.0), 10.0, 100.0)),
                "gi": float(np.clip(np.random.uniform(0.4, 0.7), 0.3, 0.95)),
            },
            {
                "name": "dinner",
                "hour": float(np.clip(np.random.normal(19.5, 0.75), 18.0, 21.5)),
                "carbs": float(np.clip(np.random.normal(65.0, 15.0), 10.0, 100.0)),
                "gi": float(np.clip(np.random.uniform(0.4, 0.9), 0.3, 0.95)),
            },
        ]

        # 40% probability of an afternoon snack
        if np.random.random() < 0.40:
            d_meals.append({
                "name": "snack",
                "hour": float(np.clip(np.random.normal(15.5, 1.0), 14.0, 17.5)),
                "carbs": float(np.clip(np.random.normal(20.0, 8.0), 10.0, 100.0)),
                "gi": float(np.clip(np.random.uniform(0.6, 0.9), 0.3, 0.95)),
            })

        # Physical activity sessions
        d_exercises: List[Dict[str, Any]] = []
        if np.random.random() < 0.30 * activity_level:
            d_exercises.append({
                "hour": float(np.clip(np.random.normal(6.5, 0.25), wake_hour, 8.0)),
                "duration": float(np.clip(np.random.normal(25.0, 4.0), 10.0, 45.0)),
                "intensity": float(np.clip(np.random.normal(0.40, 0.05), 0.20, 0.70)),
            })
        if np.random.random() < 0.20 * activity_level:
            d_exercises.append({
                "hour": float(np.clip(np.random.normal(13.0, 0.25), 12.5, 14.0)),
                "duration": float(np.clip(np.random.normal(15.0, 3.0), 10.0, 30.0)),
                "intensity": float(np.clip(np.random.normal(0.30, 0.05), 0.20, 0.50)),
            })
        if np.random.random() < 0.25 * activity_level:
            d_exercises.append({
                "hour": float(np.clip(np.random.normal(17.5, 0.30), 16.5, 18.5)),
                "duration": float(np.clip(np.random.normal(30.0, 5.0), 15.0, 60.0)),
                "intensity": float(np.clip(np.random.normal(0.60, 0.08), 0.30, 0.90)),
            })
        if np.random.random() < 0.35 * activity_level:
            d_exercises.append({
                "hour": float(np.clip(np.random.normal(19.5, 0.30), 19.0, 21.0)),
                "duration": float(np.clip(np.random.normal(20.0, 3.0), 10.0, 40.0)),
                "intensity": float(np.clip(np.random.normal(0.35, 0.05), 0.20, 0.60)),
            })

        # Pharmacological insulin dosing
        d_insulin: List[Dict[str, Any]] = []
        if med_basal and insulin_tdd > 0.0:
            d_insulin.append({
                "hour": bedtime_hour - 0.5,
                "dose": insulin_tdd * 0.60,
                "type": "basal",
            })
        if med_rapid and insulin_tdd > 0.0:
            for main_meal in d_meals[:3]:
                d_insulin.append({
                    "hour": main_meal["hour"] - 0.10,
                    "dose": insulin_tdd * 0.13,
                    "type": "rapid",
                })

        # Append to global 14-day lists with absolute timestamps in minutes
        for m in d_meals:
            meals.append({
                "time": day_offset_min + m["hour"] * 60.0,
                "carbs": m["carbs"],
                "gi_index": m["gi"],
            })
        for ex in d_exercises:
            exercises.append({
                "start": day_offset_min + ex["hour"] * 60.0,
                "duration": ex["duration"],
                "intensity": ex["intensity"],
            })
        for ins in d_insulin:
            insulin_events.append({
                "time": day_offset_min + ins["hour"] * 60.0,
                "dose": ins["dose"],
                "type": ins["type"],
            })

        daily_schedules.append({
            "bedtime_hour": bedtime_hour,
            "wake_hour": wake_hour,
            "meals": d_meals,
            "exercises": d_exercises,
            "insulin": d_insulin,
        })

    # 2. Continuous 14-Day Bergman Glucose Dynamics ODE Simulation
    cgm_res = simulate_glucose(
        params=params,
        meals=meals,
        exercises=exercises,
        insulin_events=insulin_events,
        duration_minutes=total_duration_minutes,
        dt=5.0,
    )
    # Extract exact 4032 points (288 * 14 days)
    glucose_arr: np.ndarray = np.round(cgm_res["glucose"][:n_intervals], 1)

    # 3. Construct 5-minute Multi-Sensor Channels
    start_dt: datetime = datetime(2026, 1, 1, 0, 0, 0)
    timestamps: List[str] = [
        (start_dt + timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(n_intervals)
    ]

    resting_hr: float = 62.0 + 0.40 * max(0.0, bmi - 22.0) + 0.15 * max(0.0, age - 40.0)

    heart_rate: np.ndarray = np.zeros(n_intervals, dtype=float)
    hrv_rmssd: np.ndarray = np.zeros(n_intervals, dtype=float)
    steps_5min: np.ndarray = np.zeros(n_intervals, dtype=int)
    activity_intensity: np.ndarray = np.zeros(n_intervals, dtype=float)
    is_sleeping: np.ndarray = np.zeros(n_intervals, dtype=int)
    sleep_stage: List[str] = []

    meal_flag: np.ndarray = np.zeros(n_intervals, dtype=int)
    meal_carbs: np.ndarray = np.zeros(n_intervals, dtype=float)
    meal_gi: np.ndarray = np.zeros(n_intervals, dtype=float)

    exercise_flag: np.ndarray = np.zeros(n_intervals, dtype=int)
    insulin_flag: np.ndarray = np.zeros(n_intervals, dtype=int)
    insulin_dose: np.ndarray = np.zeros(n_intervals, dtype=float)

    # Populate discrete meal flags
    for m in meals:
        idx: int = int(m["time"] // 5)
        if idx < n_intervals:
            meal_flag[idx] = 1
            meal_carbs[idx] = round(float(m["carbs"]), 1)
            meal_gi[idx] = round(float(m["gi_index"]), 2)

    # Populate discrete insulin administration flags
    for ins in insulin_events:
        idx = int(ins["time"] // 5)
        if idx < n_intervals:
            insulin_flag[idx] = 1
            insulin_dose[idx] = round(float(ins["dose"]), 1)

    # Populate active exercise session intervals
    for ex in exercises:
        start_idx: int = int(ex["start"] // 5)
        end_idx: int = int((ex["start"] + ex["duration"]) // 5)
        ex_inten: float = round(float(ex["intensity"]), 2)
        for idx in range(start_idx, min(end_idx + 1, n_intervals)):
            exercise_flag[idx] = 1
            activity_intensity[idx] = max(activity_intensity[idx], ex_inten)

    # 4. Synthesize Behavioral & Physiological Signals per 5-minute Epoch
    last_exercise_end_min: float = -999.0
    last_exercise_intensity: float = 0.0

    for i in range(n_intervals):
        t_min: float = float(i * 5)
        day_idx: int = int(t_min // 1440)
        hour: float = (t_min % 1440.0) / 60.0

        sched = daily_schedules[day_idx]
        bed_hr: float = sched["bedtime_hour"]
        wake_hr: float = sched["wake_hour"]

        # Sleep state determination
        sleeping: bool = (hour < wake_hr) or (hour >= bed_hr)
        is_sleeping[i] = 1 if sleeping else 0

        if sleeping:
            # Calculate elapsed time in sleep episode
            if hour >= bed_hr:
                sleep_elapsed = (hour - bed_hr) * 60.0
            else:
                prev_bed = (
                    daily_schedules[day_idx - 1]["bedtime_hour"]
                    if day_idx > 0
                    else 22.5
                )
                sleep_elapsed = (24.0 - prev_bed + hour) * 60.0

            # 90-minute ultradian sleep cycle: [light -> deep -> light -> rem]
            cycle_pos = (sleep_elapsed % 90.0) / 90.0
            r_stage = np.random.random()

            if r_stage < awake_pct:
                stage = "awake"
            elif cycle_pos < (light_pct / 2.0):
                stage = "light"
            elif cycle_pos < (light_pct / 2.0 + deep_pct):
                stage = "deep"
            elif cycle_pos < (light_pct + deep_pct):
                stage = "light"
            else:
                stage = "rem"

            sleep_stage.append(stage)
            activity_intensity[i] = 0.0
            steps_5min[i] = 0

        else:
            sleep_stage.append("awake")
            if exercise_flag[i] == 1:
                # Active structured exercise step count
                steps_val = int(
                    activity_intensity[i] * 500.0 + np.random.normal(0, 30)
                )
                steps_5min[i] = max(0, steps_val)
                last_exercise_end_min = t_min
                last_exercise_intensity = activity_intensity[i]
            else:
                # Random light non-exercise activity (10% of awake non-exercise epochs)
                if np.random.random() < 0.10:
                    steps_5min[i] = int(np.random.randint(20, 100))
                    activity_intensity[i] = round(
                        float(np.random.uniform(0.10, 0.25)), 2
                    )
                else:
                    steps_5min[i] = 0
                    activity_intensity[i] = 0.0

        # Circadian heart rate modulation (lower at night)
        circadian_hr: float = -8.0 * np.sin(2.0 * np.pi * hour / 24.0)

        # Heart rate response to activity and exercise recovery
        if exercise_flag[i] == 1:
            activity_hr = 30.0 * activity_intensity[i]
        else:
            dt_post = t_min - last_exercise_end_min
            if 0.0 < dt_post <= 30.0:
                activity_hr = 30.0 * last_exercise_intensity * np.exp(-0.10 * dt_post)
            elif activity_intensity[i] > 0.0:
                activity_hr = 15.0 * activity_intensity[i]
            else:
                activity_hr = 0.0

        # Parasympathetic nocturnal reduction
        sleep_hr_adj: float = -12.0 if is_sleeping[i] == 1 else 0.0

        # Total Heart Rate
        hr_val: float = (
            resting_hr
            + circadian_hr
            + activity_hr
            + sleep_hr_adj
            + np.random.normal(0.0, 2.5)
        )
        hr_val = float(np.clip(np.round(hr_val, 1), 45.0, 170.0))
        heart_rate[i] = hr_val

        # Heart Rate Variability (RMSSD in ms): Vagal tone enhanced during deep sleep
        base_hrv: float = max(15.0, 75.0 - 0.50 * max(0.0, age - 30.0) - 0.25 * hr_val)
        hrv_val: float = (
            base_hrv
            + (12.0 if sleep_stage[i] == "deep" else 0.0)
            + np.random.normal(0.0, 4.0)
        )
        hrv_val = float(np.clip(np.round(hrv_val, 1), 8.0, 120.0))
        hrv_rmssd[i] = hrv_val

    # Construct complete wearable DataFrame matching exact schema
    df = pd.DataFrame({
        "timestamp": timestamps,
        "cgm_glucose": glucose_arr,
        "heart_rate": heart_rate,
        "hrv_rmssd": hrv_rmssd,
        "steps_5min": steps_5min,
        "activity_intensity": activity_intensity,
        "is_sleeping": is_sleeping,
        "sleep_stage": sleep_stage,
        "meal_flag": meal_flag,
        "meal_carbs": meal_carbs,
        "meal_gi": meal_gi,
        "exercise_flag": exercise_flag,
        "insulin_flag": insulin_flag,
        "insulin_dose": insulin_dose,
    })

    return df


def generate_all_patients(
    ehr_path: str = "data/synthea/patient_profiles.csv",
    output_dir: str = "data/wearable",
    n_days: int = 14,
    seed: int = 42,
) -> None:
    """Generate and save 14 days of wearable time-series data for all cohort patients.

    Parameters
    ----------
    ehr_path : str, default="data/synthea/patient_profiles.csv"
        Path to synthetic EHR patient profiles CSV.
    output_dir : str, default="data/wearable"
        Directory to store patient wearable CSV files.
    n_days : int, default=14
        Number of monitoring days per patient.
    seed : int, default=42
        Base seed for reproducibility.
    """
    if not os.path.exists(ehr_path):
        raise FileNotFoundError(f"EHR profile dataset not found at: {ehr_path}")

    os.makedirs(output_dir, exist_ok=True)

    ehr_df: pd.DataFrame = pd.read_csv(ehr_path)
    n_patients: int = len(ehr_df)

    print("=" * 65)
    print("       STARTING FULL COHORT WEARABLE DATA GENERATION         ")
    print("=" * 65)
    print(f"Total Cohort Size : {n_patients} patients")
    print(f"Monitoring Window : {n_days} days ({n_days * 288} 5-min epochs per patient)")
    print(f"Output Directory  : {output_dir}")
    print("-" * 65)

    for i in range(n_patients):
        profile_row = ehr_df.iloc[i].to_dict()
        patient_id = str(profile_row.get("patient_id", f"P{i + 1:03d}"))

        patient_df = generate_patient_wearable_data(
            profile=profile_row,
            patient_index=i,
            n_days=n_days,
            seed=seed,
        )

        out_csv_path = os.path.join(output_dir, f"{patient_id}.csv")
        patient_df.to_csv(out_csv_path, index=False)

        # Print progress every 50 patients and at completion
        if (i + 1) % 50 == 0 or (i + 1) == n_patients:
            print(f"Generated {i + 1}/{n_patients} patients")

    print("=" * 65)
    print(f"Successfully generated wearable data for all {n_patients} patients.")
    print("=" * 65)


if __name__ == "__main__":
    generate_all_patients(
        ehr_path="data/synthea/patient_profiles.csv",
        output_dir="data/wearable",
        n_days=14,
        seed=42,
    )

    # Verification: count files and inspect sample patient
    output_directory = "data/wearable"
    generated_files = [f for f in os.listdir(output_directory) if f.endswith(".csv")]
    print(f"Total Wearable CSV Files Generated: {len(generated_files)}")

    # Load and display sample patient file
    sample_file = os.path.join(output_directory, "P001.csv")
    if os.path.exists(sample_file):
        sample_df = pd.read_csv(sample_file)
        print(f"\nInspection of {sample_file}:")
        print(f"DataFrame Shape: {sample_df.shape}")
        print("\nFirst 5 Rows:")
        print(sample_df.head(5))
