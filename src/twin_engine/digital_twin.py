import copy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.data_generation.glucose_dynamics_model import PatientParams, simulate_glucose


@dataclass
class Intervention:
    name: str                    # Human-readable name, e.g. "20-minute brisk walk"
    intervention_type: str       # "exercise", "meal", or "insulin"
    # Exercise params (only used if type == "exercise")
    duration: float = 20.0       # minutes
    intensity: float = 0.5       # 0-1
    # Meal params (only used if type == "meal")
    carbs: float = 0.0           # grams
    gi_index: float = 0.5        # 0-1
    # Insulin params (only used if type == "insulin")
    dose: float = 0.0            # Units
    insulin_type: str = "rapid"  # "basal" or "rapid"


@dataclass
class PredictionResult:
    predicted_glucose_2h: float        # mg/dL
    adverse_event_probability: float   # 0-1
    risk_level: str                    # "low", "moderate", "high", "critical"
    shap_top_features: list            # List of dicts: [{"feature": str, "value": float, "contribution": float, "stream": str}]


@dataclass
class SimulationResult:
    name: str                          # Scenario name
    trajectory: list                   # List of {"time_offset_min": float, "glucose": float}
    peak_glucose: float                # mg/dL
    min_glucose: float                 # mg/dL
    time_to_peak_min: float            # minutes from now
    time_in_range_pct: float           # % time glucose in 70-180 mg/dL
    enters_danger_zone: bool           # True if glucose > 250 or < 54
    risk_level: str                    # "low", "moderate", "high", "critical"


class DigitalTwin:
    """A patient-specific digital twin that maintains state, predicts, and simulates."""

    def __init__(
        self,
        patient_profile: dict,
        model_path: str = "data/processed/xgb_fused_regressor.json",
        feature_names_path: str = "data/processed/feature_names.json",
        classifier_path: str = "data/processed/xgb_fused_classifier.json",
    ):
        """Initialize the twin from a patient's EHR profile."""
        self.profile = patient_profile
        self.params = PatientParams.from_ehr_profile(patient_profile)

        # Load the 2-hour glucose regressor
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)

        # Load the dedicated adverse-event classifier (trained with class weighting).
        # The twin reports the classifier's calibrated probability rather than a
        # hand-crafted heuristic, so the dashboard reflects the real model.
        self.classifier: Optional[xgb.XGBClassifier] = None
        if classifier_path and os.path.exists(classifier_path):
            clf = xgb.XGBClassifier()
            clf.load_model(classifier_path)
            self.classifier = clf

        # Create SHAP TreeExplainer on the regressor
        self.explainer = shap.TreeExplainer(self.model)

        # Load feature names
        with open(feature_names_path, "r") as f:
            self.feature_names = json.load(f)

        # History and prediction cache
        self.history = pd.DataFrame()
        self.last_prediction: Optional[PredictionResult] = None

        # Extract 18 static features
        sex_val = str(patient_profile.get("sex", "")).strip().upper()
        self.static_features = np.array([
            float(patient_profile["age"]),
            1.0 if sex_val == "M" else 0.0,
            float(patient_profile["bmi"]),
            float(patient_profile["hba1c"]),
            float(patient_profile["fasting_glucose"]),
            float(patient_profile["egfr"]),
            float(patient_profile["ldl"]),
            float(patient_profile["hdl"]),
            float(patient_profile["systolic_bp"]),
            float(patient_profile["years_with_t2dm"]),
            float(patient_profile["comorbidity_count"]),
            float(patient_profile["med_metformin"]),
            float(patient_profile["med_insulin_basal"]),
            float(patient_profile["med_insulin_rapid"]),
            float(patient_profile["med_sglt2i"]),
            float(patient_profile["med_glp1ra"]),
            float(patient_profile["insulin_total_daily_dose"]),
            float(patient_profile["genetic_risk_score"]),
        ], dtype=np.float32)

    def load_history(self, wearable_df: pd.DataFrame):
        """Store the last 288 rows (last 24 hours) of wearable readings."""
        if len(wearable_df) < 48:
            raise ValueError(f"Wearable data must have at least 48 rows, got {len(wearable_df)}")
        self.history = wearable_df.tail(288).copy().reset_index(drop=True)
        # Ensure timestamp is parsed
        self.history["timestamp"] = pd.to_datetime(self.history["timestamp"])

    def _extract_dynamic_features(self) -> np.ndarray:
        """Extract dynamic features from the last 48 rows of self.history (4-hour lookback)."""
        if len(self.history) < 48:
            raise ValueError(f"History must have at least 48 rows to extract dynamic features, got {len(self.history)}")

        window = self.history.tail(48)
        cgm = window["cgm_glucose"].values
        hr = window["heart_rate"].values
        hrv = window["hrv_rmssd"].values
        steps = window["steps_5min"].values
        activity = window["activity_intensity"].values
        meal_flags = window["meal_flag"].values
        meal_carbs_vals = window["meal_carbs"].values
        insulin_flags = window["insulin_flag"].values
        insulin_doses = window["insulin_dose"].values

        # CGM features (10)
        cgm_mean_4h = float(np.mean(cgm))
        cgm_std_4h = float(np.std(cgm)) if len(cgm) > 1 else 0.0
        cgm_min_4h = float(np.min(cgm))
        cgm_max_4h = float(np.max(cgm))
        cgm_current = float(cgm[-1])
        cgm_cv_4h = float((cgm_std_4h / cgm_mean_4h * 100.0) if cgm_mean_4h > 0 else 0.0)
        cgm_roc_15min = float((cgm[-1] - cgm[-4]) / 15.0 if len(cgm) >= 4 else 0.0)
        cgm_roc_30min = float((cgm[-1] - cgm[-7]) / 30.0 if len(cgm) >= 7 else 0.0)
        cgm_acceleration = float(cgm_roc_15min - cgm_roc_30min)
        cgm_pct_above_180 = float(np.mean(cgm > 180.0) * 100.0)

        # HR features (4)
        hr_mean_4h = float(np.mean(hr))
        hr_std_4h = float(np.std(hr)) if len(hr) > 1 else 0.0
        hr_current = float(hr[-1])
        hr_resting_estimate = float(np.percentile(hr, 10))

        # HRV features (3)
        hrv_rmssd_mean_4h = float(np.mean(hrv))
        hrv_rmssd_current = float(hrv[-1])
        if len(hrv) > 1:
            x_vals = np.arange(len(hrv))
            hrv_trend_4h = float(np.polyfit(x_vals, hrv, 1)[0])
        else:
            hrv_trend_4h = 0.0

        # Activity features (4)
        steps_last_1h = float(np.sum(steps[-12:]))
        steps_last_4h = float(np.sum(steps))
        active_minutes_1h = float(np.sum(activity[-12:] > 0.1) * 5.0)
        active_idx = np.where(activity > 0.1)[0]
        minutes_since_last_activity = float((len(activity) - 1 - active_idx[-1]) * 5) if len(active_idx) > 0 else 240.0

        # Meal features (3)
        meal_idx = np.where(meal_flags == 1)[0]
        minutes_since_last_meal = float((len(meal_flags) - 1 - meal_idx[-1]) * 5) if len(meal_idx) > 0 else 240.0
        carbs_last_meal = float(meal_carbs_vals[meal_idx[-1]]) if len(meal_idx) > 0 else 0.0
        total_carbs_4h = float(np.sum(meal_carbs_vals))

        # Insulin features (2)
        ins_idx = np.where(insulin_flags == 1)[0]
        if len(ins_idx) > 0:
            minutes_since_last_insulin = float((len(insulin_flags) - 1 - ins_idx[-1]) * 5)
            active_insulin_on_board = 0.0
            for idx in ins_idx:
                mins_ago = (len(insulin_flags) - 1 - idx) * 5
                active_insulin_on_board += float(insulin_doses[idx]) * np.exp(-0.693 * mins_ago / 60.0)
        else:
            minutes_since_last_insulin = 240.0
            active_insulin_on_board = 0.0

        # Temporal features (3)
        last_ts = pd.to_datetime(window["timestamp"].iloc[-1])
        hour = last_ts.hour + last_ts.minute / 60.0
        hour_sin = float(np.sin(2.0 * np.pi * hour / 24.0))
        hour_cos = float(np.cos(2.0 * np.pi * hour / 24.0))
        is_sleep_period = float(window["is_sleeping"].iloc[-1])

        features_dict = {
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
            "hr_mean_4h": hr_mean_4h,
            "hr_std_4h": hr_std_4h,
            "hr_current": hr_current,
            "hr_resting_estimate": hr_resting_estimate,
            "hrv_rmssd_mean_4h": hrv_rmssd_mean_4h,
            "hrv_rmssd_current": hrv_rmssd_current,
            "hrv_trend_4h": hrv_trend_4h,
            "steps_last_1h": steps_last_1h,
            "steps_last_4h": steps_last_4h,
            "active_minutes_1h": active_minutes_1h,
            "minutes_since_last_activity": minutes_since_last_activity,
            "minutes_since_last_meal": minutes_since_last_meal,
            "carbs_last_meal": carbs_last_meal,
            "total_carbs_4h": total_carbs_4h,
            "active_insulin_on_board": active_insulin_on_board,
            "minutes_since_last_insulin": minutes_since_last_insulin,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "is_sleep_period": is_sleep_period,
        }

        # Order precisely according to feature_names["dynamic"]
        ordered_vals = [features_dict[col] for col in self.feature_names["dynamic"]]
        return np.array(ordered_vals, dtype=np.float32)

    def predict(self) -> PredictionResult:
        """Run 2-hour ahead glucose prediction with SHAP local feature attributions."""
        dynamic_features = self._extract_dynamic_features()
        features = np.concatenate([self.static_features, dynamic_features])

        features_2d = features.reshape(1, -1)
        predicted_glucose = float(self.model.predict(features_2d)[0])

        # Adverse-event probability from the dedicated trained classifier.
        if self.classifier is not None:
            prob = float(self.classifier.predict_proba(features_2d)[0, 1])
        else:
            # Fallback heuristic (only used if the classifier model is unavailable)
            if predicted_glucose > 250.0:
                prob = min(1.0, 0.5 + (predicted_glucose - 250.0) / 100.0)
            elif predicted_glucose < 70.0:
                prob = min(1.0, 0.5 + (70.0 - predicted_glucose) / 40.0)
            else:
                prob = max(0.0, 1.0 - abs(predicted_glucose - 160.0) / 160.0) * 0.3

        if prob > 0.75:
            risk_level = "critical"
        elif prob > 0.50:
            risk_level = "high"
        elif prob > 0.25:
            risk_level = "moderate"
        else:
            risk_level = "low"

        # Compute SHAP attributions
        shap_vals = self.explainer.shap_values(features.reshape(1, -1))[0]
        feature_list = self.feature_names["all"]
        static_set = set(self.feature_names["static"])

        top_indices = np.argsort(np.abs(shap_vals))[::-1][:5]
        shap_top = []
        for idx in top_indices:
            shap_top.append({
                "feature": str(feature_list[idx]),
                "value": round(float(features[idx]), 2),
                "contribution": round(float(shap_vals[idx]), 2),
                "stream": "static" if feature_list[idx] in static_set else "dynamic"
            })

        result = PredictionResult(
            predicted_glucose_2h=round(predicted_glucose, 1),
            adverse_event_probability=round(float(prob), 4),
            risk_level=risk_level,
            shap_top_features=shap_top
        )
        self.last_prediction = result
        return result

    def simulate(
        self,
        interventions: Optional[List[Intervention]] = None,
        horizon_minutes: int = 120,
        reference_meal_carbs: float = 55.0,
        reference_meal_gi: float = 0.6,
    ) -> List[SimulationResult]:
        """Run baseline and counterfactual What-If simulations for interventions.

        To make the comparison clinically meaningful, the BASELINE represents the
        patient eating a typical upcoming meal (``reference_meal_carbs`` grams) with
        no corrective action. Each intervention is then evaluated against that same
        anticipated meal, so a "20-minute walk" or "+2U insulin" visibly blunts the
        post-meal rise, and a "low-carb swap" replaces the reference meal with a
        smaller one. This mirrors how a clinician would reason about a next meal.
        """
        if interventions is None:
            interventions = [
                Intervention(name="20-minute brisk walk after the meal", intervention_type="exercise", duration=20.0, intensity=0.5),
                Intervention(name="Low-carb meal swap (25g instead of 55g)", intervention_type="meal", carbs=25.0, gi_index=0.4),
                Intervention(name="+2 unit rapid insulin bolus with the meal", intervention_type="insulin", dose=2.0, insulin_type="rapid"),
            ]

        current_glucose = float(self.history["cgm_glucose"].iloc[-1]) if len(self.history) > 0 else float(self.params.G_b)

        # Smooth deterministic baseline starting from current glucose
        params_copy = copy.deepcopy(self.params)
        params_copy.G_b = current_glucose
        params_copy.process_noise_std = 0.0
        params_copy.sensor_noise_std = 0.0

        # The anticipated meal shared by the baseline and the non-meal interventions.
        reference_meal = {"time": 0.0, "carbs": float(reference_meal_carbs), "gi_index": float(reference_meal_gi)}

        def _create_sim_result(name: str, sim_output: dict) -> SimulationResult:
            times = sim_output["time_minutes"]
            gluc = sim_output["glucose"]
            trajectory = [
                {"time_offset_min": round(float(t), 1), "glucose": round(float(g), 1)}
                for t, g in zip(times, gluc)
            ]
            peak_g = float(np.max(gluc))
            min_g = float(np.min(gluc))
            time_to_peak = float(times[np.argmax(gluc)])
            in_range = (gluc >= 70.0) & (gluc <= 180.0)
            time_in_range_pct = round(float(np.mean(in_range) * 100.0), 1)
            enters_danger = bool(peak_g > 250.0 or min_g < 54.0)

            if peak_g > 300.0 or min_g < 50.0:
                risk = "critical"
            elif peak_g > 250.0 or min_g < 54.0:
                risk = "high"
            elif peak_g > 180.0 or min_g < 70.0:
                risk = "moderate"
            else:
                risk = "low"

            return SimulationResult(
                name=name,
                trajectory=trajectory,
                peak_glucose=round(peak_g, 1),
                min_glucose=round(min_g, 1),
                time_to_peak_min=round(time_to_peak, 1),
                time_in_range_pct=time_in_range_pct,
                enters_danger_zone=enters_danger,
                risk_level=risk
            )

        results = []

        # Baseline: patient eats the typical anticipated meal, takes no action.
        base_sim = simulate_glucose(
            params=params_copy,
            meals=[dict(reference_meal)],
            exercises=[],
            insulin_events=[],
            duration_minutes=horizon_minutes,
            dt=5.0
        )
        results.append(_create_sim_result("Baseline (usual meal, no action)", base_sim))

        # Counterfactual scenarios, each evaluated against the same anticipated meal.
        for item in interventions:
            # Exercise and insulin interventions act ON TOP of the reference meal;
            # a meal intervention REPLACES the reference meal with a different one.
            itype = str(item.intervention_type).lower()
            meals = [] if itype == "meal" else [dict(reference_meal)]
            exercises = []
            insulin_events = []

            if itype == "exercise":
                exercises.append({
                    "start": 0.0,
                    "duration": float(item.duration),
                    "intensity": float(item.intensity)
                })
            elif itype == "meal":
                meals.append({
                    "time": 0.0,
                    "carbs": float(item.carbs),
                    "gi_index": float(item.gi_index)
                })
            elif itype == "insulin":
                insulin_events.append({
                    "time": 0.0,
                    "dose": float(item.dose),
                    "type": str(item.insulin_type).lower()
                })

            sim_out = simulate_glucose(
                params=params_copy,
                meals=meals,
                exercises=exercises,
                insulin_events=insulin_events,
                duration_minutes=horizon_minutes,
                dt=5.0
            )
            results.append(_create_sim_result(item.name, sim_out))

        return results

    def get_current_state(self) -> dict:
        """Return a dict summarizing the digital twin's current physiological state."""
        risk_level = self.last_prediction.risk_level if self.last_prediction else "unknown"
        return {
            "patient_id": str(self.profile.get("patient_id", "")),
            "timestamp": str(self.history["timestamp"].iloc[-1]) if len(self.history) > 0 else None,
            "current_glucose": float(self.history["cgm_glucose"].iloc[-1]) if len(self.history) > 0 else None,
            "current_heart_rate": float(self.history["heart_rate"].iloc[-1]) if len(self.history) > 0 else None,
            "current_hrv": float(self.history["hrv_rmssd"].iloc[-1]) if len(self.history) > 0 else None,
            "is_sleeping": bool(self.history["is_sleeping"].iloc[-1]) if len(self.history) > 0 else False,
            "risk_level": risk_level,
        }

    def get_timeline(self, hours: int = 24) -> dict:
        """Return the last N hours of wearable readings formatted for JSON serialization."""
        # Clamp to a sane range: a negative value would make tail() drop rows from
        # the front (wrong slice), and absurdly large values are capped by history.
        hours = max(1, min(int(hours), 168))
        n_rows = min(hours * 12, len(self.history))
        recent = self.history.tail(n_rows)
        return {
            "timestamps": [str(t) for t in recent["timestamp"].tolist()],
            "glucose": [float(x) for x in recent["cgm_glucose"].round(1).tolist()],
            "heart_rate": [float(x) for x in recent["heart_rate"].round(1).tolist()],
            "hrv_rmssd": [float(x) for x in recent["hrv_rmssd"].round(1).tolist()],
            "steps": [int(x) for x in recent["steps_5min"].tolist()],
            "meal_flags": [int(x) for x in recent["meal_flag"].tolist()],
            "meal_carbs": [float(x) for x in recent["meal_carbs"].round(1).tolist()],
            "exercise_flags": [int(x) for x in recent["exercise_flag"].tolist()],
            "insulin_flags": [int(x) for x in recent["insulin_flag"].tolist()],
        }


def load_twin_from_files(
    patient_id: str,
    ehr_path: str = "data/synthea/patient_profiles.csv",
    wearable_dir: str = "data/wearable",
    model_path: str = "data/processed/xgb_fused_regressor.json",
    feature_names_path: str = "data/processed/feature_names.json",
    classifier_path: str = "data/processed/xgb_fused_classifier.json",
) -> DigitalTwin:
    """Convenience function to load a fully initialized DigitalTwin from files."""
    profiles = pd.read_csv(ehr_path)
    matched = profiles[profiles["patient_id"] == patient_id]
    if len(matched) == 0:
        raise ValueError(f"Patient {patient_id} not found in {ehr_path}")
    profile = matched.iloc[0].to_dict()

    twin = DigitalTwin(profile, model_path, feature_names_path, classifier_path)

    wearable_path = os.path.join(wearable_dir, f"{patient_id}.csv")
    if not os.path.exists(wearable_path):
        raise FileNotFoundError(f"Wearable file {wearable_path} does not exist")
    wearable_df = pd.read_csv(wearable_path)
    twin.load_history(wearable_df)

    return twin


if __name__ == "__main__":
    # Demo: load a twin for patient P001
    twin = load_twin_from_files("P001")

    # Predict
    prediction = twin.predict()
    print(f"\n{'='*60}")
    print(f"DIGITAL TWIN PREDICTION — Patient {twin.profile['patient_id']}")
    print(f"{'='*60}")
    print(f"Predicted glucose (2h): {prediction.predicted_glucose_2h:.1f} mg/dL")
    print(f"Adverse event probability: {prediction.adverse_event_probability:.2%}")
    print(f"Risk level: {prediction.risk_level}")
    print(f"\nTop contributing factors:")
    for f in prediction.shap_top_features:
        direction = "↑" if f["contribution"] > 0 else "↓"
        stream_icon = "🔵" if f["stream"] == "static" else "🟠"
        print(f"  {stream_icon} {f['feature']}: {f['contribution']:+.1f} ({f['stream']})")

    # Simulate
    print(f"\n{'='*60}")
    print("WHAT-IF SIMULATION")
    print(f"{'='*60}")
    scenarios = twin.simulate()
    for s in scenarios:
        danger = "⚠️" if s.enters_danger_zone else "✅"
        print(f"\n{danger} {s.name}:")
        print(f"  Peak: {s.peak_glucose:.1f} mg/dL | Min: {s.min_glucose:.1f} mg/dL")
        print(f"  Time in range: {s.time_in_range_pct:.1f}% | Risk: {s.risk_level}")
