"""Bergman Minimal Model implementation for synthetic Continuous Glucose Monitor (CGM) data.

This module simulates glucose-insulin dynamics in patients with Type 2 Diabetes Mellitus (T2DM).
It implements:
1. Two-compartment Bergman Minimal Model equations:
   dG/dt = -S_G * (G - G_b) - X * G + Ra(t) - Ex(t)
   dX/dt = -p2 * X + p3 * (I(t) - I_b)
2. Double-exponential (Dalla Man) gut carbohydrate absorption kinetics.
3. Post-exercise insulin-independent glucose uptake and clearance decay.
4. Circadian hepatic gluconeogenesis elevation (Dawn phenomenon).
5. Exogenous basal/bolus insulin pharmacokinetics.
6. Process (physiological) and sensor measurement noise modeling.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class PatientParams:
    """Dataclass holding individual patient parameters for the Bergman Minimal Model.

    Attributes
    ----------
    patient_id : str
        Unique identifier for the patient.
    G_b : float
        Basal glucose concentration in mg/dL (derived from fasting glucose in EHR).
    I_b : float
        Basal plasma insulin concentration in μU/mL (estimated from medications).
    S_G : float
        Fractional glucose effectiveness at basal insulin (1/min).
    p2 : float
        Insulin action spontaneous decay rate constant (1/min).
    p3 : float
        Insulin sensitivity rate parameter (1/min per μU/mL).
    insulin_sensitivity : float
        Derived insulin sensitivity index S_I = p3 / p2 (dL/min per μU).
    carb_sensitivity : float
        Glycemic response factor per gram of carbohydrate (mg/dL per g).
    absorption_rate : float
        Base gastrointestinal glucose absorption rate constant (1/min).
    exercise_sensitivity : float
        Exercise-induced glucose disposal clearance rate (mg/dL/min per unit intensity).
    dawn_amplitude : float
        Peak elevation of circadian morning hepatic glucose output (mg/dL).
    process_noise_std : float
        Standard deviation of intrinsic physiological/metabolic process noise (mg/dL).
    sensor_noise_std : float
        Standard deviation of CGM sensor measurement error (mg/dL).
    """

    # Identity
    patient_id: str

    # Bergman parameters
    G_b: float
    I_b: float
    S_G: float
    p2: float
    p3: float

    # Derived
    insulin_sensitivity: float

    # Meal response
    carb_sensitivity: float
    absorption_rate: float

    # Exercise response
    exercise_sensitivity: float

    # Circadian
    dawn_amplitude: float

    # Noise
    process_noise_std: float
    sensor_noise_std: float

    @classmethod
    def from_ehr_profile(cls, profile: Dict[str, Any]) -> "PatientParams":
        """Derive patient-specific ODE parameters from Electronic Health Record (EHR) data.

        Parameters
        ----------
        profile : dict
            Dictionary containing patient EHR biomarkers:
            - fasting_glucose (float): Basal fasting plasma glucose in mg/dL.
            - hba1c (float): Glycated hemoglobin percentage.
            - bmi (float): Body Mass Index in kg/m^2.
            - age (float, optional): Age in years.
            - med_metformin (int, optional): Flag indicating metformin therapy.
            - med_insulin_basal (int, optional): Flag indicating basal insulin therapy.
            - med_insulin_rapid (int, optional): Flag indicating prandial insulin therapy.
            - med_sglt2i (int, optional): Flag indicating SGLT2 inhibitor therapy.
            - med_glp1ra (int, optional): Flag indicating GLP-1 receptor agonist therapy.
            - patient_id (str, optional): Patient identifier string.

        Returns
        -------
        PatientParams
            Initialized model parameter instance.
        """
        patient_id: str = str(profile.get("patient_id", "P001"))
        G_b: float = float(profile["fasting_glucose"])

        # Basal insulin: baseline 8.0 μU/mL, elevated in patients requiring exogenous insulin
        I_b: float = (
            8.0
            + (5.0 if profile.get("med_insulin_basal", 0) else 0.0)
            + (3.0 if profile.get("med_insulin_rapid", 0) else 0.0)
        )

        hba1c: float = float(profile["hba1c"])
        bmi: float = float(profile["bmi"])
        age: float = float(profile.get("age", 55.0))

        # Glucose effectiveness S_G: Higher HbA1c impairs insulin-independent glucose disposal
        S_G: float = float(np.clip(0.025 * (10.0 - hba1c) / 4.0, 0.005, 0.04))

        # Insulin action decay p2: Fixed standard literature constant
        p2: float = 0.025

        # Insulin action sensitivity p3: Improved by insulin-sensitizing and incretin therapies
        med_metformin: float = float(profile.get("med_metformin", 0))
        med_sglt2i: float = float(profile.get("med_sglt2i", 0))
        med_glp1ra: float = float(profile.get("med_glp1ra", 0))
        p3_raw: float = S_G * 0.4 * (
            1.0 + 0.15 * med_metformin + 0.10 * med_sglt2i + 0.20 * med_glp1ra
        )
        p3: float = float(np.clip(p3_raw, 0.00001, 0.0005))

        # Derived insulin sensitivity index S_I = p3 / p2
        insulin_sensitivity: float = float(p3 / p2)

        # Carbohydrate sensitivity: Higher HbA1c and severe adiposity amplify postprandial glycemic excursions
        carb_sens_raw: float = 1.5 + 0.6 * (hba1c - 5.5) + 0.2 * (bmi - 25.0) / 10.0
        carb_sensitivity: float = float(np.clip(carb_sens_raw, 0.8, 5.0))

        # Base gastrointestinal absorption rate (~40 min peak absorption for moderate GI meal)
        absorption_rate: float = 0.025

        # Exercise sensitivity: Preserved uptake in leaner/younger patients, reduced with adiposity and aging
        ex_sens_raw: float = max(0.3, 1.2 - 0.01 * (bmi - 22.0) - 0.005 * (age - 40.0))
        exercise_sensitivity: float = float(np.clip(ex_sens_raw, 0.3, 1.5))

        # Dawn phenomenon amplitude: Hepatic insulin resistance causes pronounced morning gluconeogenesis
        dawn_amp_raw: float = 8.0 + 4.0 * (hba1c - 5.5)
        dawn_amplitude: float = float(np.clip(dawn_amp_raw, 5.0, 35.0))

        # Noise parameters
        process_noise_std: float = 3.0
        sensor_noise_std: float = 5.0

        return cls(
            patient_id=patient_id,
            G_b=G_b,
            I_b=I_b,
            S_G=S_G,
            p2=p2,
            p3=p3,
            insulin_sensitivity=insulin_sensitivity,
            carb_sensitivity=carb_sensitivity,
            absorption_rate=absorption_rate,
            exercise_sensitivity=exercise_sensitivity,
            dawn_amplitude=dawn_amplitude,
            process_noise_std=process_noise_std,
            sensor_noise_std=sensor_noise_std,
        )


def meal_absorption_rate(
    t: float,
    meal_time: float,
    carbs: float,
    gi_index: float,
    params: PatientParams,
) -> float:
    """Compute rate of glucose appearance Ra(t) from a meal event.

    Uses a double-exponential (Dalla Man) absorption curve.
    Higher glycemic index (GI) accelerates the absorption rate constant k_abs.

    Parameters
    ----------
    t : float
        Current simulation time in minutes.
    meal_time : float
        Time the meal occurred in minutes from start of simulation.
    carbs : float
        Amount of carbohydrates consumed in grams.
    gi_index : float
        Glycemic index of the meal (normalized scale 0.0 to 1.0).
    params : PatientParams
        Patient physiological parameters.

    Returns
    -------
    float
        Rate of glucose appearance Ra(t) in mg/dL/min, clipped to >= 0.0.
    """
    dt: float = t - meal_time
    if dt <= 0.0:
        return 0.0

    k_abs: float = params.absorption_rate * (0.8 + 0.4 * gi_index)
    k_gut: float = k_abs * 0.6
    diff: float = k_abs - k_gut

    # L'Hôpital edge case when gut transit and absorption rates are identical
    if abs(diff) < 1e-8:
        ra = carbs * params.carb_sensitivity * k_abs * dt * np.exp(-k_abs * dt)
    else:
        ra = (
            carbs
            * params.carb_sensitivity
            * k_abs
            * (np.exp(-k_gut * dt) - np.exp(-k_abs * dt))
            / diff
        )

    return float(max(0.0, ra))


def exercise_uptake_rate(
    t: float,
    exercise_start: float,
    duration: float,
    intensity: float,
    params: PatientParams,
) -> float:
    """Compute rate of glucose disappearance Ex(t) from an exercise session.

    Models non-insulin-mediated glucose uptake during activity (GLUT4 translocation)
    and residual clearance decay post-exercise.

    Parameters
    ----------
    t : float
        Current simulation time in minutes.
    exercise_start : float
        Start time of the exercise in minutes.
    duration : float
        Duration of the exercise in minutes.
    intensity : float
        Exercise intensity (normalized scale 0.0 to 1.0).
    params : PatientParams
        Patient physiological parameters.

    Returns
    -------
    float
        Rate of glucose disappearance Ex(t) in mg/dL/min, clipped to >= 0.0.
    """
    dt: float = t - exercise_start

    # During exercise phase
    if 0.0 <= dt <= duration:
        ex = intensity * params.exercise_sensitivity
    # Post-exercise recovery phase (exponential decay over 30 minutes)
    elif duration < dt <= duration + 30.0:
        ex = (
            intensity
            * params.exercise_sensitivity
            * np.exp(-0.05 * (dt - duration))
        )
    else:
        ex = 0.0

    return float(max(0.0, ex))


def dawn_phenomenon(t_minutes_from_midnight: float, params: PatientParams) -> float:
    """Compute early morning circadian hepatic glucose elevation (Dawn phenomenon).

    Generates a Gaussian elevation profile centered at 5:00 AM with a standard width of ~1.5 hours.

    Parameters
    ----------
    t_minutes_from_midnight : float
        Time in minutes from midnight (0.0 to 1440.0).
    params : PatientParams
        Patient physiological parameters.

    Returns
    -------
    float
        Circadian glucose elevation in mg/dL.
    """
    hour: float = (t_minutes_from_midnight % 1440.0) / 60.0
    dawn: float = params.dawn_amplitude * np.exp(-0.5 * (((hour - 5.0) / 1.5) ** 2))
    return float(dawn)


def simulate_glucose(
    params: PatientParams,
    meals: List[Dict[str, Any]],
    exercises: List[Dict[str, Any]],
    insulin_events: List[Dict[str, Any]],
    duration_minutes: float,
    dt: float = 5.0,
) -> Dict[str, np.ndarray]:
    """Simulate continuous glucose monitor (CGM) dynamics over a specified duration.

    Solves the Bergman Minimal Model ODE system using scipy.integrate.solve_ivp with RK45:
        dG/dt = -S_G * (G - G_b(t)) - X * G + Ra(t) - Ex(t) + process_noise
        dX/dt = -p2 * X + p3 * (I(t) - I_b)

    Parameters
    ----------
    params : PatientParams
        Patient model parameters.
    meals : list of dict
        Meal events with keys: "time" (minutes), "carbs" (grams), "gi_index" (float 0-1).
    exercises : list of dict
        Exercise events with keys: "start" (minutes), "duration" (minutes), "intensity" (float 0-1).
    insulin_events : list of dict
        Insulin administration events with keys: "time" (minutes), "dose" (Units), "type" ("basal" or "rapid").
    duration_minutes : float
        Total simulation duration in minutes.
    dt : float, default=5.0
        Sampling step interval in minutes.

    Returns
    -------
    dict
        Dictionary containing:
        - "time_minutes": np.ndarray of simulation timestamps.
        - "glucose": np.ndarray of CGM glucose values in mg/dL (clipped to [40, 400]).
    """
    t_eval: np.ndarray = np.arange(0.0, float(duration_minutes) + 0.5 * dt, float(dt))
    t_span: tuple = (0.0, float(duration_minutes))

    # Pre-generate smooth continuous process noise over a regular grid for RK45 integration stability
    np.random.seed(42)
    grid_points: int = len(t_eval) + 2
    t_grid: np.ndarray = np.linspace(0.0, float(duration_minutes) + dt, grid_points)
    # Scaled to maintain physiological variance (process_noise_std) across differential time-steps
    noise_rate_std = params.process_noise_std / np.sqrt(dt)
    noise_vals: np.ndarray = np.random.normal(0.0, noise_rate_std, size=grid_points)

    def _get_process_noise(t_curr: float) -> float:
        return float(np.interp(t_curr, t_grid, noise_vals))

    def _insulin_level(t_curr: float) -> float:
        """Calculate total plasma insulin I(t) at time t_curr."""
        extra_insulin: float = 0.0
        for event in insulin_events:
            dt_ins = t_curr - float(event["time"])
            if 0.0 < dt_ins < 400.0:
                dose = float(event["dose"])
                ins_type = str(event.get("type", "rapid")).lower()
                k = 0.02 if ins_type == "basal" else 0.05
                bump = dose * 10.0 * k * np.exp(-k * dt_ins)
                extra_insulin += bump
        return params.I_b + extra_insulin

    def _meal_rate(t_curr: float) -> float:
        """Calculate total rate of glucose appearance Ra(t) into systemic circulation.

        To satisfy the clinical minimal model definition where total glycemic excursion
        integral matches carbs * carb_sensitivity, the rate of appearance flux is
        normalized by the intestinal transit rate constant k_gut.
        """
        total_ra: float = 0.0
        for meal in meals:
            dt_m = t_curr - float(meal["time"])
            if 0.0 < dt_m < 360.0:
                gi = float(meal.get("gi_index", 0.6))
                k_abs = params.absorption_rate * (0.8 + 0.4 * gi)
                k_gut = k_abs * 0.6
                raw_ra = meal_absorption_rate(
                    t=t_curr,
                    meal_time=float(meal["time"]),
                    carbs=float(meal["carbs"]),
                    gi_index=gi,
                    params=params,
                )
                # Area-normalized flux: ensures peak glucose rise equals carbs * carb_sensitivity
                total_ra += raw_ra * k_gut
        return total_ra

    def _exercise_rate(t_curr: float) -> float:
        """Calculate total rate of exercise-induced glucose disposal Ex(t)."""
        total_ex: float = 0.0
        for ex in exercises:
            dt_e = t_curr - float(ex["start"])
            duration = float(ex["duration"])
            if 0.0 <= dt_e <= duration + 30.0:
                total_ex += exercise_uptake_rate(
                    t=t_curr,
                    exercise_start=float(ex["start"]),
                    duration=duration,
                    intensity=float(ex.get("intensity", 0.5)),
                    params=params,
                )
        return total_ex

    def bergman_ode(t_curr: float, state: List[float]) -> List[float]:
        """Compute system state derivatives [dG/dt, dX/dt]."""
        G: float = state[0]
        X: float = state[1]

        # Circadian baseline elevation at 5:00 AM (Dawn phenomenon)
        G_b_eff: float = params.G_b + dawn_phenomenon(t_curr, params)

        # External metabolic inputs
        Ra: float = _meal_rate(t_curr)
        Ex: float = _exercise_rate(t_curr)
        proc_noise: float = _get_process_noise(t_curr)

        # Bergman minimal model equations
        dG_dt = -params.S_G * (G - G_b_eff) - X * G + Ra - Ex + proc_noise
        dX_dt = -params.p2 * X + params.p3 * (_insulin_level(t_curr) - params.I_b)

        return [dG_dt, dX_dt]

    # Initial conditions: Basal equilibrium G(0) = G_b, remote insulin action X(0) = 0.0
    initial_state = [params.G_b, 0.0]

    sol = solve_ivp(
        fun=bergman_ode,
        t_span=t_span,
        y0=initial_state,
        method="RK45",
        t_eval=t_eval,
        max_step=dt,
    )

    raw_glucose: np.ndarray = sol.y[0]

    # Add Gaussian sensor measurement noise
    sensor_noise: np.ndarray = np.random.normal(
        0.0, params.sensor_noise_std, size=len(raw_glucose)
    )
    cgm_glucose: np.ndarray = raw_glucose + sensor_noise

    # Enforce physiological sensor limits [40, 400] mg/dL
    cgm_glucose = np.clip(cgm_glucose, 40.0, 400.0)

    return {
        "time_minutes": sol.t,
        "glucose": cgm_glucose,
    }


def simulate_patient_day(
    params: PatientParams,
    day_schedule: Dict[str, Any],
    t_start: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Convenience function simulating a full 24-hour day (1440 minutes).

    Parameters
    ----------
    params : PatientParams
        Patient parameters.
    day_schedule : dict
        Dictionary containing event lists:
        - "meals": list of meal dicts {"time", "carbs", "gi_index"}
        - "exercises": list of exercise dicts {"start", "duration", "intensity"}
        - "insulin_events": list of insulin dicts {"time", "dose", "type"}
    t_start : float, default=0.0
        Starting time offset in minutes from midnight.

    Returns
    -------
    dict
        Dictionary containing "time_minutes" and "glucose" arrays.
    """
    meals = day_schedule.get("meals", [])
    exercises = day_schedule.get("exercises", [])
    insulin_events = day_schedule.get("insulin_events", [])

    return simulate_glucose(
        params=params,
        meals=meals,
        exercises=exercises,
        insulin_events=insulin_events,
        duration_minutes=1440.0,
        dt=5.0,
    )


if __name__ == "__main__":
    # Create sample patient profile (T2DM with uncontrolled fasting glycemia and obesity)
    sample_ehr = {
        "patient_id": "P_SAMPLE_001",
        "fasting_glucose": 150.0,
        "hba1c": 8.5,
        "bmi": 31.0,
        "age": 55,
        "med_metformin": 1,
        "med_insulin_basal": 0,
        "med_insulin_rapid": 0,
        "med_sglt2i": 0,
        "med_glp1ra": 0,
    }

    sample_params = PatientParams.from_ehr_profile(sample_ehr)

    # 24-hour daily schedule:
    # - Breakfast (45g carbs, GI 0.6) at t = 450 min (7:30 AM)
    # - Lunch (60g carbs, GI 0.6) at t = 750 min (12:30 PM)
    # - Post-lunch 20-min brisk walk at t = 780 min (1:00 PM)
    # - Dinner (70g carbs, GI 0.6) at t = 1140 min (7:00 PM)
    sample_day_schedule = {
        "meals": [
            {"time": 450.0, "carbs": 45.0, "gi_index": 0.6},
            {"time": 750.0, "carbs": 60.0, "gi_index": 0.6},
            {"time": 1140.0, "carbs": 70.0, "gi_index": 0.6},
        ],
        "exercises": [
            {"start": 780.0, "duration": 20.0, "intensity": 0.5},
        ],
        "insulin_events": [],
    }

    # Run 24-hour simulation
    cgm_data = simulate_patient_day(sample_params, sample_day_schedule)

    time_mins = cgm_data["time_minutes"]
    glucose = cgm_data["glucose"]

    min_g = float(np.min(glucose))
    max_g = float(np.max(glucose))
    mean_g = float(np.mean(glucose))

    print("=" * 60)
    print("      CONTINUOUS GLUCOSE MONITOR (CGM) SIMULATION STATS      ")
    print("=" * 60)
    print(f"Patient ID         : {sample_params.patient_id}")
    print(f"Fasting Glucose    : {sample_params.G_b:.1f} mg/dL")
    print(f"HbA1c              : {sample_ehr['hba1c']:.1f}%")
    print(f"BMI                : {sample_ehr['bmi']:.1f} kg/m²")
    print("-" * 60)
    print(f"Simulated Min      : {min_g:.1f} mg/dL  (Target: > 50 mg/dL)")
    print(f"Simulated Max      : {max_g:.1f} mg/dL  (Target: < 350 mg/dL)")
    print(f"Simulated Mean     : {mean_g:.1f} mg/dL")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs("data/wearable", exist_ok=True)
    plot_path = "data/wearable/sample_glucose_trace.png"

    # Plot synthetic CGM trace
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)

    # Target clinical Time-in-Range (TIR) band: 70 - 180 mg/dL
    ax.axhspan(70, 180, color="#10B981", alpha=0.15, label="Target Range (70–180 mg/dL)")
    ax.axhline(sample_params.G_b, color="#64748B", linestyle=":", alpha=0.7, label=f"Fasting Baseline ({sample_params.G_b:.0f} mg/dL)")

    # Continuous CGM line
    hours = time_mins / 60.0
    ax.plot(hours, glucose, color="#2563EB", linewidth=2.0, label="CGM Glucose (mg/dL)")

    # Annotate Dawn Phenomenon at 5:00 AM (t = 300 min)
    dawn_val = sample_params.G_b + sample_params.dawn_amplitude
    ax.scatter([5.0], [dawn_val], color="#F59E0B", s=60, zorder=5)
    ax.annotate(
        "Dawn Phenomenon\n(~5:00 AM)",
        xy=(5.0, dawn_val),
        xytext=(3.5, dawn_val + 25),
        arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.5),
        fontsize=9,
        fontweight="semibold",
        color="#B45309",
    )

    # Annotate meals
    meal_labels = [
        ("Breakfast (45g)", 450.0 / 60.0),
        ("Lunch (60g)", 750.0 / 60.0),
        ("Dinner (70g)", 1140.0 / 60.0),
    ]
    for label, meal_hr in meal_labels:
        ax.axvline(meal_hr, color="#EF4444", linestyle="--", alpha=0.6, linewidth=1.2)
        ax.text(
            meal_hr + 0.15,
            320,
            label,
            color="#B91C1C",
            fontsize=8.5,
            fontweight="medium",
            rotation=0,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEE2E2", edgecolor="#FCA5A5", alpha=0.8),
        )

    # Annotate exercise
    ex_hr_start = 780.0 / 60.0
    ex_hr_end = 800.0 / 60.0
    ax.axvspan(ex_hr_start, ex_hr_end, color="#3B82F6", alpha=0.2, label="Post-Lunch Walk (20 min)")

    ax.set_title(
        f"Synthetic CGM Glucose Dynamics (24-Hour Profile) — Patient {sample_params.patient_id}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Time of Day (Hours from Midnight)", fontsize=10, fontweight="semibold")
    ax.set_ylabel("Sensor Glucose (mg/dL)", fontsize=10, fontweight="semibold")
    ax.set_xlim(0, 24)
    ax.set_ylim(40, 360)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved plot successfully to: {plot_path}")
