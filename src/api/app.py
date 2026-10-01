from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.twin_engine.digital_twin import (
    DigitalTwin,
    Intervention,
    PredictionResult,
    SimulationResult,
    load_twin_from_files,
)

# Global cache
twins_cache: Dict[str, DigitalTwin] = {}
patient_list: List[str] = []


def _load_patients_fallback():
    """Ensure patient list is loaded into memory."""
    global patient_list
    if not patient_list:
        ehr_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthea", "patient_profiles.csv")
        if os.path.exists(ehr_path):
            df = pd.read_csv(ehr_path)
            patient_list = df["patient_id"].tolist()
            print(f"Loaded {len(patient_list)} patients")
        else:
            print(f"WARNING: {ehr_path} not found")


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Modern lifespan context manager for startup and shutdown."""
    _load_patients_fallback()
    yield


app = FastAPI(
    title="Cardiometabolic Digital Twin API",
    description="API for patient digital twin prediction and simulation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_twin(patient_id: str) -> DigitalTwin:
    """Get or create a cached DigitalTwin for the given patient."""
    if patient_id not in twins_cache:
        try:
            twin = load_twin_from_files(patient_id)
            twins_cache[patient_id] = twin
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found: {str(e)}")
    return twins_cache[patient_id]


# Pydantic models for request/response
class PatientProfile(BaseModel):
    patient_id: str
    age: int
    sex: str
    bmi: float
    hba1c: float
    fasting_glucose: float
    egfr: float
    ldl: float
    hdl: float
    systolic_bp: float
    years_with_t2dm: float
    comorbidity_count: int
    med_metformin: int
    med_insulin_basal: int
    med_insulin_rapid: int
    med_sglt2i: int
    med_glp1ra: int
    insulin_total_daily_dose: float
    genetic_risk_score: float


class PredictionResponse(BaseModel):
    patient_id: str
    predicted_glucose_2h: float
    adverse_event_probability: float
    risk_level: str
    shap_top_features: list


class SimulationRequest(BaseModel):
    interventions: Optional[List[dict]] = None  # If None, use defaults


class TrajectoryPoint(BaseModel):
    time_offset_min: float
    glucose: float


class SimulationScenario(BaseModel):
    name: str
    trajectory: List[dict]
    peak_glucose: float
    min_glucose: float
    time_to_peak_min: float
    time_in_range_pct: float
    enters_danger_zone: bool
    risk_level: str


class SimulationResponse(BaseModel):
    patient_id: str
    scenarios: List[SimulationScenario]


class TimelineResponse(BaseModel):
    patient_id: str
    timestamps: List[str]
    glucose: List[float]
    heart_rate: List[float]
    hrv_rmssd: List[float]
    steps: List[int]
    meal_flags: List[int]
    meal_carbs: List[float]
    exercise_flags: List[int]
    insulin_flags: List[int]


# API Endpoints
@app.get("/api/patients")
def list_patients():
    """Return list of all patient IDs."""
    _load_patients_fallback()
    return {"patients": patient_list, "count": len(patient_list)}


@app.get("/api/patient/{patient_id}", response_model=PatientProfile)
def get_patient(patient_id: str):
    """Return patient EHR profile."""
    twin = get_twin(patient_id)
    profile = twin.profile
    clean_profile = {
        "patient_id": str(profile["patient_id"]),
        "age": int(profile["age"]),
        "sex": str(profile["sex"]),
        "bmi": float(profile["bmi"]),
        "hba1c": float(profile["hba1c"]),
        "fasting_glucose": float(profile["fasting_glucose"]),
        "egfr": float(profile["egfr"]),
        "ldl": float(profile["ldl"]),
        "hdl": float(profile["hdl"]),
        "systolic_bp": float(profile["systolic_bp"]),
        "years_with_t2dm": float(profile["years_with_t2dm"]),
        "comorbidity_count": int(profile["comorbidity_count"]),
        "med_metformin": int(profile["med_metformin"]),
        "med_insulin_basal": int(profile["med_insulin_basal"]),
        "med_insulin_rapid": int(profile["med_insulin_rapid"]),
        "med_sglt2i": int(profile["med_sglt2i"]),
        "med_glp1ra": int(profile["med_glp1ra"]),
        "insulin_total_daily_dose": float(profile["insulin_total_daily_dose"]),
        "genetic_risk_score": float(profile["genetic_risk_score"]),
    }
    return clean_profile


@app.get("/api/patient/{patient_id}/state")
def get_patient_state(patient_id: str):
    """Return current twin state."""
    twin = get_twin(patient_id)
    return twin.get_current_state()


@app.get("/api/patient/{patient_id}/timeline", response_model=TimelineResponse)
def get_timeline(patient_id: str, hours: int = 24):
    """Return historical wearable data for charting."""
    twin = get_twin(patient_id)
    timeline = twin.get_timeline(hours=hours)
    timeline["patient_id"] = patient_id
    return timeline


@app.get("/api/patient/{patient_id}/predict", response_model=PredictionResponse)
def predict(patient_id: str):
    """Run 2-hour glucose prediction with SHAP explanation."""
    twin = get_twin(patient_id)
    result = twin.predict()
    return PredictionResponse(
        patient_id=patient_id,
        predicted_glucose_2h=round(result.predicted_glucose_2h, 1),
        adverse_event_probability=round(result.adverse_event_probability, 3),
        risk_level=result.risk_level,
        shap_top_features=result.shap_top_features,
    )


@app.post("/api/patient/{patient_id}/simulate", response_model=SimulationResponse)
def simulate(patient_id: str, request: SimulationRequest = SimulationRequest()):
    """Run counterfactual What-If simulations."""
    twin = get_twin(patient_id)

    interventions = None
    if request.interventions:
        interventions = []
        for i in request.interventions:
            interventions.append(
                Intervention(
                    name=i.get("name", "Custom intervention"),
                    intervention_type=i.get("intervention_type", "exercise"),
                    duration=float(i.get("duration", 20.0)),
                    intensity=float(i.get("intensity", 0.5)),
                    carbs=float(i.get("carbs", 0.0)),
                    gi_index=float(i.get("gi_index", 0.5)),
                    dose=float(i.get("dose", 0.0)),
                    insulin_type=str(i.get("insulin_type", "rapid")),
                )
            )

    results = twin.simulate(interventions=interventions)

    scenarios = []
    for r in results:
        scenarios.append(
            SimulationScenario(
                name=r.name,
                trajectory=r.trajectory,
                peak_glucose=round(r.peak_glucose, 1),
                min_glucose=round(r.min_glucose, 1),
                time_to_peak_min=round(r.time_to_peak_min, 1),
                time_in_range_pct=round(r.time_in_range_pct, 1),
                enters_danger_zone=r.enters_danger_zone,
                risk_level=r.risk_level,
            )
        )

    return SimulationResponse(patient_id=patient_id, scenarios=scenarios)


# Serve the dashboard
dashboard_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dashboard")
if os.path.isdir(dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")


@app.get("/")
def root():
    """Redirect to dashboard or return API status."""
    dashboard_index = os.path.join(dashboard_dir, "index.html")
    if os.path.exists(dashboard_index):
        return FileResponse(dashboard_index)
    return {"message": "Cardiometabolic Digital Twin API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    # Change to project root so relative paths work
    os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
