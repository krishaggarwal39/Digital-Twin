from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
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

# Global cache. FastAPI runs the sync (`def`) route handlers in a threadpool, so
# multiple requests touch `twins_cache` concurrently from different threads.
# `_cache_lock` guards the cache dict and the per-patient lock registry; the
# per-patient `_patient_locks` serialize the expensive load for a single patient
# without blocking loads of *other* patients (so one slow patient can't stall the
# whole demo).
twins_cache: Dict[str, DigitalTwin] = {}
patient_list: List[str] = []
_cache_lock = threading.Lock()
_patient_locks: Dict[str, threading.Lock] = {}


def _load_patients_fallback():
    """Ensure patient list is loaded into memory.

    Only list patients whose wearable time-series file actually exists, so the
    dashboard never offers a patient the API cannot load. This keeps the demo
    robust whether the deployment ships the full 300-patient cohort or a curated
    subset.
    """
    global patient_list
    if not patient_list:
        base = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        ehr_path = os.path.join(base, "synthea", "patient_profiles.csv")
        wearable_dir = os.path.join(base, "wearable")
        if os.path.exists(ehr_path):
            df = pd.read_csv(ehr_path)
            all_ids = df["patient_id"].tolist()
            if os.path.isdir(wearable_dir):
                available = {
                    os.path.splitext(f)[0]
                    for f in os.listdir(wearable_dir)
                    if f.endswith(".csv")
                }
                filtered = [pid for pid in all_ids if pid in available]
                patient_list = filtered if filtered else all_ids
            else:
                patient_list = all_ids
            print(f"Loaded {len(patient_list)} patients (of {len(all_ids)} profiles)")
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

# CORS. The dashboard is served same-origin by this app, so credentials are not
# needed. A wildcard origin combined with allow_credentials=True is rejected by
# browsers, so we keep credentials disabled, which makes the wildcard valid and
# still allows a separately-hosted dashboard to call the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def get_twin(patient_id: str) -> DigitalTwin:
    """Get or create a cached DigitalTwin for the given patient.

    Thread-safe: concurrent requests for the same uncached patient load the twin
    exactly once instead of racing to build and overwrite the same cache entry.
    """
    # Fast path: already cached (dict reads are atomic under the GIL).
    twin = twins_cache.get(patient_id)
    if twin is not None:
        return twin

    # Grab (or create) the lock dedicated to this patient, under the registry lock.
    with _cache_lock:
        patient_lock = _patient_locks.setdefault(patient_id, threading.Lock())

    # Only one thread loads a given patient; others wait here, then hit the cache.
    with patient_lock:
        twin = twins_cache.get(patient_id)
        if twin is None:
            try:
                twin = load_twin_from_files(patient_id)
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found: {str(e)}")
            twins_cache[patient_id] = twin
        return twin


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
    reference_meal_carbs: Optional[float] = None  # anticipated meal (g); None -> engine default (55)
    reference_meal_gi: Optional[float] = None     # anticipated meal GI 0-1; None -> engine default (0.6)


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


def _read_json(filename: str) -> Optional[dict]:
    """Load a JSON artifact from data/processed, returning None if absent."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


@app.get("/api/validation")
def get_validation():
    """Return the real-world clinical validation results.

    Aggregates the honest sim-to-real study on the ShanghaiT2DM cohort:
      - zero_shot: synthetic-trained model applied to real patients (the gap)
      - real_adapted: the architecture trained + tested on real patients
      - clinical: Clarke Error Grid and adverse-event lead-time
    """
    return {
        "zero_shot": _read_json("real_world_results.json"),
        "real_adapted": _read_json("real_world_trained_results.json"),
        "clinical": _read_json("clinical_metrics_real.json"),
        "synthetic": _read_json("evaluation_results.json"),
        "multi_horizon": _read_json("multi_horizon_results.json"),
    }


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
    def _clamp(x: float, lo: float, hi: float) -> float:
        try:
            return float(min(max(float(x), lo), hi))
        except (TypeError, ValueError):
            return lo

    if request.interventions:
        interventions = []
        for i in request.interventions:
            # Clamp to clinically sane ranges so a hand-crafted API payload cannot
            # drive the ODE with nonsensical values (the UI sliders are already bounded).
            interventions.append(
                Intervention(
                    name=str(i.get("name", "Custom intervention"))[:80],
                    intervention_type=i.get("intervention_type", "exercise"),
                    duration=_clamp(i.get("duration", 20.0), 0.0, 180.0),
                    intensity=_clamp(i.get("intensity", 0.5), 0.0, 1.0),
                    carbs=_clamp(i.get("carbs", 0.0), 0.0, 200.0),
                    gi_index=_clamp(i.get("gi_index", 0.5), 0.0, 1.0),
                    dose=_clamp(i.get("dose", 0.0), 0.0, 50.0),
                    insulin_type=str(i.get("insulin_type", "rapid")),
                )
            )

    # Only override the engine's reference-meal defaults when the client sends them,
    # so an empty POST {} keeps the exact previous behavior.
    sim_kwargs: Dict[str, Any] = {"interventions": interventions}
    if request.reference_meal_carbs is not None:
        sim_kwargs["reference_meal_carbs"] = _clamp(request.reference_meal_carbs, 0.0, 200.0)
    if request.reference_meal_gi is not None:
        sim_kwargs["reference_meal_gi"] = _clamp(request.reference_meal_gi, 0.0, 1.0)

    try:
        results = twin.simulate(**sim_kwargs)
    except (ValueError, FloatingPointError) as e:
        # A pathological intervention that destabilizes the ODE -> clean 422,
        # never a 500.
        raise HTTPException(status_code=422, detail=f"Simulation could not run: {e}")

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


# Serve the dashboard and assets.
#
# The v2 "Clinical Instrument" dashboard is the primary UI, mounted at /dashboard.
# The original v1 dashboard is retained as a fallback at /dashboard-v1 so there is
# always a known-good UI to fall back to during the demo. Both reference their
# css/ and js/ via relative paths, so each html=True mount resolves assets within
# its own subtree (no shared top-level /css or /js mount needed).
_base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
dashboard_v2_dir = os.path.join(_base_dir, "dashboard-v2")
dashboard_v1_dir = os.path.join(_base_dir, "dashboard")

# Primary dashboard (v2 if present, otherwise fall back to v1).
dashboard_dir = dashboard_v2_dir if os.path.isdir(dashboard_v2_dir) else dashboard_v1_dir
if os.path.isdir(dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")

# v1 fallback, always available at its own path when it exists.
if os.path.isdir(dashboard_v1_dir):
    app.mount("/dashboard-v1", StaticFiles(directory=dashboard_v1_dir, html=True), name="dashboard-v1")

# Serve ONLY the specific validation plots the dashboard needs — via an explicit
# allow-list, not a blanket StaticFiles mount of data/processed (which would also
# expose the full labeled training CSVs, split manifest and model files).
_assets_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
_PUBLIC_ASSETS = {
    "clarke_error_grid_real.png",
    "shap_summary_plot.png",
}


@app.get("/assets/{filename}")
def get_asset(filename: str):
    """Return a whitelisted validation plot (image) from data/processed."""
    if filename not in _PUBLIC_ASSETS:
        raise HTTPException(status_code=404, detail="Asset not found")
    path = os.path.join(_assets_dir, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path)


@app.get("/")
def root():
    """Redirect to dashboard or return API status."""
    dashboard_index = os.path.join(dashboard_dir, "index.html")
    if os.path.exists(dashboard_index):
        return RedirectResponse(url="/dashboard/")
    return {"message": "Cardiometabolic Digital Twin API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    # Change to project root so relative paths work
    os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
    # Bind to the platform-provided PORT (Render/Railway/Heroku inject it);
    # default to 8000 for local development.
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
