
import io
import os
import math
import time
import random
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import joblib
import requests
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from skyfield.api import load, wgs84
from skyfield.iokit import parse_tle_file


# -----------------------------
# PROJECT CONFIGURATION
# -----------------------------

BASE = Path(__file__).resolve().parents[1]

DATA = BASE / "data"
MODELS = BASE / "ml" / "models"

DATA.mkdir(exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA / "satellite.db"

CELESTRAK_URL = (
    "https://celestrak.org/NORAD/elements/gp.php"
    "?GROUP=stations&FORMAT=tle"
)

app = FastAPI(
    title="Satellite Link AI",
    version="1.0.0",
    description="Orbital tracking and simulated link monitoring"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)

ts = load.timescale()

satellites = {}
last_fetch = 0
fetch_error = None

DEFAULT_THRESHOLDS = {
    "latency": 180,
    "packet_loss": 5,
    "jitter": 30,
    "snr": 10
}

state = {
    "running": False,
    "scenario": "normal",
    "seed": 42,
    "step": 0,
    "satellite_id": None,
    "thresholds": DEFAULT_THRESHOLDS.copy(),
    "latest": None
}


# -----------------------------
# DATABASE
# -----------------------------

def connection():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    with connection() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                scenario TEXT NOT NULL,
                state TEXT NOT NULL,
                latency REAL,
                packet_loss REAL,
                jitter REAL,
                snr REAL,
                throughput REAL,
                quality REAL,
                violations TEXT
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value REAL NOT NULL
            )
        """)

        for key, value in DEFAULT_THRESHOLDS.items():
            db.execute(
                "INSERT OR IGNORE INTO settings VALUES (?, ?)",
                (key, value)
            )

        db.commit()


init_db()

with connection() as db:
    for item in db.execute("SELECT * FROM settings"):
        state["thresholds"][item["key"]] = item["value"]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def log_event(metrics):
    with connection() as db:
        db.execute("""
            INSERT INTO events (
                timestamp, scenario, state,
                latency, packet_loss, jitter, snr,
                throughput, quality, violations
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics["timestamp"],
            metrics["scenario"],
            metrics["state"],
            metrics["latency"],
            metrics["packet_loss"],
            metrics["jitter"],
            metrics["snr"],
            metrics["throughput"],
            metrics["quality"],
            ", ".join(metrics["violations"])
        ))
        db.commit()


# -----------------------------
# SATELLITE TRACKING
# -----------------------------


def refresh_satellites(force=False):
    global satellites, last_fetch, fetch_error

    if not force and satellites and time.time() - last_fetch < 3600:
        return

    try:
        response = requests.get(
            CELESTRAK_URL,
            timeout=20,
            headers={"User-Agent": "CollegeSatelliteResearch/1.0"}
        )

        response.raise_for_status()

        parsed = list(parse_tle_file(
            io.BytesIO(response.content),
            ts=ts
        ))

        if not parsed:
            raise ValueError("No valid orbital elements returned")

        satellites = {
            str(s.model.satnum): s
            for s in parsed
        }

        last_fetch = time.time()
        fetch_error = None

    except Exception as exc:
        fetch_error = str(exc)

        if not satellites:
            raise HTTPException(
                status_code=503,
                detail="Orbital data unavailable: " + fetch_error
            )



def satellite_position(sat):
    current = ts.now()

    geocentric = sat.at(current)
    subpoint = wgs84.subpoint(geocentric)

    epoch = sat.epoch.utc_datetime()
    age_hours = abs(
        (datetime.now(timezone.utc) - epoch).total_seconds()
    ) / 3600

    return {
        "id": str(sat.model.satnum),
        "name": sat.name,
        "latitude": round(subpoint.latitude.degrees, 5),
        "longitude": round(subpoint.longitude.degrees, 5),
        "altitude_km": round(subpoint.elevation.km, 3),
        "epoch": epoch.isoformat(),
        "orbital_age_hours": round(age_hours, 2),
        "stale": age_hours > 168,
        "calculated_at": now_iso(),
        "source": "CelesTrak orbital elements; Skyfield calculation"
    }


@app.get("/api/health")
def health():
    return {
        "status": "running",
        "time": now_iso(),
        "mode": "research prototype"
    }


@app.get("/api/satellites")
def get_satellites():
    refresh_satellites()

    return {
        "source": CELESTRAK_URL,
        "last_fetch": datetime.fromtimestamp(
            last_fetch, timezone.utc
        ).isoformat(),
        "fetch_error": fetch_error,
        "satellites": [
            satellite_position(s)
            for s in satellites.values()
        ]
    }


@app.get("/api/satellites/{satellite_id}")
def get_satellite(satellite_id: str):
    refresh_satellites()

    sat = satellites.get(satellite_id)

    if sat is None:
        raise HTTPException(404, "Satellite not found")

    return satellite_position(sat)


# -----------------------------
# GROUND STATION
# -----------------------------

class VisibilityRequest(BaseModel):
    satellite_id: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    altitude_m: float = Field(default=0, ge=-500, le=10000)
    elevation_mask: float = Field(default=10, ge=0, le=90)


def calculate_visibility(
    sat, latitude, longitude, altitude_m, elevation_mask
):
    station = wgs84.latlon(
        latitude,
        longitude,
        elevation_m=altitude_m
    )

    difference = sat - station
    topocentric = difference.at(ts.now())

    altitude, azimuth, distance = topocentric.altaz()

    elevation = altitude.degrees

    return {
        "elevation_deg": round(elevation, 3),
        "azimuth_deg": round(azimuth.degrees, 3),
        "range_km": round(distance.km, 3),
        "geometric_line_of_sight": elevation > 0,
        "above_elevation_mask": elevation >= elevation_mask,
        "calculated_at": now_iso(),
        "note": (
            "Geometric visibility does not establish "
            "an operational communication link."
        )
    }


@app.post("/api/visibility")
def visibility(request: VisibilityRequest):
    refresh_satellites()

    sat = satellites.get(request.satellite_id)

    if sat is None:
        raise HTTPException(404, "Satellite not found")

    return calculate_visibility(
        sat,
        request.latitude,
        request.longitude,
        request.altitude_m,
        request.elevation_mask
    )


# -----------------------------
# LINK SIMULATOR
# -----------------------------

SCENARIOS = [
    "normal",
    "degradation",
    "high_latency",
    "packet_loss",
    "interference",
    "outage",
    "recovery",
    "handover"
]


class StartRequest(BaseModel):
    seed: int = Field(default=42, ge=0, le=1000000)
    satellite_id: str | None = None


class ScenarioRequest(BaseModel):
    scenario: str


class ThresholdRequest(BaseModel):
    latency: float = Field(gt=0, le=10000)
    packet_loss: float = Field(ge=0, le=100)
    jitter: float = Field(ge=0, le=10000)
    snr: float = Field(ge=-50, le=100)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def evaluate_thresholds(metrics, thresholds):
    violations = []

    if metrics["latency"] > thresholds["latency"]:
        violations.append("Latency threshold exceeded")

    if metrics["packet_loss"] > thresholds["packet_loss"]:
        violations.append("Packet-loss threshold exceeded")

    if metrics["jitter"] > thresholds["jitter"]:
        violations.append("Jitter threshold exceeded")

    if metrics["snr"] < thresholds["snr"]:
        violations.append("SNR below minimum")

    return violations


def generate_metrics(scenario, seed, step, thresholds):
    rng = random.Random(seed + step)

    profiles = {
        "normal": (65, 0.5, 5, 22, 85),
        "degradation": (160, 7, 25, 11, 48),
        "high_latency": (330, 2, 65, 19, 65),
        "packet_loss": (110, 28, 18, 16, 40),
        "interference": (190, 35, 45, 3, 22),
        "outage": (1000, 100, 0, -10, 0),
        "recovery": (85, 1, 8, 20, 78),
        "handover": (240, 12, 35, 12, 42)
    }

    if scenario not in profiles:
        raise ValueError("Unknown scenario")

    latency, loss, jitter, snr, throughput = profiles[scenario]

    if scenario != "outage":
        latency = clamp(latency + rng.gauss(0, 8), 1, 10000)
        loss = clamp(loss + rng.gauss(0, 1.2), 0, 100)
        jitter = clamp(jitter + rng.gauss(0, 3), 0, 10000)
        snr = clamp(snr + rng.gauss(0, 1.5), -50, 100)
        throughput = clamp(
            throughput + rng.gauss(0, 4), 0, 1000
        )

    metrics = {
        "timestamp": now_iso(),
        "mode": "SIMULATED SATELLITE LINK",
        "scenario": scenario,
        "latency": round(latency, 2),
        "packet_loss": round(loss, 2),
        "jitter": round(jitter, 2),
        "snr": round(snr, 2),
        "throughput": round(throughput, 2)
    }

    violations = evaluate_thresholds(metrics, thresholds)

    quality = 100 - (
        min(loss, 100) * 0.45
        + min(latency / 10, 35)
        + min(jitter / 4, 15)
        + max(0, 15 - snr)
    )

    quality = round(clamp(quality, 0, 100), 2)

    if scenario == "outage":
        quality = 0
        link_state = "OUTAGE"
    elif violations:
        link_state = "DEGRADED"
    else:
        link_state = "HEALTHY"

    metrics.update({
        "quality": quality,
        "state": link_state,
        "violations": violations
    })

    return metrics


@app.post("/api/simulation/start")
def start_simulation(request: StartRequest):
    if request.satellite_id is not None:
        refresh_satellites()

        if request.satellite_id not in satellites:
            raise HTTPException(404, "Satellite not found")

    state.update({
        "running": True,
        "scenario": "normal",
        "seed": request.seed,
        "step": 0,
        "satellite_id": request.satellite_id,
        "latest": None
    })

    return {
        "message": "Simulation started",
        "seed": request.seed,
        "mode": "simulation"
    }


@app.post("/api/simulation/scenario")
def change_scenario(request: ScenarioRequest):
    if not state["running"]:
        raise HTTPException(409, "Start the simulation first")

    if request.scenario not in SCENARIOS:
        raise HTTPException(422, "Invalid scenario")

    state["scenario"] = request.scenario

    return {
        "scenario": request.scenario,
        "message": "Scenario updated"
    }


@app.post("/api/simulation/stop")
def stop_simulation():
    state["running"] = False

    return {"message": "Simulation stopped"}


@app.get("/api/monitoring/metrics")
def monitoring_metrics():
    if not state["running"]:
        return {
            "running": False,
            "latest": state["latest"]
        }

    metrics = generate_metrics(
        state["scenario"],
        state["seed"],
        state["step"],
        state["thresholds"]
    )

    state["step"] += 1
    state["latest"] = metrics

    log_event(metrics)

    return {
        "running": True,
        "satellite_id": state["satellite_id"],
        "latest": metrics
    }


@app.get("/api/thresholds")
def get_thresholds():
    return state["thresholds"]


@app.post("/api/thresholds")
def save_thresholds(request: ThresholdRequest):
    values = request.model_dump()

    state["thresholds"] = values

    with connection() as db:
        for key, value in values.items():
            db.execute(
                "INSERT OR REPLACE INTO settings VALUES (?, ?)",
                (key, value)
            )

        db.commit()

    return values


@app.get("/api/events")
def events(limit: int = 100):
    limit = clamp(limit, 1, 500)

    with connection() as db:
        rows = db.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()

    return [dict(row) for row in rows]


# -----------------------------
# MACHINE LEARNING
# -----------------------------

JAMMING_MODEL = MODELS / "jamming.joblib"
LINK_MODEL = MODELS / "link.joblib"


class JammingInput(BaseModel):
    latency: float = Field(ge=0, le=10000)
    packet_loss: float = Field(ge=0, le=100)
    jitter: float = Field(ge=0, le=10000)
    snr: float = Field(ge=-50, le=100)


class LinkInput(BaseModel):
    latency: float = Field(ge=0, le=10000)
    packet_loss: float = Field(ge=0, le=100)
    jitter: float = Field(ge=0, le=10000)
    snr: float = Field(ge=-50, le=100)


@app.post("/api/ml/jamming/predict")
def predict_interference(data: JammingInput):
    if not JAMMING_MODEL.exists():
        raise HTTPException(503, "Train the ML model first")

    model = joblib.load(JAMMING_MODEL)

    x = np.array([[
        data.latency,
        data.packet_loss,
        data.jitter,
        data.snr
    ]])

    prediction = int(model.predict(x)[0])
    confidence = float(model.predict_proba(x)[0].max())

    return {
        "prediction": prediction,
        "label": (
            "Simulated interference pattern"
            if prediction else "Normal-like pattern"
        ),
        "confidence": round(confidence, 4),
        "warning": (
            "Synthetic-data classification only. "
            "This is not proof of real RF jamming."
        )
    }


@app.post("/api/ml/link/predict")
def predict_link(data: LinkInput):
    if not LINK_MODEL.exists():
        raise HTTPException(503, "Train the ML model first")

    model = joblib.load(LINK_MODEL)

    x = np.array([[
        data.latency,
        data.packet_loss,
        data.jitter,
        data.snr
    ]])

    result = float(model.predict(x)[0])

    return {
        "predicted_next_latency_ms": round(max(0, result), 2),
        "source": "Synthetic simulation training data",
        "warning": "Experimental prediction, not a live satellite forecast"
    }


@app.get("/api/ml/evaluation")
def ml_evaluation():
    import json

    path = MODELS / "evaluation.json"

    if not path.exists():
        raise HTTPException(503, "Run ML training first")

    return json.loads(path.read_text())


# -----------------------------
# OPTIONAL ORDINARY INTERNET PROBE
# -----------------------------

@app.get("/api/network/probe")
def network_probe():
    """
    Measures a local HTTP request to this backend.
    It is not a satellite measurement.
    """

    start = time.perf_counter()

    response = {
        "status": "ok",
        "timestamp": now_iso()
    }

    elapsed = (time.perf_counter() - start) * 1000

    return {
        **response,
        "server_processing_ms": round(elapsed, 3),
        "note": (
            "Server-side processing time only. "
            "Not satellite latency, SNR or packet loss."
        )
    }
