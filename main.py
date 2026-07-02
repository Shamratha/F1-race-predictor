"""FastAPI backend for the F1 Podium Predictor.

Wraps the trained XGBoost model behind a small JSON API and serves the custom
dashboard in web/. No FastF1 / SHAP imports at runtime -> lean enough for a
free-tier host.

Endpoints:
    GET /                -> the dashboard (web/index.html)
    GET /api/meta        -> model metrics, seasons, teams
    GET /api/races       -> available races grouped by season
    GET /api/predict     -> per-driver podium predictions + constructor standings
                            query: season, circuit, rain, track_temp, wind,
                                   penalty_driver, penalty_grid
"""
import os
import json
import math

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "models")
WEB = os.path.join(ROOT, "web")

# F1 points for a top-10 finish -> used to turn predicted order into a
# predicted constructors' championship for the selected race.
POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6", "Ferrari": "#E8002D", "Mercedes": "#27F4D2",
    "McLaren": "#FF8000", "Aston Martin": "#229971", "Alpine": "#0093CC",
    "Williams": "#64C4FF", "RB": "#6692FF", "AlphaTauri": "#5E8FAA",
    "Kick Sauber": "#52E252", "Alfa Romeo": "#C92D4B", "Alfa Romeo Racing": "#C92D4B",
    "Haas F1 Team": "#B6BABD",
}

DRIVER_NAMES = {
    "VER": "Max Verstappen", "PER": "Sergio Perez", "LEC": "Charles Leclerc",
    "SAI": "Carlos Sainz", "HAM": "Lewis Hamilton", "RUS": "George Russell",
    "NOR": "Lando Norris", "PIA": "Oscar Piastri", "ALO": "Fernando Alonso",
    "STR": "Lance Stroll", "OCO": "Esteban Ocon", "GAS": "Pierre Gasly",
    "BOT": "Valtteri Bottas", "ZHO": "Zhou Guanyu", "TSU": "Yuki Tsunoda",
    "RIC": "Daniel Ricciardo", "MAG": "Kevin Magnussen", "HUL": "Nico Hulkenberg",
    "ALB": "Alexander Albon", "SAR": "Logan Sargeant", "COL": "Franco Colapinto",
    "BEA": "Oliver Bearman", "LAW": "Liam Lawson", "MSC": "Mick Schumacher",
    "VET": "Sebastian Vettel", "LAT": "Nicholas Latifi", "DEV": "Nyck de Vries",
    "GIO": "Antonio Giovinazzi", "RAI": "Kimi Raikkonen", "MAZ": "Nikita Mazepin",
}

app = FastAPI(title="F1 Podium Predictor")

model = joblib.load(os.path.join(MODELS, "model.joblib"))
with open(os.path.join(MODELS, "metadata.json")) as f:
    META = json.load(f)
DF = pd.read_csv(os.path.join(ROOT, "data", "dataset.csv"))
FEATURES = META["numeric"] + META["categorical"]


def clean(v):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(v, (np.floating, float)):
        return None if (v is None or math.isnan(v) or math.isinf(v)) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def team_color(team):
    return TEAM_COLORS.get(team, "#9AA0A6")


@app.get("/api/meta")
def api_meta():
    return {
        "metrics": META["metrics"],
        "seasons": META["seasons"],
        "n_rows": META["n_rows"],
    }


@app.get("/api/races")
def api_races():
    races = {}
    for season in sorted(DF["season"].unique(), reverse=True):
        circuits = DF[DF["season"] == season]["circuit"].tolist()
        # keep original chronological order, de-duplicated
        seen, ordered = set(), []
        for c in circuits:
            if c not in seen:
                seen.add(c); ordered.append(c)
        races[int(season)] = ordered
    return {"races": races}


@app.get("/api/predict")
def api_predict(
    season: int,
    circuit: str,
    rain: str = Query("actual"),          # "actual" | "0" | "1"
    track_temp: float | None = None,
    wind: float | None = None,
    penalty_driver: str = "",
    penalty_grid: int = 15,
):
    race = DF[(DF["season"] == season) & (DF["circuit"] == circuit)].copy()
    if race.empty:
        return JSONResponse({"error": "race not found"}, status_code=404)
    race = race.reset_index(drop=True)

    actual_rain = int(bool(race["rainfall"].iloc[0]))
    scenario = race.copy()

    # --- apply what-if overrides ---
    changed = False
    if rain in ("0", "1"):
        scenario["rainfall"] = int(rain)
        changed = changed or (int(rain) != actual_rain)
    if track_temp is not None:
        scenario["track_temp"] = track_temp
    if wind is not None:
        scenario["wind_speed"] = wind
    if penalty_driver and penalty_driver in scenario["driver"].values:
        scenario.loc[scenario["driver"] == penalty_driver, "grid_position"] = float(penalty_grid)
        changed = True

    # --- predict ---
    scenario["podium_proba"] = model.predict_proba(scenario[FEATURES])[:, 1]
    scenario = scenario.sort_values("podium_proba", ascending=False).reset_index(drop=True)
    scenario["pred_rank"] = scenario.index + 1

    # predicted championship points for this race (top 10 predicted finishers)
    scenario["pred_points"] = [POINTS[i] if i < len(POINTS) else 0 for i in range(len(scenario))]

    actual_top3 = (
        race.sort_values("finish_position")["driver"].head(3).tolist()
        if race["finish_position"].notna().any() else []
    )
    predicted_top3 = scenario.head(3)["driver"].tolist()

    drivers = []
    for _, r in scenario.iterrows():
        drivers.append({
            "pred_rank": int(r["pred_rank"]),
            "driver": r["driver"],
            "name": DRIVER_NAMES.get(r["driver"], r["driver"]),
            "team": r["team"],
            "team_color": team_color(r["team"]),
            "grid": clean(r["grid_position"]),
            "quali_gap": clean(r["quali_gap_to_pole_s"]),
            "podium_proba": clean(r["podium_proba"]),
            "pred_points": int(r["pred_points"]),
            "actual_finish": clean(r["finish_position"]),
            "podium_hit": r["driver"] in actual_top3,
        })

    # --- constructor standings for this race (sum of predicted points) ---
    cons = (
        scenario.groupby("team")["pred_points"].sum()
        .sort_values(ascending=False).reset_index()
    )
    constructors = [
        {"team": row["team"], "team_color": team_color(row["team"]),
         "pred_points": int(row["pred_points"])}
        for _, row in cons.iterrows()
    ]

    return {
        "race": {"season": season, "circuit": circuit, "changed": changed},
        "weather": {
            "air_temp": clean(scenario["air_temp"].iloc[0]),
            "track_temp": clean(scenario["track_temp"].iloc[0]),
            "humidity": clean(scenario["humidity"].iloc[0]),
            "wind_speed": clean(scenario["wind_speed"].iloc[0]),
            "rainfall": int(scenario["rainfall"].iloc[0]),
            "actual_rain": actual_rain,
        },
        "drivers": drivers,
        "predicted_podium": predicted_top3,
        "actual_podium": actual_top3,
        "hits": len(set(predicted_top3) & set(actual_top3)),
        "constructors": constructors,
    }


# --- serve the dashboard ---
@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


@app.get("/shap_summary.png")
def shap_png():
    return FileResponse(os.path.join(MODELS, "shap_summary.png"))


app.mount("/", StaticFiles(directory=WEB), name="static")
