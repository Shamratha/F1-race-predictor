"""FastAPI backend for the F1 Podium Predictor.

Thin routing layer: it wires together the service classes and serves the custom
dashboard in web/. No FastF1 / SHAP imports at runtime -> lean enough for a
free-tier host.

Endpoints:
    GET /                   -> the dashboard (web/index.html)
    GET /api/meta           -> model metrics, seasons
    GET /api/races          -> available races grouped by season
    GET /api/predict        -> per-driver podium predictions + constructor standings
    GET /api/weather/live   -> current real-world conditions at a circuit (opt-in)
"""
import os

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from services.f1_data_service import F1DataService
from services.prediction_service import PredictionService
from services.weather_service import WeatherService

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "models")
WEB = os.path.join(ROOT, "web")

app = FastAPI(title="F1 Podium Predictor")

data_service = F1DataService()
prediction_service = PredictionService()
weather_service = WeatherService()


@app.get("/api/meta")
def api_meta():
    return prediction_service.meta_summary()


@app.get("/api/races")
def api_races():
    return {"races": data_service.races_by_season()}


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
    race = data_service.get_race(season, circuit)
    if race is None:
        return JSONResponse({"error": "race not found"}, status_code=404)
    return prediction_service.predict(
        race, season=season, circuit=circuit, rain=rain, track_temp=track_temp,
        wind=wind, penalty_driver=penalty_driver, penalty_grid=penalty_grid,
    )


@app.get("/api/weather/live")
def api_weather_live(circuit: str):
    return weather_service.get_live_weather(circuit)


# --- serve the dashboard ---
@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


@app.get("/shap_summary.png")
def shap_png():
    return FileResponse(os.path.join(MODELS, "shap_summary.png"))


app.mount("/", StaticFiles(directory=WEB), name="static")
