"""Prediction layer: turns a race's driver rows into podium predictions."""
import os
import json

import joblib
import pandas as pd

from constants import POINTS, team_color, driver_name, clean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")


class PredictionService:
    """Owns the trained model and produces the JSON payload the dashboard needs."""

    def __init__(self, models_dir: str = MODELS):
        self.model = joblib.load(os.path.join(models_dir, "model.joblib"))
        with open(os.path.join(models_dir, "metadata.json")) as f:
            self.meta = json.load(f)
        self.features = self.meta["numeric"] + self.meta["categorical"]

    def meta_summary(self) -> dict:
        return {
            "metrics": self.meta["metrics"],
            "seasons": self.meta["seasons"],
            "n_rows": self.meta["n_rows"],
        }

    def predict(
        self,
        race: pd.DataFrame,
        season: int,
        circuit: str,
        rain: str = "actual",          # "actual" | "0" | "1"
        track_temp: float | None = None,
        wind: float | None = None,
        penalty_driver: str = "",
        penalty_grid: int = 15,
    ) -> dict:
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

        # --- predict + rank ---
        scenario["podium_proba"] = self.model.predict_proba(scenario[self.features])[:, 1]
        scenario = scenario.sort_values("podium_proba", ascending=False).reset_index(drop=True)
        scenario["pred_rank"] = scenario.index + 1
        scenario["pred_points"] = [POINTS[i] if i < len(POINTS) else 0 for i in range(len(scenario))]

        actual_top3 = (
            race.sort_values("finish_position")["driver"].head(3).tolist()
            if race["finish_position"].notna().any() else []
        )
        predicted_top3 = scenario.head(3)["driver"].tolist()

        drivers = [{
            "pred_rank": int(r["pred_rank"]),
            "driver": r["driver"],
            "name": driver_name(r["driver"]),
            "team": r["team"],
            "team_color": team_color(r["team"]),
            "grid": clean(r["grid_position"]),
            "quali_gap": clean(r["quali_gap_to_pole_s"]),
            "podium_proba": clean(r["podium_proba"]),
            "pred_points": int(r["pred_points"]),
            "actual_finish": clean(r["finish_position"]),
            "podium_hit": r["driver"] in actual_top3,
        } for _, r in scenario.iterrows()]

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
