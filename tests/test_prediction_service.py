"""Tests for the PredictionService business logic."""
import numpy as np
import pytest

from constants import POINTS


def test_predict_shape_and_probabilities(prediction_service, a_race):
    season, circuit, race = a_race
    out = prediction_service.predict(race, season=season, circuit=circuit)

    assert set(out) >= {"race", "weather", "drivers", "predicted_podium",
                        "actual_podium", "hits", "constructors"}
    assert len(out["drivers"]) == len(race)
    assert len(out["predicted_podium"]) == 3

    # every probability is a valid [0, 1] number
    for d in out["drivers"]:
        assert 0.0 <= d["podium_proba"] <= 1.0

    # ranks are 1..N and sorted by probability (descending)
    ranks = [d["pred_rank"] for d in out["drivers"]]
    assert ranks == list(range(1, len(race) + 1))
    probs = [d["podium_proba"] for d in out["drivers"]]
    assert probs == sorted(probs, reverse=True)


def test_constructor_points_total_matches_scoring(prediction_service, a_race):
    season, circuit, race = a_race
    out = prediction_service.predict(race, season=season, circuit=circuit)
    total = sum(c["pred_points"] for c in out["constructors"])
    # top-10 predicted finishers score POINTS; everyone else zero
    assert total == sum(POINTS[:min(len(race), len(POINTS))])


def test_grid_penalty_lowers_probability_and_flags_change(prediction_service, a_race):
    season, circuit, race = a_race
    base = prediction_service.predict(race, season=season, circuit=circuit)
    top_driver = base["predicted_podium"][0]
    base_prob = next(d["podium_proba"] for d in base["drivers"] if d["driver"] == top_driver)

    penalised = prediction_service.predict(
        race, season=season, circuit=circuit,
        penalty_driver=top_driver, penalty_grid=20,
    )
    new_prob = next(d["podium_proba"] for d in penalised["drivers"] if d["driver"] == top_driver)

    assert penalised["race"]["changed"] is True
    assert new_prob < base_prob            # sending the favourite to the back hurts them


def test_rain_override_flags_change_when_different(prediction_service, a_race):
    season, circuit, race = a_race
    actual_rain = int(bool(race["rainfall"].iloc[0]))
    opposite = "0" if actual_rain == 1 else "1"
    out = prediction_service.predict(race, season=season, circuit=circuit, rain=opposite)
    assert out["race"]["changed"] is True
    assert out["weather"]["rainfall"] == int(opposite)


def test_missing_weather_still_predicts(prediction_service, a_race):
    """XGBoost handles NaN natively — a race with no weather must not crash."""
    season, circuit, race = a_race
    broken = race.copy()
    for col in ["air_temp", "track_temp", "humidity", "wind_speed"]:
        broken[col] = np.nan
    out = prediction_service.predict(broken, season=season, circuit=circuit)
    assert len(out["drivers"]) == len(race)
    assert out["weather"]["air_temp"] is None      # cleaned NaN -> None
