"""Tests for the live-weather service (no real network calls)."""
from services.weather_service import WeatherService


def test_disabled_without_api_key():
    svc = WeatherService(api_key="")
    assert svc.enabled is False
    out = svc.get_live_weather("Monaco Grand Prix")
    assert out["source"] == "unavailable"
    assert "api_key" in out["reason"].lower() or "key" in out["reason"].lower()


def test_unknown_track_is_unavailable():
    svc = WeatherService(api_key="dummy-key")
    out = svc.get_live_weather("Nonexistent Grand Prix")
    assert out["source"] == "unavailable"
    assert "track" in out["reason"].lower()


def test_map_dry_payload():
    payload = {
        "name": "Monte Carlo",
        "main": {"temp": 24.0, "humidity": 55},
        "wind": {"speed": 3.2},
        "weather": [{"main": "Clear"}],
    }
    out = WeatherService._map(payload, "Monaco Grand Prix")
    assert out["source"] == "live"
    assert out["rainfall"] == 0
    assert out["air_temp"] == 24.0
    assert out["track_temp"] == 36.0          # dry -> air + 12
    assert out["humidity"] == 55
    assert out["wind_speed"] == 3.2


def test_map_wet_payload():
    payload = {
        "name": "Spa",
        "main": {"temp": 15.0, "humidity": 90},
        "wind": {"speed": 5.0},
        "weather": [{"main": "Rain"}],
    }
    out = WeatherService._map(payload, "Belgian Grand Prix")
    assert out["rainfall"] == 1
    assert out["track_temp"] == 17.0          # wet -> air + 2


def test_map_defaults_when_fields_missing():
    out = WeatherService._map({}, "Bahrain Grand Prix")
    assert out["source"] == "live"
    assert out["rainfall"] == 0
    assert isinstance(out["air_temp"], float)
