"""Live-weather layer: current conditions at a circuit via OpenWeatherMap.

Free tier, opt-in: set the OPENWEATHER_API_KEY environment variable. With no key
(or if the request fails / the track is unknown) every method degrades gracefully
to {"source": "unavailable"} so the app keeps working on historical data alone.
"""
import os

import requests

from constants import TRACK_COORDS

API_URL = "https://api.openweathermap.org/data/2.5/weather"
WET_CONDITIONS = {"Rain", "Drizzle", "Thunderstorm", "Snow"}


class WeatherService:
    def __init__(self, api_key: str | None = None, timeout: float = 6.0):
        self.api_key = api_key if api_key is not None else os.environ.get("OPENWEATHER_API_KEY", "")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def get_live_weather(self, circuit: str) -> dict:
        """Current conditions mapped to the model's weather features."""
        coords = TRACK_COORDS.get(circuit)
        if coords is None:
            return {"source": "unavailable", "reason": "unknown track location"}
        if not self.enabled:
            return {"source": "unavailable", "reason": "no OPENWEATHER_API_KEY set"}

        lat, lon = coords
        try:
            resp = requests.get(
                API_URL,
                params={"lat": lat, "lon": lon, "units": "metric", "appid": self.api_key},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return {"source": "unavailable", "reason": f"api status {resp.status_code}"}
            data = resp.json()
        except requests.RequestException as e:
            return {"source": "unavailable", "reason": f"request failed: {type(e).__name__}"}

        return self._map(data, circuit)

    @staticmethod
    def _map(data: dict, circuit: str) -> dict:
        """Translate an OpenWeatherMap payload into our feature names."""
        main = data.get("main", {})
        wind = data.get("wind", {})
        condition = (data.get("weather") or [{}])[0].get("main", "Clear")
        rainfall = 1 if (condition in WET_CONDITIONS or "rain" in data) else 0

        air_temp = float(main.get("temp", 20.0))
        # F1 track temperature runs well above air temp when dry, only slightly when wet.
        track_temp = round(air_temp + (2.0 if rainfall else 12.0), 1)

        return {
            "source": "live",
            "circuit": circuit,
            "location": data.get("name", circuit),
            "condition": condition,
            "air_temp": round(air_temp, 1),
            "track_temp": track_temp,
            "humidity": float(main.get("humidity", 50.0)),
            "wind_speed": round(float(wind.get("speed", 0.0)), 1),
            "rainfall": rainfall,
        }
