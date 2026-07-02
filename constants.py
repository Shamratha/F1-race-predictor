"""Static reference data and small helpers shared across services."""
import math

import numpy as np

# F1 points for a top-10 finish -> turns a predicted finishing order into a
# predicted constructors' championship for a single race.
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

# (latitude, longitude) of each circuit — used to fetch live weather.
TRACK_COORDS = {
    "Bahrain Grand Prix": (26.0325, 50.5106),
    "Saudi Arabian Grand Prix": (21.6319, 39.1044),
    "Australian Grand Prix": (-37.8497, 144.968),
    "Emilia Romagna Grand Prix": (44.3439, 11.7167),
    "Miami Grand Prix": (25.9581, -80.2389),
    "Monaco Grand Prix": (43.7347, 7.4206),
    "Spanish Grand Prix": (41.57, 2.2611),
    "Canadian Grand Prix": (45.5, -73.5228),
    "Austrian Grand Prix": (47.2197, 14.7647),
    "Styrian Grand Prix": (47.2197, 14.7647),
    "British Grand Prix": (52.0786, -1.0169),
    "Hungarian Grand Prix": (47.5789, 19.2486),
    "Belgian Grand Prix": (50.4372, 5.9714),
    "Dutch Grand Prix": (52.3888, 4.5409),
    "Italian Grand Prix": (45.6156, 9.2811),
    "Azerbaijan Grand Prix": (40.3725, 49.8533),
    "Singapore Grand Prix": (1.2914, 103.864),
    "Japanese Grand Prix": (34.8431, 136.541),
    "Qatar Grand Prix": (25.49, 51.4542),
    "United States Grand Prix": (30.1328, -97.6411),
    "Mexico City Grand Prix": (19.4042, -99.0907),
    "Mexican Grand Prix": (19.4042, -99.0907),
    "São Paulo Grand Prix": (-23.7036, -46.6997),
    "Sao Paulo Grand Prix": (-23.7036, -46.6997),
    "Las Vegas Grand Prix": (36.1147, -115.173),
    "Abu Dhabi Grand Prix": (24.4672, 54.6031),
    "Chinese Grand Prix": (31.3389, 121.22),
    "French Grand Prix": (43.2506, 5.7917),
    "Portuguese Grand Prix": (37.2306, -8.6267),
    "Turkish Grand Prix": (40.9517, 29.405),
    "Russian Grand Prix": (43.4057, 39.9578),
    "Sakhir Grand Prix": (26.0325, 50.5106),
}

FALLBACK_COLOR = "#9AA0A6"


def team_color(team: str) -> str:
    return TEAM_COLORS.get(team, FALLBACK_COLOR)


def driver_name(abbr: str) -> str:
    return DRIVER_NAMES.get(abbr, abbr)


def clean(v):
    """JSON-safe: NaN/inf -> None, numpy scalars -> Python scalars, floats rounded."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    return v
