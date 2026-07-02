"""Data-access layer: loads the engineered dataset and answers race queries."""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "data", "dataset.csv")


class F1DataService:
    """Owns the driver-races dataset and simple lookups over it."""

    def __init__(self, data_path: str = DEFAULT_DATA):
        self.df = pd.read_csv(data_path)

    def races_by_season(self) -> dict:
        """{season: [circuit, ...]} in chronological order, newest season first."""
        out = {}
        for season in sorted(self.df["season"].unique(), reverse=True):
            circuits = self.df[self.df["season"] == season]["circuit"].tolist()
            seen, ordered = set(), []
            for c in circuits:
                if c not in seen:
                    seen.add(c)
                    ordered.append(c)
            out[int(season)] = ordered
        return out

    def get_race(self, season: int, circuit: str) -> pd.DataFrame | None:
        """All driver rows for one race, or None if that race isn't in the data."""
        race = self.df[(self.df["season"] == season) & (self.df["circuit"] == circuit)]
        if race.empty:
            return None
        return race.copy().reset_index(drop=True)
