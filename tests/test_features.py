"""Unit tests for the pure feature-engineering functions."""
import numpy as np
import pandas as pd
import pytest

from features import is_dnf, best_quali_time, add_form_features


# ---------------- is_dnf ----------------
@pytest.mark.parametrize("status,expected", [
    ("Finished", 0),
    ("+1 Lap", 0),
    ("+2 Laps", 0),
    ("Accident", 1),
    ("Engine", 1),
    ("Retired", 1),
    ("Collision", 1),
])
def test_is_dnf(status, expected):
    assert is_dnf(status) == expected


# ---------------- best_quali_time ----------------
def test_best_quali_time_picks_minimum():
    row = {"Q1": pd.Timedelta(seconds=91.2),
           "Q2": pd.Timedelta(seconds=90.5),
           "Q3": pd.Timedelta(seconds=90.8)}
    assert best_quali_time(row) == pytest.approx(90.5)


def test_best_quali_time_handles_missing_sessions():
    # driver knocked out in Q1 -> only Q1 set
    row = {"Q1": pd.Timedelta(seconds=92.0), "Q2": pd.NaT, "Q3": pd.NaT}
    assert best_quali_time(row) == pytest.approx(92.0)


def test_best_quali_time_all_missing_is_nan():
    row = {"Q1": pd.NaT, "Q2": pd.NaT, "Q3": pd.NaT}
    assert np.isnan(best_quali_time(row))


# ---------------- add_form_features (no leakage) ----------------
def _toy_df():
    # Driver A finishes 1,2,3 across three races; B finishes 10,9,8.
    rows = []
    for i, date in enumerate(["2023-01-01", "2023-02-01", "2023-03-01"]):
        rows.append(dict(season=2023, date=date, circuit=f"R{i}", driver="A",
                         team="Alpha", finish_position=i + 1, podium=1,
                         grid_position=i + 1, status="Finished", points=25 - i))
        rows.append(dict(season=2023, date=date, circuit=f"R{i}", driver="B",
                         team="Beta", finish_position=10 - i, podium=0,
                         grid_position=10 - i, status="Finished", points=1))
    return pd.DataFrame(rows)


def test_form_first_race_has_no_history():
    out = add_form_features(_toy_df())
    first_A = out[(out.driver == "A") & (out.circuit == "R0")].iloc[0]
    assert pd.isna(first_A["driver_form_last5"])          # nothing before it
    assert first_A["driver_points_before"] == 0.0


def test_form_uses_only_past_races_no_leakage():
    out = add_form_features(_toy_df()).sort_values(["driver", "date"])
    a = out[out.driver == "A"].reset_index(drop=True)
    # 2nd race form == 1st race finish (1); 3rd race form == mean(1,2)=1.5
    assert a.loc[1, "driver_form_last5"] == pytest.approx(1.0)
    assert a.loc[2, "driver_form_last5"] == pytest.approx(1.5)
    # points-before accumulates previous rounds only: 0, 25, 25+24
    assert list(a["driver_points_before"]) == pytest.approx([0.0, 25.0, 49.0])


def test_places_gained_and_dnf_columns_created():
    out = add_form_features(_toy_df())
    for col in ["dnf", "places_gained", "driver_places_gained_last5",
                "driver_dnf_rate_last5", "constructor_dnf_rate_last5",
                "constructor_form_last5", "driver_podium_rate_last5"]:
        assert col in out.columns
