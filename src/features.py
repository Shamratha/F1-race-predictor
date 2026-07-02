"""Pure feature-engineering functions (no FastF1 / network dependency).

Kept separate from build_dataset.py so they can be unit-tested in isolation and
reused without pulling in the heavy data-collection stack.
"""
import numpy as np
import pandas as pd


def best_quali_time(row):
    """Best (smallest) lap time across Q1/Q2/Q3, in seconds. NaN if none set."""
    times = [row.get("Q1"), row.get("Q2"), row.get("Q3")]
    secs = [t.total_seconds() for t in times if pd.notna(t)]
    return min(secs) if secs else np.nan


def is_dnf(status):
    """A driver is 'classified' if Finished or lapped (+N Laps); else it's a DNF."""
    s = str(status)
    return 0 if (s == "Finished" or s.startswith("+")) else 1


def add_form_features(df):
    """Add rolling recent-form features. Uses only PAST races (no leakage).

    Every rolling feature is shifted by one race per group, so a driver's row
    never sees its own result — only earlier races feed the features.
    """
    df = df.sort_values(["date", "circuit"]).reset_index(drop=True)

    # Per-race raw signals the rolling features are built from
    df["dnf"] = df["status"].apply(is_dnf)
    grid = df["grid_position"].replace(0, 20)  # pit-lane start counts as back row
    df["places_gained"] = grid - df["finish_position"]

    # Driver recent form: mean finish + podium rate over previous 5 races
    df["driver_form_last5"] = (
        df.groupby("driver")["finish_position"]
          .transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    )
    df["driver_podium_rate_last5"] = (
        df.groupby("driver")["podium"]
          .transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    )

    # Constructor recent form: mean finish of the team over previous 5 races
    df["constructor_form_last5"] = (
        df.groupby("team")["finish_position"]
          .transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    )

    # Race-pace edge: does this driver typically gain places on Sunday?
    # This is signal the starting grid does NOT contain.
    df["driver_places_gained_last5"] = (
        df.groupby("driver")["places_gained"]
          .transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    )

    # Reliability: how often the driver / car has failed to finish recently.
    df["driver_dnf_rate_last5"] = (
        df.groupby("driver")["dnf"]
          .transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    )
    df["constructor_dnf_rate_last5"] = (
        df.groupby("team")["dnf"]
          .transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    )

    # Points accumulated in the season BEFORE this race (championship momentum)
    df["driver_points_before"] = (
        df.groupby(["season", "driver"])["points"]
          .transform(lambda s: s.shift().cumsum())
          .fillna(0.0)
    )
    return df
