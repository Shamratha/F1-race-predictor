"""Build the F1 podium-prediction dataset from real FastF1 data.

For every race in the chosen seasons we extract, per driver, the "changing
factors" that decide a race:

  * qualifying pace   -> grid_position, quali_gap_to_pole_s
  * track             -> circuit (categorical)
  * weather           -> air_temp, track_temp, humidity, rainfall, wind_speed
  * recent form       -> driver_form_last5, driver_podium_rate_last5,
                         constructor_form_last5, driver_points_before
  * constructor       -> team (categorical)

Target:  podium = 1 if the driver finished in the top 3, else 0.

Usage:
    python src/build_dataset.py 2021 2024      # inclusive season range
    python src/build_dataset.py 2023           # single season
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import fastf1

warnings.filterwarnings("ignore")
fastf1.logger.set_log_level("ERROR")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")
OUT = os.path.join(ROOT, "data", "dataset.csv")
fastf1.Cache.enable_cache(CACHE)


def best_quali_time(row):
    """Best (smallest) lap time across Q1/Q2/Q3, in seconds. NaN if none."""
    times = [row.get("Q1"), row.get("Q2"), row.get("Q3")]
    secs = [t.total_seconds() for t in times if pd.notna(t)]
    return min(secs) if secs else np.nan


def extract_race(year, event_name, event_date):
    """Return a list of per-driver feature dicts for one race, or []."""
    # Load race (results + weather; skip heavy telemetry for speed)
    race = fastf1.get_session(year, event_name, "R")
    race.load(laps=False, telemetry=False, weather=True, messages=False)
    res = race.results.copy()
    if res is None or len(res) == 0:
        return []

    # --- Weather: average conditions across the race session ---
    w = race.weather_data
    if w is not None and len(w) > 0:
        weather = dict(
            air_temp=float(w["AirTemp"].mean()),
            track_temp=float(w["TrackTemp"].mean()),
            humidity=float(w["Humidity"].mean()),
            rainfall=int(bool(w["Rainfall"].any())),
            wind_speed=float(w["WindSpeed"].mean()),
        )
    else:
        weather = dict(air_temp=np.nan, track_temp=np.nan, humidity=np.nan,
                       rainfall=0, wind_speed=np.nan)

    # --- Qualifying pace: gap to pole ---
    quali_gap = {}
    try:
        quali = fastf1.get_session(year, event_name, "Q")
        quali.load(laps=False, telemetry=False, weather=False, messages=False)
        q = quali.results.copy()
        q["best_q"] = q.apply(best_quali_time, axis=1)
        pole = q["best_q"].min(skipna=True)
        for _, r in q.iterrows():
            gap = r["best_q"] - pole if pd.notna(r["best_q"]) else np.nan
            quali_gap[r["Abbreviation"]] = gap
    except Exception:
        pass  # gap stays NaN -> XGBoost handles it natively

    rows = []
    for _, r in res.iterrows():
        abbr = r["Abbreviation"]
        pos = r["Position"]
        rows.append(dict(
            season=year,
            round=int(r.get("_round", 0)) if "_round" in r else 0,
            date=event_date,
            circuit=event_name,
            driver=abbr,
            team=r["TeamName"],
            grid_position=float(r["GridPosition"]) if pd.notna(r["GridPosition"]) else np.nan,
            quali_gap_to_pole_s=quali_gap.get(abbr, np.nan),
            **weather,
            finish_position=float(pos) if pd.notna(pos) else np.nan,
            status=r["Status"],
            points=float(r["Points"]) if pd.notna(r["Points"]) else 0.0,
            podium=int(pd.notna(pos) and pos <= 3),
        ))
    return rows


def is_dnf(status):
    """A driver is 'classified' if Finished or lapped (+N Laps); else it's a DNF."""
    s = str(status)
    return 0 if (s == "Finished" or s.startswith("+")) else 1


def add_form_features(df):
    """Add rolling recent-form features. Uses only PAST races (no leakage)."""
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


def main():
    args = [int(a) for a in sys.argv[1:]] or [2023]
    start, end = (args[0], args[0]) if len(args) == 1 else (args[0], args[1])
    seasons = list(range(start, end + 1))
    print(f"Building dataset for seasons: {seasons}")

    all_rows = []
    for year in seasons:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, ev in schedule.iterrows():
            name = ev["EventName"]
            date = ev["EventDate"]
            if pd.isna(date):
                continue
            try:
                rows = extract_race(year, name, date)
                if rows:
                    all_rows.extend(rows)
                    print(f"  [{year}] {name:<28} {len(rows)} drivers")
            except Exception as e:
                print(f"  [{year}] {name:<28} SKIPPED ({type(e).__name__})")

    if not all_rows:
        print("No data collected. Exiting.")
        return

    df = pd.DataFrame(all_rows)
    df = add_form_features(df)
    df.to_csv(OUT, index=False)
    print(f"\nSaved {len(df)} rows ({df['podium'].sum()} podiums) -> {OUT}")
    print(f"Seasons: {sorted(df['season'].unique())} | "
          f"Races: {df.groupby(['season','circuit']).ngroups}")


if __name__ == "__main__":
    main()
