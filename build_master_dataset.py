"""
Build the joined master dataset: one row per driver per race, 2018-2025.

Joins together:
    results.csv      (base table - finish position, points, status)
    qualifying.csv    (grid position, Q1/Q2/Q3 times)
    races.csv         (race date - already mostly redundant with results.csv but used for safety)
    circuits.csv      (circuit name, country, lat/long)
    engines.csv       (engine manufacturer per season/constructor)
    weather.csv       (race-day weather summary: temps, rain)

Also cleans up the messy "classified but didn't finish" issue:
    The raw finish_position column treats a car that crashed out but was
    still classified 18th the same as a car that actually finished 18th
    after running the full race distance. We add a clean `dnf` flag so the
    model can tell these apart, since they represent very different races.

Output:
    f1_data/master.csv - the joined, cleaned dataset
"""

import pandas as pd
import os

DATA_DIR = "f1_data"


def load_all():
    results = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))
    qualifying = pd.read_csv(os.path.join(DATA_DIR, "qualifying.csv"))
    circuits = pd.read_csv(os.path.join(DATA_DIR, "circuits.csv"))
    engines = pd.read_csv(os.path.join(DATA_DIR, "engines.csv"))
    weather = pd.read_csv(os.path.join(DATA_DIR, "weather.csv"))
    return results, qualifying, circuits, engines, weather


def add_dnf_flag(df):
    """
    finish_position_text tells us the REAL outcome:
      - a number string ('1', '2', ... '20')  -> actually finished/classified normally
      - 'R' -> retired (didn't finish)
      - 'D' -> disqualified
      - 'W' -> withdrew

    finish_position (the integer column) is the classified running order,
    which can include retirees who completed most of the distance. For
    modeling, we want a clean separate signal for "did this car actually
    have a normal finish" vs "did something go wrong" - DNF, crash,
    mechanical failure, disqualification are fundamentally different
    events from a clean finish, even if both got "classified 18th".
    """
    df = df.copy()
    df["dnf"] = ~df["finish_position_text"].str.isnumeric()
    # also keep the specific reason in a clean column (mirrors status, but
    # explicit so it's obvious this is the DNF reason, not a generic status string)
    df["dnf_reason"] = df["status"].where(df["dnf"], other=None)
    return df


def build_master():
    results, qualifying, circuits, engines, weather = load_all()

    # 1. Start from results (the base grain: one row per driver per race)
    df = results.copy()

    # 2. Add DNF flag/reason before anything else, since it's derived
    #    purely from columns already in results.
    df = add_dnf_flag(df)

    # 3. Join qualifying data (grid position is already in results from the
    #    race itself, but qualifying_position + Q1/Q2/Q3 times only live
    #    in qualifying.csv). Join on (season, round, driver_id).
    quali_cols = ["season", "round", "driver_id", "qualifying_position",
                  "q1_time", "q2_time", "q3_time"]
    df = df.merge(
        qualifying[quali_cols],
        on=["season", "round", "driver_id"],
        how="left",  # left join: keep all results rows even if quali is missing (rare, e.g. wet-weather no-time sessions)
    )

    # 4. Join circuit info on circuit_id
    df = df.merge(
        circuits,
        on="circuit_id",
        how="left",
    )

    # 5. Join engine manufacturer on (season, constructor_id)
    df = df.merge(
        engines,
        on=["season", "constructor_id"],
        how="left",
    )

    # 6. Join weather on (season, round) - this is race-level data (same
    #    weather summary applies to every driver in that race), so every
    #    driver row for a given race gets the same weather values.
    weather_cols = ["season", "round", "air_temp_mean", "air_temp_max",
                     "track_temp_mean", "track_temp_max", "humidity_mean",
                     "wind_speed_mean", "rained_during_race", "rain_fraction_of_race"]
    df = df.merge(
        weather[weather_cols],
        on=["season", "round"],
        how="left",
    )

    return df


def sanity_check(df):
    print("=== MASTER DATASET SANITY CHECK ===")
    print(f"Shape: {df.shape}")
    print()
    print("Missing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print()
    print(f"DNF rate: {df['dnf'].mean():.1%}")
    print()
    print("Rows with no engine_manufacturer matched (should be 0):")
    print(df[df["engine_manufacturer"].isnull()][["season", "constructor_id"]].drop_duplicates())
    print()
    print("Rows with no qualifying_position matched:")
    missing_quali = df[df["qualifying_position"].isnull()]
    print(f"  {len(missing_quali)} rows")
    if len(missing_quali) > 0:
        print(missing_quali[["season", "round", "driver_id", "status"]].head(10))
    print()
    print("Rows with no weather matched (should be 0):")
    missing_weather = df[df["air_temp_mean"].isnull()][["season", "round", "race_name"]].drop_duplicates()
    print(missing_weather if len(missing_weather) > 0 else "  none")
    print()
    print(f"Race-day rain rate (% of races, not driver-rows): "
          f"{df.drop_duplicates(subset=['season','round'])['rained_during_race'].mean():.1%}")


if __name__ == "__main__":
    df = build_master()
    sanity_check(df)
    out_path = os.path.join(DATA_DIR, "master.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved master dataset: {out_path} ({df.shape[0]} rows, {df.shape[1]} columns)")