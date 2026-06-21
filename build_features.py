"""
Feature engineering for the F1 race result predictor.

CRITICAL RULE FOLLOWED THROUGHOUT THIS SCRIPT:
Every rolling/historical feature for a given race must be computed using
ONLY data from races that happened BEFORE it. If we accidentally include
the current race's own result in a "recent form" feature, the model will
appear to perform great during training (because it's partially looking
at the answer) and then fail in real prediction, where future results
obviously aren't available yet. This is called data leakage and it's the
single most common mistake in this kind of project.

The way we guarantee this: every rolling calculation uses pandas .shift(1)
BEFORE computing a rolling window, so the window for race N only ever
includes races 1..N-1, never race N itself.

Features built:
    1. Rolling driver form     - avg finish position, last 3 and last 5 races
    2. Rolling constructor form - same, at team level
    3. Track-specific history   - driver's and constructor's avg finish at this circuit (all prior years)
    4. Grid-to-finish tendency  - rolling avg of (grid - finish) = positions gained/lost
    5. DNF risk                 - rolling DNF rate, driver and constructor
    6. Wet-weather performance  - driver's avg finish in past wet races specifically
    7. Engine tier              - simple ordinal encoding of engine manufacturer strength

Output:
    f1_data/master_with_features.csv
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = "f1_data"


def load_master():
    return pd.read_csv(os.path.join(DATA_DIR, "master.csv"))


def sort_chronologically(df):
    """
    Ensure rows are in true chronological race order. (season, round) works
    because round number always increases within a season in order raced.
    """
    return df.sort_values(["season", "round"]).reset_index(drop=True)


def add_rolling_driver_form(df, windows=(3, 5)):
    """
    Rolling average finish position per driver, computed using only PAST
    races (shift(1) before rolling - see module docstring).

    We use finish_position here, not a DNF-aware metric, deliberately:
    a DNF still gets some classified position (often last), which is a
    fair reflection of "this race went badly" for a form metric. We treat
    DNF risk as its own separate feature instead (see add_rolling_dnf_rate).
    """
    df = df.copy()
    for w in windows:
        col = f"driver_form_last{w}"
        df[col] = (
            df.groupby("driver_id")["finish_position"]
            .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
        )
    return df


def add_rolling_constructor_form(df, windows=(3, 5)):
    """
    Same idea as driver form, but at constructor level. Uses BOTH of that
    constructor's cars (so it reflects true team form, not just one driver).
    """
    df = df.copy()
    for w in windows:
        col = f"constructor_form_last{w}"
        df[col] = (
            df.groupby("constructor_id")["finish_position"]
            .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
        )
    return df


def add_track_history(df):
    """
    Driver's and constructor's average finish position at this exact
    circuit, using only prior years/visits (never the current race).

    Example: for the 2024 Monaco GP row, this looks at how that driver did
    at Monaco in 2018-2023, NOT including the 2024 result itself.
    """
    df = df.copy()

    df["driver_track_history"] = (
        df.groupby(["driver_id", "circuit_id"])["finish_position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df["constructor_track_history"] = (
        df.groupby(["constructor_id", "circuit_id"])["finish_position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    return df


def add_grid_to_finish_tendency(df, windows=(5,)):
    """
    Positive value = driver tends to GAIN positions during the race
    (good racecraft/overtaking). Negative = tends to LOSE positions
    (e.g. poor starts, or a car that's faster in quali than in the race).

    Computed as (grid - finish_position) per race, then rolled using only
    past races, same leakage-safe pattern as everything else here.
    """
    df = df.copy()
    df["positions_gained"] = df["grid"] - df["finish_position"]
    for w in windows:
        col = f"driver_overtaking_form_last{w}"
        df[col] = (
            df.groupby("driver_id")["positions_gained"]
            .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
        )
    return df


def add_rolling_dnf_rate(df, windows=(5, 10)):
    """
    Rolling fraction of races that ended in a DNF, per driver and per
    constructor. This is a genuinely different signal from "form" - a
    driver might have great finish positions when they finish, but a high
    crash/mechanical-failure rate, which matters a lot for predicting
    podium probability specifically (DNF = zero chance of podium).
    """
    df = df.copy()
    df["dnf_int"] = df["dnf"].astype(int)
    for w in windows:
        df[f"driver_dnf_rate_last{w}"] = (
            df.groupby("driver_id")["dnf_int"]
            .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
        )
        df[f"constructor_dnf_rate_last{w}"] = (
            df.groupby("constructor_id")["dnf_int"]
            .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
        )
    df = df.drop(columns=["dnf_int"])
    return df


def add_wet_weather_performance(df):
    """
    Driver's average finish position specifically in past WET races only.
    This is a different, more specific signal than general form - some
    drivers are known to perform relatively better or worse in the rain
    compared to their dry-weather baseline.

    Uses expanding (not a fixed window) since wet races are relatively
    rare - a fixed window of 5 could span many seasons for some drivers.

    NOTE: only valid for rows where the row itself might be wet or dry -
    we're not filtering the OUTPUT rows, just the HISTORY used to compute
    the feature. Every row gets a "how has this driver done in wet races
    historically" value regardless of whether today's race is wet, since
    the model itself will combine this with today's rain forecast/actual
    as a separate feature.
    """
    df = df.copy()

    # rained_during_race can come in as bool, float (if NaNs are present
    # after a CSV round-trip), or object dtype - normalize explicitly
    # rather than assume a clean boolean column. Rows where we genuinely
    # don't know the weather (e.g. not-yet-fetched future/recent races)
    # are treated as "not wet" for this feature's purposes, since we
    # can't include them in wet-race history either way - they simply
    # won't contribute to the wet-weather average, same as any dry race.
    rained_bool = df["rained_during_race"].fillna(False).astype(bool)

    df["_wet_finish"] = df["finish_position"].where(rained_bool, other=np.nan)

    df["driver_wet_weather_form"] = (
        df.groupby("driver_id")["_wet_finish"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df = df.drop(columns=["_wet_finish"])
    return df


def add_engine_tier(df):
    """
    Simple ordinal tier for engine manufacturers based on broad real-world
    performance reputation across the 2018-2025 period. This is a coarse
    simplification (engine performance shifts year to year) but gives the
    model a useful prior beyond the one-hot/raw manufacturer name alone.

    Tier 3 = historically strongest, Tier 1 = historically weakest.
    This is intentionally simple - rolling constructor form already
    captures more precise, time-varying performance.
    """
    df = df.copy()
    tier_map = {
        "Mercedes": 3,
        "Ferrari": 3,
        "Honda": 2,
        "Honda RBPT": 3,   # Red Bull's engine was a top performer 2022+ despite the rebrand
        "Renault": 1,
    }
    df["engine_tier"] = df["engine_manufacturer"].map(tier_map)
    return df


def build_features():
    df = load_master()
    df = sort_chronologically(df)

    df = add_rolling_driver_form(df)
    df = add_rolling_constructor_form(df)
    df = add_track_history(df)
    df = add_grid_to_finish_tendency(df)
    df = add_rolling_dnf_rate(df)
    df = add_wet_weather_performance(df)
    df = add_engine_tier(df)

    return df


def leakage_check(df):
    """
    Spot-check that rolling features are NaN on the FIRST appearance of
    the entity they're keyed on:
      - driver-level features  -> NaN on that driver's first-ever race
      - constructor-level features -> NaN on that constructor's first-ever race
    (track-history features are checked on first visit to that specific
    circuit, since that's the entity they're scoped to.)

    Checking against the wrong entity gives false alarms - e.g. a rookie
    driver's first race is very likely NOT their team's first race, so
    constructor_form being populated there is correct, not a leak.
    """
    print("=== LEAKAGE SANITY CHECK ===")
    df_sorted = df.sort_values(["season", "round"])

    checks = {
        "driver_id": ["driver_form_last3", "driver_form_last5",
                       "driver_overtaking_form_last5", "driver_dnf_rate_last5",
                       "driver_dnf_rate_last10", "driver_wet_weather_form"],
        "constructor_id": ["constructor_form_last3", "constructor_form_last5",
                            "constructor_dnf_rate_last5", "constructor_dnf_rate_last10"],
    }

    any_leak = False
    for group_key, cols in checks.items():
        first_rows = df_sorted.groupby(group_key).head(1)
        bad = first_rows[cols].notna().any(axis=0)
        leaking = bad[bad].index.tolist()
        if leaking:
            any_leak = True
            print(f"[!] POSSIBLE LEAKAGE (grouped by {group_key}): {leaking}")
        else:
            print(f"[OK] All {group_key}-level features are NaN on first appearance, as expected.")

    # Track history is double-keyed (entity, circuit) - check first visit
    # of each (driver, circuit) / (constructor, circuit) pair specifically.
    first_track_visits_driver = df_sorted.groupby(["driver_id", "circuit_id"]).head(1)
    if first_track_visits_driver["driver_track_history"].notna().any():
        any_leak = True
        print("[!] POSSIBLE LEAKAGE: driver_track_history populated on first visit to a circuit")
    else:
        print("[OK] driver_track_history is NaN on first visit to each circuit, as expected.")

    first_track_visits_constructor = df_sorted.groupby(["constructor_id", "circuit_id"]).head(1)
    if first_track_visits_constructor["constructor_track_history"].notna().any():
        any_leak = True
        print("[!] POSSIBLE LEAKAGE: constructor_track_history populated on first visit to a circuit")
    else:
        print("[OK] constructor_track_history is NaN on first visit to each circuit, as expected.")

    if not any_leak:
        print("\n[OK] No leakage detected across any feature group.")
    print()


def feature_summary(df):
    print("=== FEATURE SUMMARY ===")
    feature_cols = [c for c in df.columns if any(
        kw in c for kw in ["form", "history", "dnf_rate", "overtaking", "engine_tier", "positions_gained"]
    )]
    print(f"New feature columns added: {len(feature_cols)}")
    for c in feature_cols:
        non_null = df[c].notna().sum()
        print(f"  {c}: {non_null}/{len(df)} non-null ({non_null/len(df):.1%})")


if __name__ == "__main__":
    df = build_features()
    leakage_check(df)
    feature_summary(df)

    out_path = os.path.join(DATA_DIR, "master_with_features.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({df.shape[0]} rows, {df.shape[1]} columns)")