"""
Train/test split for the F1 race result predictor.

SPLIT STRATEGY: time-based, not random.
    Train: 2018-2023 (6 seasons, 2500 rows)
    Test:  2024-2025 (2 seasons, 958 rows)

This matters because randomly shuffling rows into train/test could put a
driver's 2023 races in training and their 2022 races in test - meaning
the model would effectively be "predicting the past" using future
information for that driver. A time-based split guarantees the model is
only ever evaluated on races that happen AFTER everything it trained on,
which is the only honest way to simulate real prediction.

FEATURE SELECTION - three categories, handled very differently:

1. TARGET (what we're predicting):
     finish_position

2. EXCLUDED - race-outcome columns (only known AFTER the race finishes -
   including these as features would let the model see the answer):
     points, status, laps, fastest_lap_rank, fastest_lap_time, dnf,
     dnf_reason, finish_position_text, positions_gained
   (positions_gained = grid - finish_position, literally derived from
   the target itself, so it's an extreme case of leakage if included.)

3. EXCLUDED - pure identifiers (not predictive on their own, kept only
   for grouping/joining/debugging, not fed to the model):
     season, round, race_name, circuit_id, date, driver_id, driver_code,
     driver_name, constructor_id, circuit_name, locality

4. FEATURES (everything else - legitimately known before the race):
     grid, qualifying_position, q1_time, q2_time, q3_time,
     country, lat, long, engine_manufacturer, engine_tier,
     air_temp_mean, air_temp_max, track_temp_mean, track_temp_max,
     humidity_mean, wind_speed_mean, rained_during_race, rain_fraction_of_race,
     driver_form_last3, driver_form_last5,
     constructor_form_last3, constructor_form_last5,
     driver_track_history, constructor_track_history,
     driver_overtaking_form_last5,
     driver_dnf_rate_last5, driver_dnf_rate_last10,
     constructor_dnf_rate_last5, constructor_dnf_rate_last10,
     driver_wet_weather_form

   NOTE on weather features: these are ACTUAL race-day weather (what
   really happened), not a forecast. In true future prediction we
   wouldn't know this in advance - we'd only have a forecast, which is
   less certain. We're using actual weather for now to measure its
   ceiling effect on prediction quality; swapping in forecast data is a
   follow-up step before any real deployment.

   NOTE on q1_time/q2_time/q3_time: these are lap time STRINGS
   (e.g. "1:23.456"), not usable directly by most models - they need
   converting to seconds (float) before training. Handled in this script.

Output:
    f1_data/train.csv
    f1_data/test.csv
"""

import pandas as pd
import os

DATA_DIR = "f1_data"

TARGET_COL = "finish_position"

EXCLUDED_OUTCOME_COLS = [
    "points", "status", "laps", "fastest_lap_rank", "fastest_lap_time",
    "dnf", "dnf_reason", "finish_position_text", "positions_gained",
]

ID_COLS = [
    "season", "round", "race_name", "circuit_id", "date", "driver_id",
    "driver_code", "driver_name", "constructor_id", "circuit_name", "locality",
]

RAW_TIME_COLS = ["q1_time", "q2_time", "q3_time"]  # need conversion, handled below

CATEGORICAL_FEATURES = ["country", "engine_manufacturer"]

NUMERIC_FEATURES = [
    "grid", "qualifying_position",
    "lat", "long", "engine_tier",
    "air_temp_mean", "air_temp_max", "track_temp_mean", "track_temp_max",
    "humidity_mean", "wind_speed_mean", "rain_fraction_of_race",
    "driver_form_last3", "driver_form_last5",
    "constructor_form_last3", "constructor_form_last5",
    "driver_track_history", "constructor_track_history",
    "driver_overtaking_form_last5",
    "driver_dnf_rate_last5", "driver_dnf_rate_last10",
    "constructor_dnf_rate_last5", "constructor_dnf_rate_last10",
    "driver_wet_weather_form",
]

BOOLEAN_FEATURES = ["rained_during_race"]


def lap_time_to_seconds(time_str):
    """
    Convert F1 lap time strings like '1:23.456' (1 min 23.456 sec) to
    total seconds as a float. Returns None for missing/unparseable values
    (e.g. drivers eliminated in Q1 have no Q2/Q3 time at all).
    """
    if pd.isna(time_str):
        return None
    try:
        parts = str(time_str).split(":")
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        return float(parts[0])
    except (ValueError, IndexError):
        return None


def prepare_dataset():
    df = pd.read_csv(os.path.join(DATA_DIR, "master_with_features.csv"))

    # Convert lap time strings to numeric seconds
    for col in RAW_TIME_COLS:
        df[col + "_sec"] = df[col].apply(lap_time_to_seconds)

    numeric_time_cols = [c + "_sec" for c in RAW_TIME_COLS]

    feature_cols = NUMERIC_FEATURES + numeric_time_cols + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

    # Keep ID columns alongside features/target so we can trace any row
    # back to the actual race/driver later (e.g. for error analysis) -
    # the modeling script will be responsible for dropping ID_COLS before
    # actually fitting, this script just organizes the split.
    keep_cols = ID_COLS + feature_cols + [TARGET_COL]
    df = df[keep_cols].copy()

    return df, feature_cols


def split_train_test(df):
    train = df[df["season"] <= 2023].copy()
    test = df[df["season"] >= 2024].copy()
    return train, test


def sanity_check(train, test, feature_cols):
    print("=== TRAIN/TEST SPLIT SANITY CHECK ===")
    print(f"Train: {train.shape[0]} rows, seasons {sorted(train['season'].unique())}")
    print(f"Test:  {test.shape[0]} rows, seasons {sorted(test['season'].unique())}")
    print()

    overlap = set(train["season"]) & set(test["season"])
    if overlap:
        print(f"[!] SEASON OVERLAP between train and test: {overlap}")
    else:
        print("[OK] No season overlap between train and test.")
    print()

    print(f"Feature columns ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"  {c}")
    print()

    print("Missing value % in TRAIN features:")
    print((train[feature_cols].isnull().mean() * 100).round(1).sort_values(ascending=False))
    print()
    print("Missing value % in TEST features:")
    print((test[feature_cols].isnull().mean() * 100).round(1).sort_values(ascending=False))


if __name__ == "__main__":
    df, feature_cols = prepare_dataset()
    train, test = split_train_test(df)
    sanity_check(train, test, feature_cols)

    train.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    test.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
    print(f"\nSaved {DATA_DIR}/train.csv ({len(train)} rows) and {DATA_DIR}/test.csv ({len(test)} rows)")