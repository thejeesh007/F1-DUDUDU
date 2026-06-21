"""
Build 2026 prediction feature rows, applying cold-start fixes from
grid_2026.py to each driver based on their situation category.

This does NOT predict race results directly - it prepares one feature
row per 2026 driver that can be fed into the trained model. Since we
don't know 2026 race-specific details yet (qualifying position, weather,
specific circuit) at this stage, this builds the PRE-SEASON feature
baseline - the model can be re-run per-race once real qualifying/weather
data is available, by overwriting those specific columns.

APPROACH PER CATEGORY:

  normal / partial_history / transition_now_settled / constructor_rebrand:
      Use the driver's (and constructor's) most recent real rolling-feature
      values from master_with_features.csv as the carried-forward baseline.

  true_rookie (Lindblad):
      No real driver history exists. Substitute the empirically-derived
      rookie prior (mean finish ~12.1, DNF rate ~21%) for driver-level
      rolling features. Track-history features stay NaN (genuinely
      unknown) - LightGBM handles this natively.

  returning_veteran_new_team (Perez, Bottas):
      Use their real historical features (better than nothing) but flag
      the gap via seasons_since_last_race, and note that constructor-level
      features will reflect the NEW team (Cadillac), not their old one.

  constructor_rebrand (Audi inheriting Sauber):
      Constructor-level features computed from Sauber's rows directly,
      via CONSTRUCTOR_CONTINUITY_MAP.

  engine_change (Aston Martin -> Honda):
      Just an engine_manufacturer/engine_tier update, applied directly.

Output:
    f1_data/predictions_2026_preseason.csv
"""

import pandas as pd
import numpy as np
import os
from grid_2026 import (
    GRID_2026, CONSTRUCTOR_CONTINUITY_MAP, ENGINE_UPDATES_2026,
    ROOKIE_PRIOR_FINISH, ROOKIE_PRIOR_DNF_RATE,
    NEW_CONSTRUCTOR_PRIOR_FINISH, NEW_CONSTRUCTOR_PRIOR_DNF_RATE,
)

DATA_DIR = "f1_data"

DRIVER_ROLLING_FEATURES = [
    "driver_form_last3", "driver_form_last5",
    "driver_overtaking_form_last5",
    "driver_dnf_rate_last5", "driver_dnf_rate_last10",
    "driver_wet_weather_form",
]
CONSTRUCTOR_ROLLING_FEATURES = [
    "constructor_form_last3", "constructor_form_last5",
    "constructor_dnf_rate_last5", "constructor_dnf_rate_last10",
]
TRACK_HISTORY_FEATURES = ["driver_track_history", "constructor_track_history"]


def load_master():
    return pd.read_csv(os.path.join(DATA_DIR, "master_with_features.csv"))


def get_latest_driver_features(master, driver_id):
    """
    Returns the most recent row of rolling features for a driver, plus
    which season that came from (so we can compute seasons_since_last_race).
    Returns None if the driver has no history at all (true rookie case).
    """
    rows = master[master["driver_id"] == driver_id].sort_values(["season", "round"])
    if len(rows) == 0:
        return None
    return rows.iloc[-1]


def get_latest_constructor_features(master, constructor_id):
    """
    Same idea at constructor level. Applies CONSTRUCTOR_CONTINUITY_MAP
    first, so e.g. 'audi' correctly pulls Sauber's most recent rows.
    """
    real_id = CONSTRUCTOR_CONTINUITY_MAP.get(constructor_id, constructor_id)
    rows = master[master["constructor_id"] == real_id].sort_values(["season", "round"])
    if len(rows) == 0:
        return None
    return rows.iloc[-1]


def build_driver_row(driver_info, master, current_season=2026):
    """
    Builds one feature row for a single 2026 driver, applying the
    appropriate cold-start fix based on their situation category.
    """
    driver_id = driver_info["driver_id"]
    constructor_id = driver_info["constructor_id"]
    category = driver_info["category"]

    row = {
        "driver_id": driver_id,
        "driver_name": driver_info["driver_name"],
        "constructor_id": constructor_id,
        "category": category,
    }

    # --- Driver-level rolling features ---
    latest_driver = get_latest_driver_features(master, driver_id)

    if category == "true_rookie":
        # No real history exists - use the empirical rookie prior.
        row["driver_form_last3"] = ROOKIE_PRIOR_FINISH
        row["driver_form_last5"] = ROOKIE_PRIOR_FINISH
        row["driver_overtaking_form_last5"] = 0.0  # no basis to assume gain or loss
        row["driver_dnf_rate_last5"] = ROOKIE_PRIOR_DNF_RATE
        row["driver_dnf_rate_last10"] = ROOKIE_PRIOR_DNF_RATE
        row["driver_wet_weather_form"] = np.nan  # genuinely unknown, let model handle via NaN
        row["driver_track_history"] = np.nan
        row["seasons_since_last_race"] = np.nan  # never raced - not applicable
        row["_note"] = "TRUE ROOKIE - using empirical rookie-debut prior, not real history"

    else:
        # All other categories: carry forward real historical features.
        for col in DRIVER_ROLLING_FEATURES + TRACK_HISTORY_FEATURES[:1]:  # driver_track_history only here
            row[col] = latest_driver[col] if latest_driver is not None else np.nan

        last_season_raced = latest_driver["season"] if latest_driver is not None else None
        row["seasons_since_last_race"] = (
            current_season - last_season_raced - 1 if last_season_raced is not None else np.nan
        )
        # -1 because e.g. last raced 2025, predicting 2026 -> 0 seasons gap (consecutive)
        # last raced 2024, predicting 2026 -> 1 season gap (this is Perez/Bottas's case)

        if category == "returning_veteran_new_team":
            row["_note"] = f"Returning after {row['seasons_since_last_race']:.0f} season(s) away, now at a new team - history carried forward but treat with extra caution"
        else:
            row["_note"] = "Normal - using real historical rolling features"

    # --- Constructor-level rolling features ---
    if constructor_id == "cadillac":
        # Genuinely new constructor - use the conservative new-team prior,
        # NOT extrapolated from Haas's outlier 2016 debut.
        row["constructor_form_last3"] = NEW_CONSTRUCTOR_PRIOR_FINISH
        row["constructor_form_last5"] = NEW_CONSTRUCTOR_PRIOR_FINISH
        row["constructor_dnf_rate_last5"] = NEW_CONSTRUCTOR_PRIOR_DNF_RATE
        row["constructor_dnf_rate_last10"] = NEW_CONSTRUCTOR_PRIOR_DNF_RATE
        row["constructor_track_history"] = np.nan
    else:
        latest_constructor = get_latest_constructor_features(master, constructor_id)
        for col in CONSTRUCTOR_ROLLING_FEATURES + ["constructor_track_history"]:
            row[col] = latest_constructor[col] if latest_constructor is not None else np.nan

    # --- Engine manufacturer (apply 2026 updates where relevant) ---
    if constructor_id in ENGINE_UPDATES_2026:
        row["engine_manufacturer"] = ENGINE_UPDATES_2026[constructor_id]
    else:
        latest_constructor = get_latest_constructor_features(master, constructor_id)
        row["engine_manufacturer"] = (
            latest_constructor["engine_manufacturer"] if latest_constructor is not None else np.nan
        )

    return row


def build_all_2026_rows():
    master = load_master()
    rows = [build_driver_row(d, master) for d in GRID_2026]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_all_2026_rows()

    print("=== 2026 PRE-SEASON FEATURE BASELINE ===\n")
    print(df[["driver_name", "constructor_id", "category",
               "driver_form_last5", "constructor_form_last5",
               "seasons_since_last_race", "_note"]].to_string())

    out_path = os.path.join(DATA_DIR, "predictions_2026_preseason.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")