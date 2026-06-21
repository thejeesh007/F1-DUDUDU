"""
F1 Predictor - Streamlit App

Run locally with:
    pip install streamlit pandas lightgbm
    streamlit run app.py

Two main sections:
    1. 2026 Austria GP Prediction (uses the trained LightGBM model + real
       2026 season-to-date form - see predict_austria_2026.py for the
       underlying logic, reused here).
    2. Grid Info panel - driver/team/team principal/race engineer lookup
       for the full 2026 grid, shown via a button click per team.

Data files expected in ./f1_data/:
    master_with_features.csv, f1_model.txt
(These are produced by the earlier pipeline scripts in this project.)
"""

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_info"))

from grid_info.grid_2026_info import GRID_INFO_2026
from calendar_2026 import CALENDAR_2026

DATA_DIR = "f1_data"

FEATURE_COLS = [
    "grid", "qualifying_position", "lat", "long", "engine_tier",
    "air_temp_mean", "air_temp_max", "track_temp_mean", "track_temp_max",
    "humidity_mean", "wind_speed_mean", "rain_fraction_of_race",
    "driver_form_last3", "driver_form_last5",
    "constructor_form_last3", "constructor_form_last5",
    "driver_track_history", "constructor_track_history",
    "driver_overtaking_form_last5",
    "driver_dnf_rate_last5", "driver_dnf_rate_last10",
    "constructor_dnf_rate_last5", "constructor_dnf_rate_last10",
    "driver_wet_weather_form",
    "q1_time_sec", "q2_time_sec", "q3_time_sec",
    "country", "engine_manufacturer", "rained_during_race",
]
CATEGORICAL_FEATURES = ["country", "engine_manufacturer"]


st.set_page_config(page_title="F1 Predictor", page_icon="\U0001F3CE", layout="wide")
st.title("\U0001F3CE F1 Race Predictor")


# ---------------------------------------------------------------------------
# SECTION 1: Grid Info panel
# ---------------------------------------------------------------------------
st.header("2026 Grid Info")
st.caption(
    "Driver, team principal, and race engineer lookup for the full 2026 grid. "
    "Click a team to expand."
)

for team in GRID_INFO_2026:
    with st.expander(f"**{team['constructor']}** \u2014 TP: {team['team_principal']}"):
        for d in team["drivers"]:
            engineer = d["race_engineer"] or "_not publicly confirmed as of writing_"
            st.markdown(f"**#{d['number']} {d['name']}** \u2014 Race Engineer: {engineer}")

st.divider()


# ---------------------------------------------------------------------------
# SECTION 2: Next-race prediction (automatically determined - no manual
# race-name editing needed between race weekends)
# ---------------------------------------------------------------------------


@st.cache_data
def load_2026_data():
    master = pd.read_csv(os.path.join(DATA_DIR, "master_with_features.csv"))
    season_2026 = master[master["season"] == 2026].sort_values(["driver_id", "round"])
    return season_2026


def determine_next_race(season_2026):
    """
    The next race to predict is simply the round after the most recent
    one we have real data for. This is what makes the app NOT need manual
    editing every other weekend - it just looks at what's actually in the
    data and infers what's coming next from the calendar lookup.
    """
    last_completed_round = int(season_2026["round"].max())
    next_round = last_completed_round + 1
    next_race_name = CALENDAR_2026.get(next_round, f"Round {next_round}")
    return next_round, next_race_name, last_completed_round


season_2026 = load_2026_data()
next_round, next_race_name, last_completed_round = determine_next_race(season_2026)

st.header(f"2026 {next_race_name} \u2014 Pre-Race Prediction")
st.caption(
    f"Round {next_round} of 22. Based on real 2026 season data through "
    f"round {last_completed_round} ({CALENDAR_2026.get(last_completed_round, 'last known race')})."
)
st.warning(
    f"**Limitation:** {next_race_name} qualifying hasn't happened yet as of this prediction. "
    "This uses each driver's 2026 season-to-date average qualifying position as a "
    "proxy for grid position, combined with real current rolling form. "
    "It reflects who looks strongest heading into this race based on real 2026 form \u2014 "
    "NOT a true grid-aware prediction, which requires the actual qualifying session."
)


@st.cache_resource
def load_model():
    return lgb.Booster(model_file=os.path.join(DATA_DIR, "f1_model.txt"))


def build_prediction_rows(season_2026):
    rows = []
    for driver_id, driver_df in season_2026.groupby("driver_id"):
        latest = driver_df.iloc[-1]
        avg_quali_2026 = driver_df["qualifying_position"].mean()

        rows.append({
            "driver_id": driver_id,
            "driver_name": latest["driver_name"],
            "constructor_id": latest["constructor_id"],
            "grid": avg_quali_2026,
            "qualifying_position": avg_quali_2026,
            "lat": latest["lat"], "long": latest["long"],
            "engine_tier": latest["engine_tier"],
            "engine_manufacturer": latest["engine_manufacturer"],
            "country": latest["country"],
            "air_temp_mean": np.nan, "air_temp_max": np.nan,
            "track_temp_mean": np.nan, "track_temp_max": np.nan,
            "humidity_mean": np.nan, "wind_speed_mean": np.nan,
            "rain_fraction_of_race": np.nan, "rained_during_race": False,
            "driver_form_last3": latest["driver_form_last3"],
            "driver_form_last5": latest["driver_form_last5"],
            "constructor_form_last3": latest["constructor_form_last3"],
            "constructor_form_last5": latest["constructor_form_last5"],
            "driver_track_history": latest["driver_track_history"],
            "constructor_track_history": latest["constructor_track_history"],
            "driver_overtaking_form_last5": latest["driver_overtaking_form_last5"],
            "driver_dnf_rate_last5": latest["driver_dnf_rate_last5"],
            "driver_dnf_rate_last10": latest["driver_dnf_rate_last10"],
            "constructor_dnf_rate_last5": latest["constructor_dnf_rate_last5"],
            "constructor_dnf_rate_last10": latest["constructor_dnf_rate_last10"],
            "driver_wet_weather_form": latest["driver_wet_weather_form"],
            "q1_time_sec": np.nan, "q2_time_sec": np.nan, "q3_time_sec": np.nan,
        })
    return pd.DataFrame(rows)


try:
    model = load_model()

    pred_df = build_prediction_rows(season_2026)
    for col in CATEGORICAL_FEATURES:
        pred_df[col] = pred_df[col].astype("category")

    pred_df["model_score"] = model.predict(pred_df[FEATURE_COLS])
    pred_df["predicted_rank"] = (
        pred_df["model_score"].rank(ascending=False, method="first").astype(int)
    )
    ranked = pred_df.sort_values("predicted_rank")

    display_df = ranked[["predicted_rank", "driver_name", "constructor_id",
                          "grid", "driver_form_last5"]].rename(columns={
        "predicted_rank": "Rank", "driver_name": "Driver",
        "constructor_id": "Team", "grid": "Avg Quali 2026",
        "driver_form_last5": "Form (Last 5)",
    })
    display_df["Avg Quali 2026"] = display_df["Avg Quali 2026"].round(1)
    display_df["Form (Last 5)"] = display_df["Form (Last 5)"].round(2)

    st.dataframe(display_df.head(10), hide_index=True, width="stretch")

except FileNotFoundError as e:
    st.error(
        f"Couldn't find required data file: {e}. "
        f"Make sure master_with_features.csv and f1_model.txt are in the "
        f"./{DATA_DIR}/ folder next to this app."
    )