"""
Weekly 2026 Data Refresh - run this after each race weekend.

This is the ONE script to run every other weekend once a new race has
happened. It re-pulls the full 2026 season-to-date data (results +
qualifying) from Jolpica, then re-runs the rest of the pipeline so the
app picks up the new race automatically.

WHAT THIS DOES, IN ORDER:
    1. Re-pulls ALL 2026 results/qualifying so far (not just the new
       race) - simplest and safest approach, avoids any partial-update
       bugs, and the API call is cheap/fast for one season.
    2. Replaces the old 2026 rows in results.csv/qualifying.csv with the
       fresh pull (keeps 2018-2025 rows untouched).
    3. Pulls REAL weather data (via FastF1) for any new race that's
       happened since the last update. This step is the slowest part -
       FastF1 talks to F1's live timing servers, not a lightweight API,
       so expect it to take anywhere from a few seconds (if nothing new
       happened) up to a minute or so for one new race. It automatically
       skips races that haven't happened yet, so it stays fast even as
       the season goes on.
    4. Re-runs build_master_dataset.py and build_features.py so
       master_with_features.csv reflects the new race.
    5. The Streamlit app's "next race" detection (calendar_2026.py +
       determine_next_race()) automatically picks up the change - NO
       manual editing of race names needed anywhere.

Usage:
    python update_2026_data.py

Then just restart/rerun the Streamlit app as normal:
    streamlit run app.py
"""

import requests
import pandas as pd
import time
import os
import subprocess
import sys

BASE_URL = "https://api.jolpi.ca/ergast/f1"
SEASON = 2026
DATA_DIR = "f1_data"
REQUEST_DELAY = 0.5


def get_json(url, params=None):
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=25)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"  retry {attempt+1}/3 for {url}: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to fetch {url} after 3 attempts")


def pull_race_results(round_num):
    data = get_json(f"{BASE_URL}/{SEASON}/{round_num}/results.json", {"limit": 100})
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return []
    race = races[0]
    rows = []
    for r in race.get("Results", []):
        rows.append({
            "season": SEASON, "round": round_num, "race_name": race.get("raceName"),
            "circuit_id": race.get("Circuit", {}).get("circuitId"), "date": race.get("date"),
            "driver_id": r.get("Driver", {}).get("driverId"),
            "driver_code": r.get("Driver", {}).get("code"),
            "driver_name": f"{r.get('Driver', {}).get('givenName', '')} {r.get('Driver', {}).get('familyName', '')}".strip(),
            "constructor_id": r.get("Constructor", {}).get("constructorId"),
            "grid": r.get("grid"), "finish_position": r.get("position"),
            "finish_position_text": r.get("positionText"), "points": r.get("points"),
            "status": r.get("status"), "laps": r.get("laps"),
            "fastest_lap_rank": r.get("FastestLap", {}).get("rank") if r.get("FastestLap") else None,
            "fastest_lap_time": r.get("FastestLap", {}).get("Time", {}).get("time") if r.get("FastestLap") else None,
        })
    return rows


def pull_qualifying_results(round_num):
    data = get_json(f"{BASE_URL}/{SEASON}/{round_num}/qualifying.json", {"limit": 100})
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return []
    race = races[0]
    rows = []
    for r in race.get("QualifyingResults", []):
        rows.append({
            "season": SEASON, "round": round_num, "race_name": race.get("raceName"),
            "driver_id": r.get("Driver", {}).get("driverId"),
            "constructor_id": r.get("Constructor", {}).get("constructorId"),
            "qualifying_position": r.get("position"),
            "q1_time": r.get("Q1"), "q2_time": r.get("Q2"), "q3_time": r.get("Q3"),
        })
    return rows


def pull_all_2026():
    data = get_json(f"{BASE_URL}/{SEASON}.json", {"limit": 100})
    races = data["MRData"]["RaceTable"]["Races"]

    all_results, all_qualifying = [], []
    for race in races:
        round_num = int(race["round"])
        race_name = race.get("raceName")
        print(f"  Round {round_num}: {race_name} ...", end="")

        results = pull_race_results(round_num)
        time.sleep(REQUEST_DELAY)
        qualifying = pull_qualifying_results(round_num)
        time.sleep(REQUEST_DELAY)

        if results:
            all_results.extend(results)
            all_qualifying.extend(qualifying)
            print(f" OK ({len(results)} results)")
        else:
            print(" not yet run")

    return all_results, all_qualifying


def merge_into_main_files(new_2026_results, new_2026_qualifying):
    results_path = os.path.join(DATA_DIR, "results.csv")
    qualifying_path = os.path.join(DATA_DIR, "qualifying.csv")

    results = pd.read_csv(results_path)
    qualifying = pd.read_csv(qualifying_path)

    # Drop any existing 2026 rows, then add the fresh pull back in.
    # This avoids duplicate/stale rows from previous partial updates.
    results = results[results["season"] != SEASON]
    qualifying = qualifying[qualifying["season"] != SEASON]

    results = pd.concat([results, pd.DataFrame(new_2026_results)], ignore_index=True)
    qualifying = pd.concat([qualifying, pd.DataFrame(new_2026_qualifying)], ignore_index=True)

    results.to_csv(results_path, index=False)
    qualifying.to_csv(qualifying_path, index=False)

    print(f"\nUpdated {results_path} ({len(results)} total rows)")
    print(f"Updated {qualifying_path} ({len(qualifying)} total rows)")


def fill_missing_weather_placeholders():
    """
    New 2026 races need a weather.csv row too (even if just NaN
    placeholders), or build_master_dataset.py's weather join will be
    missing rows for them. Real weather still requires the separate
    FastF1-based pull_weather_data.py - this just ensures every race has
    SOME row so the pipeline doesn't break.
    """
    weather_path = os.path.join(DATA_DIR, "weather.csv")
    weather = pd.read_csv(weather_path)
    results = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))

    races_2026 = results[results["season"] == SEASON][["round", "race_name"]].drop_duplicates()
    existing_2026_rounds = set(weather[weather["season"] == SEASON]["round"])

    new_rows = []
    for _, r in races_2026.iterrows():
        if r["round"] not in existing_2026_rounds:
            new_rows.append({
                "season": SEASON, "round": r["round"], "race_name": r["race_name"],
                "air_temp_mean": None, "air_temp_max": None,
                "track_temp_mean": None, "track_temp_max": None,
                "humidity_mean": None, "wind_speed_mean": None,
                "rained_during_race": None, "rain_fraction_of_race": None,
            })

    if new_rows:
        new_rows_df = pd.DataFrame(new_rows)
        # Ensure new rows match existing dtypes (avoids a pandas
        # FutureWarning about all-NaN columns affecting concat dtype
        # inference) - cast new placeholder columns to match weather's
        # existing dtype for each column where possible.
        for col in new_rows_df.columns:
            if col in weather.columns:
                try:
                    new_rows_df[col] = new_rows_df[col].astype(weather[col].dtype)
                except (ValueError, TypeError):
                    pass  # leave as-is if direct cast isn't possible (e.g. object/bool mix)
        weather = pd.concat([weather, new_rows_df], ignore_index=True)
        weather.to_csv(weather_path, index=False)
        print(f"Added {len(new_rows)} new weather placeholder rows for 2026.")


def run_pipeline():
    print("\nPulling real weather data for any new races...")
    # pull_weather_data.py is already resumable and skips races it already
    # has real data for, AND now skips not-yet-run races without wasting
    # time on retries - so it's safe and fast to call every single week,
    # not just a one-time setup step.
    subprocess.run([sys.executable, "pull_weather_data.py"], check=True)

    # Safety net: if the real weather pull above failed for any specific
    # race (e.g. rate-limited), make sure it at least has a placeholder
    # row so the master-dataset join doesn't break. Runs AFTER the real
    # pull so it only fills genuine gaps, never overwrites real data.
    fill_missing_weather_placeholders()

    print("\nRebuilding master dataset and features...")
    subprocess.run([sys.executable, "build_master_dataset.py"], check=True)
    subprocess.run([sys.executable, "build_features.py"], check=True)


if __name__ == "__main__":
    print("=== Refreshing 2026 season data ===\n")
    results, qualifying = pull_all_2026()
    merge_into_main_files(results, qualifying)
    run_pipeline()
    print("\nDone! Restart the Streamlit app to see the updated prediction:")
    print("    streamlit run app.py")