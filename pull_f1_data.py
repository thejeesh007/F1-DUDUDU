"""
F1 2026 Season-to-Date Puller — run this on YOUR machine (not in the chat sandbox).

Pulls all races run SO FAR in the 2026 season (results + qualifying) from
the Jolpica-F1 API. As of writing, 2026 is mid-season (7 rounds done,
next is Austria). This lets us:
    1. See actual 2026 results so far (confirm who's actually fast this year)
    2. Compute REAL 2026 rolling-form features (not carried-forward 2025
       guesses) for predicting the upcoming Austria GP

This follows the exact same pattern as pull_f1_data.py from earlier in
the project - same retry logic, same incremental saving.

Requirements:
    pip install requests pandas

Output:
    f1_data/results_2026.csv
    f1_data/qualifying_2026.csv
"""

import requests
import pandas as pd
import time
import os

BASE_URL = "https://api.jolpi.ca/ergast/f1"
SEASON = 2026
OUTPUT_DIR = "f1_data"
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


def get_races_so_far():
    data = get_json(f"{BASE_URL}/{SEASON}.json", {"limit": 100})
    return data["MRData"]["RaceTable"]["Races"]


def pull_race_results(round_num):
    data = get_json(f"{BASE_URL}/{SEASON}/{round_num}/results.json", {"limit": 100})
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return []  # race hasn't happened yet, or no data available

    race = races[0]
    rows = []
    for r in race.get("Results", []):
        rows.append({
            "season": SEASON,
            "round": round_num,
            "race_name": race.get("raceName"),
            "circuit_id": race.get("Circuit", {}).get("circuitId"),
            "date": race.get("date"),
            "driver_id": r.get("Driver", {}).get("driverId"),
            "driver_code": r.get("Driver", {}).get("code"),
            "driver_name": f"{r.get('Driver', {}).get('givenName', '')} {r.get('Driver', {}).get('familyName', '')}".strip(),
            "constructor_id": r.get("Constructor", {}).get("constructorId"),
            "grid": r.get("grid"),
            "finish_position": r.get("position"),
            "finish_position_text": r.get("positionText"),
            "points": r.get("points"),
            "status": r.get("status"),
            "laps": r.get("laps"),
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
            "season": SEASON,
            "round": round_num,
            "race_name": race.get("raceName"),
            "driver_id": r.get("Driver", {}).get("driverId"),
            "constructor_id": r.get("Constructor", {}).get("constructorId"),
            "qualifying_position": r.get("position"),
            "q1_time": r.get("Q1"),
            "q2_time": r.get("Q2"),
            "q3_time": r.get("Q3"),
        })
    return rows


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Checking 2026 season schedule...")
    races = get_races_so_far()
    print(f"Found {len(races)} races on the 2026 calendar (includes future, not-yet-run races)\n")

    all_results = []
    all_qualifying = []
    races_with_data = 0

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
            races_with_data += 1
            print(f" OK ({len(results)} results)")
        else:
            print(" not yet run / no data")

    pd.DataFrame(all_results).to_csv(os.path.join(output_dir, "results_2026.csv"), index=False)
    pd.DataFrame(all_qualifying).to_csv(os.path.join(output_dir, "qualifying_2026.csv"), index=False)

    print(f"\nDone! {races_with_data} races had real data.")
    print(f"Saved results_2026.csv ({len(all_results)} rows) and qualifying_2026.csv ({len(all_qualifying)} rows)")


if __name__ == "__main__":
    main()