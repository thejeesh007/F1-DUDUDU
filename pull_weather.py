"""
F1 Weather Puller — run this on YOUR machine (not in the chat sandbox).

Pulls race-day weather summary for every race in 2018-2025 using FastF1,
which reads from F1's official live timing feed (minute-by-minute weather
samples during each session).

We don't need minute-by-minute detail for our model — we need ONE summary
row per race: was it wet, what was the temperature. So this script pulls
the full per-minute weather data per race, then collapses it down to
summary stats (mean/max temp, whether ANY rain was recorded, etc.)

Requirements:
    pip install fastf1 pandas

IMPORTANT - rate limiting:
F1's live timing servers WILL block/rate-limit you if you hit them too
fast across ~165 races. This script:
  - waits longer between requests than before (2.5s, was 0.5s)
  - retries a failed race up to 3 times with increasing backoff
  - is RESUMABLE: if it dies partway (or gets blocked), rerunning the
    script skips races already saved in weather.csv and only fetches
    what's still missing. So if you get blocked again, just wait a
    while (10-15 min) and rerun - you won't lose progress or have to
    start over.

Output:
    f1_data/weather.csv - one row per (season, round) with weather summary
"""

import fastf1
import pandas as pd
import os
import time

DATA_DIR = "f1_data"
CACHE_DIR = "f1_cache"
SEASONS = list(range(2018, 2026))  # 2018-2025, matches our other data
DELAY_BETWEEN_REQUESTS = 2.5  # seconds - be conservative, F1's servers rate-limit
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5  # seconds - first retry waits 5s, second 10s, third 20s


def setup():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, DATA_DIR)
    cache_dir = os.path.join(script_dir, CACHE_DIR)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    return data_dir


def load_existing(out_path):
    """
    Load any previously-saved weather.csv so we can resume instead of
    re-fetching races we already have.

    IMPORTANT: only counts a row as "done" if it actually has real data.
    Earlier versions of this script could write rows full of None when a
    fetch failed - if we treated those as "done" too, we'd permanently
    skip retrying them forever. So a saved row only counts if its
    air_temp_mean is not null.

    Returns (list_of_row_dicts, set_of_done_keys).
    """
    if not os.path.exists(out_path):
        return [], set()
    existing = pd.read_csv(out_path)

    has_real_data = existing["air_temp_mean"].notna()
    good_rows = existing[has_real_data]
    bad_rows = existing[~has_real_data]

    if len(bad_rows) > 0:
        print(f"Found {len(bad_rows)} previously-saved rows with no real data "
              f"(from an old/failed run) - these will be retried, not skipped:")
        for _, r in bad_rows.iterrows():
            print(f"    {r['season']} round {r['round']}: {r['race_name']}")

    rows = good_rows.to_dict("records")
    done_keys = set(zip(good_rows["season"], good_rows["round"]))
    return rows, done_keys


def summarize_weather(weather_df, season, round_num, race_name):
    """Collapse minute-by-minute weather samples into one summary row."""
    rainfall_col = weather_df["Rainfall"].astype(bool)
    return {
        "season": season,
        "round": round_num,
        "race_name": race_name,
        "air_temp_mean": round(weather_df["AirTemp"].mean(), 1),
        "air_temp_max": round(weather_df["AirTemp"].max(), 1),
        "track_temp_mean": round(weather_df["TrackTemp"].mean(), 1),
        "track_temp_max": round(weather_df["TrackTemp"].max(), 1),
        "humidity_mean": round(weather_df["Humidity"].mean(), 1),
        "wind_speed_mean": round(weather_df["WindSpeed"].mean(), 1),
        "rained_during_race": bool(rainfall_col.any()),
        # what fraction of the weather samples showed rain -
        # gives a sense of "light drizzle for 5 min" vs "rained the whole race"
        "rain_fraction_of_race": round(rainfall_col.mean(), 2),
    }


def fetch_one_race(season, round_num):
    """
    Try to load weather for one race, retrying with backoff on failure.
    Raises the last exception if all retries are exhausted (caller decides
    whether to skip and move on).
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            session = fastf1.get_session(season, round_num, "R")
            session.load(laps=False, telemetry=False, weather=True, messages=False)
            weather_df = session.weather_data
            if weather_df is None or weather_df.empty:
                raise ValueError("weather_data came back empty")
            return weather_df
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))  # 5s, 10s, 20s
                print(f" retry {attempt}/{MAX_RETRIES} in {wait}s ({e})", end="")
                time.sleep(wait)
    raise last_error


def check_for_suspicious_duplicates(df):
    """
    Sanity check: flag any two races with IDENTICAL weather values across
    every single column. Real weather essentially never matches exactly
    between two different races/circuits, so an exact match almost always
    means a caching or indexing bug (e.g. accidentally reusing the
    previous race's session object), not a real coincidence.
    """
    weather_cols = ["air_temp_mean", "air_temp_max", "track_temp_mean",
                     "track_temp_max", "humidity_mean", "wind_speed_mean",
                     "rained_during_race", "rain_fraction_of_race"]
    dupes = df[df.duplicated(subset=weather_cols, keep=False)]
    if len(dupes) > 0:
        print(f"\n[!] WARNING: {len(dupes)} rows have IDENTICAL weather values "
              f"to another row - this is almost certainly a bug, not a real")
        print("    coincidence (different races essentially never have exactly")
        print("    matching weather down to every decimal). Check these manually:")
        print(dupes[["season", "round", "race_name"] + weather_cols].to_string())
        print("\n    Recommended fix: delete these rows from weather.csv and")
        print("    rerun this script - it will only re-fetch what's missing.")
    else:
        print("\n[OK] No suspicious duplicate weather rows found.")


def main():
    data_dir = setup()
    out_path = os.path.join(data_dir, "weather.csv")

    rows, done_keys = load_existing(out_path)
    if done_keys:
        print(f"Resuming: {len(done_keys)} races already saved with real data, skipping those.\n")

    skipped_failures = []

    for season in SEASONS:
        print(f"\n=== Season {season} ===")
        try:
            schedule = fastf1.get_event_schedule(season)
        except Exception as e:
            print(f"  [!] couldn't load schedule for {season}: {e}")
            print(f"      (will need to rerun this script later to pick up {season})")
            continue

        race_rounds = schedule[schedule["RoundNumber"] > 0]  # round 0 = testing, skip

        for _, event in race_rounds.iterrows():
            round_num = int(event["RoundNumber"])
            race_name = event["EventName"]

            if (season, round_num) in done_keys:
                continue  # already have this one from a previous run

            print(f"  Round {round_num}: {race_name} ...", end="")

            try:
                weather_df = fetch_one_race(season, round_num)
                summary = summarize_weather(weather_df, season, round_num, race_name)
                rows.append(summary)
                rain_note = "RAIN" if summary["rained_during_race"] else "dry"
                print(f" OK ({rain_note})")
            except Exception as e:
                print(f" SKIPPED after retries: {e}")
                skipped_failures.append((season, round_num, race_name))
                # Deliberately NOT adding a row here. A missing row means
                # "we never successfully got this" - rerunning the script
                # will retry it. A row full of None would look identical to
                # "we checked and there's no data", which is misleading.

            time.sleep(DELAY_BETWEEN_REQUESTS)

        # Save progress after every season - don't lose everything if
        # something crashes or gets blocked partway through.
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  [saved progress: {len(rows)} race rows total so far]")

    final_df = pd.DataFrame(rows)
    print(f"\nDone! Saved to {out_path} ({len(final_df)} rows)")
    check_for_suspicious_duplicates(final_df)

    if skipped_failures:
        print(f"\n{len(skipped_failures)} races still missing after retries:")
        for s, r, n in skipped_failures:
            print(f"  {s} round {r}: {n}")
        print("\nThis is likely F1's servers rate-limiting you. Wait 10-15 min,")
        print("then just rerun this script - it will skip everything already")
        print("saved and only retry what's missing.")


if __name__ == "__main__":
    main()