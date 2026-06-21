"""
Engine Manufacturer Lookup — generates engines.csv

WHY THIS EXISTS AS A SEPARATE SCRIPT:
Jolpica's API gives us the constructor (team), e.g. "Red Bull" or "Williams",
but NOT who makes their engine. Engine supplier matters a lot for performance
and it changes between seasons (e.g. Red Bull went from full Honda, to a
Honda-derived "Red Bull Powertrains" unit, to their own engine in 2026).

No free API tracks this cleanly per season, so this is a hand-maintained
mapping, checked against engine-history sources. Run this once to generate
engines.csv, then join it onto results.csv on (season, constructor_id).

Sources checked: Wikipedia "Formula One engines" / "Honda in Formula One",
motorsport.com and planetf1.com engine-history articles (2026).

If a constructor/season combo isn't listed here, double check it before
assuming - team/engine deals do change mid-era - and add it.

Usage:
    python build_engine_lookup.py
"""

import pandas as pd

# (season, constructor_id) -> engine_manufacturer
# constructor_id values match Jolpica's constructorId field
ENGINE_MAP = {}

def add(years, constructor_id, engine):
    for y in years:
        ENGINE_MAP[(y, constructor_id)] = engine

# --- Mercedes-powered ---
add(range(2018, 2027), "mercedes", "Mercedes")
add(range(2018, 2027), "williams", "Mercedes")
add([2018], "force_india", "Mercedes")                 # team's 2018 name before Racing Point buyout
add([2019, 2020], "racing_point", "Mercedes")           # renamed from Force India mid-2018
add(range(2021, 2026), "aston_martin", "Mercedes")      # racing_point rebranded to aston_martin in 2021
add([2026], "aston_martin", "Honda")                    # switches to Honda power for 2026

# --- Renault / Alpine-powered ---
add([2018, 2019, 2020], "renault", "Renault")
add(range(2021, 2027), "alpine", "Renault")            # renault team rebranded to alpine in 2021

# --- Ferrari-powered ---
add(range(2018, 2027), "ferrari", "Ferrari")
add(range(2018, 2027), "haas", "Ferrari")
add(range(2019, 2024), "alfa", "Ferrari")              # Sauber ran as "Alfa Romeo" branding 2019-2023
add([2018], "sauber", "Ferrari")                       # Sauber pre Alfa-Romeo branding
add([2024, 2025], "sauber", "Ferrari")                 # reverted to Sauber name in 2024
add([2026], "cadillac", "Ferrari")                     # Cadillac's debut season runs Ferrari customer engines

# --- Honda / Red Bull Powertrains-powered ---
add([2018], "red_bull", "Renault")                     # Red Bull ran Renault (badged TAG Heuer) through 2018
add([2018, 2019], "toro_rosso", "Honda")
add([2019, 2020, 2021], "red_bull", "Honda")
add([2020, 2021, 2022, 2023], "alphatauri", "Honda")   # toro_rosso renamed alphatauri in 2020; Honda-branded through 2023
add(range(2022, 2027), "red_bull", "Honda RBPT")       # Honda IP transferred to Red Bull Powertrains from 2022
add(range(2024, 2027), "rb", "Honda RBPT")             # alphatauri renamed to "RB" starting 2024

# --- McLaren: Renault until 2020, Mercedes from 2021 onward ---
add([2018, 2019, 2020], "mclaren", "Renault")
add(range(2021, 2027), "mclaren", "Mercedes")

# --- Audi: new works team for 2026, replacing Sauber (own engine program) ---
add([2026], "audi", "Audi")


def build_lookup():
    rows = []
    for (season, constructor_id), engine in ENGINE_MAP.items():
        if not engine:
            continue
        rows.append({
            "season": season,
            "constructor_id": constructor_id,
            "engine_manufacturer": engine,
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["season", "constructor_id"], keep="last")
    df = df.sort_values(["season", "constructor_id"]).reset_index(drop=True)
    return df


def verify_against_results(engines_df, results_csv_path="f1_data/results.csv"):
    """
    IMPORTANT: constructor_id strings here (e.g. "alfa", "alphatauri", "rb")
    are best-effort guesses at Jolpica's actual ID format. This function
    checks them against your real results.csv and flags anything that
    doesn't match, so you don't silently end up with blank engine data.
    """
    import os
    if not os.path.exists(results_csv_path):
        print(f"\n[!] Skipping verification - {results_csv_path} not found yet.")
        print("    Run pull_f1_data.py first, then rerun this script to verify IDs.")
        return

    results = pd.read_csv(results_csv_path)
    real_pairs = set(zip(results["season"], results["constructor_id"]))
    mapped_pairs = set(zip(engines_df["season"], engines_df["constructor_id"]))

    missing = real_pairs - mapped_pairs       # real combos with no engine mapped
    unused = mapped_pairs - real_pairs         # mapped combos that don't exist in real data (likely wrong ID string)

    if missing:
        print(f"\n[!] {len(missing)} (season, constructor_id) combos in results.csv have NO engine mapped:")
        for s, c in sorted(missing):
            print(f"    {s}: '{c}'")
    if unused:
        print(f"\n[!] {len(unused)} mapped combos don't appear in results.csv (likely wrong constructor_id spelling):")
        for s, c in sorted(unused):
            print(f"    {s}: '{c}'")
    if not missing and not unused:
        print("\n[OK] Every constructor_id in engines.csv matches results.csv exactly.")


if __name__ == "__main__":
    df = build_lookup()
    df.to_csv("f1_data/engines.csv", index=False)
    print(f"Saved {len(df)} rows to f1_data/engines.csv")
    print(df.to_string())
    verify_against_results(df)