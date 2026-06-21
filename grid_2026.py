"""
2026 grid setup + cold-start feature handling for the F1 race predictor.

Built from researched 2026 driver/team lineup (confirmed as of Dec 2025,
cross-checked against BBC Sport, RaceFans, ESPN, Silverstone.co.uk).

CATEGORIES OF "PROBLEM" DRIVER/CONSTRUCTOR SITUATIONS FOR 2026, AND HOW
EACH IS HANDLED (this maps directly to the error-analysis findings from
the 2024-2025 test set, where rookies and driver-team transitions showed
the highest prediction error):

1. TRUE ROOKIE, ZERO F1 HISTORY -> Lindblad (Racing Bulls)
   No driver-level rolling features exist. We substitute a ROOKIE PRIOR
   computed from real historical rookie debut performance in our dataset
   (43 rookie debuts, 2018-2025): mean finish ~12.1, DNF rate ~21%.

2. PARTIAL/FULL HISTORY ALREADY IN OUR DATA -> Antonelli, Colapinto, Hadjar
   These already have real 2025 rows in master_with_features.csv. No
   special handling needed - standard rolling features work as normal.

3. DRIVER-TEAM TRANSITION (the Hamilton/Ferrari case from error analysis)
   By 2026, Hamilton will have a FULL 2025 season at Ferrari in our data,
   so driver_track_history etc. will correctly reflect Ferrari-specific
   performance, not stale Mercedes-era history. This mostly self-resolves
   with time - flagged here as a known limitation rather than over-engineered.

4. RETURNING VETERAN AFTER ABSENCE -> Perez (absent 2025), Bottas (absent 2025)
   Real history exists but has a gap. We add a `seasons_since_last_race`
   feature so the model can learn that long-gap returns carry more
   uncertainty, rather than treating 2024 data as if it were as fresh as
   2025 data.

5. NEW CONSTRUCTOR ID -> Cadillac (genuinely new), Audi (rebrand of Sauber)
   - Cadillac: zero history. We use a CONSERVATIVE new-team prior, NOT
     based on Haas's 2016 debut alone - that season was a historical
     outlier (first team to score points on debut since 2002), and using
     a sample size of 1 outlier would badly bias the estimate. Instead we
     use a conservative prior similar to the rookie-driver prior (assume
     midfield-to-back, not "second coming of Haas").
   - Audi: this is Sauber renamed, with driver/personnel continuity. We
     MAP Audi's constructor_id to inherit Sauber's historical rows so
     constructor-level features aren't artificially blanked out.

6. ENGINE SUPPLIER CHANGE -> Aston Martin moving to Honda for 2026
   Simple data update: engines.csv needs a 2026 entry reflecting this.

Output:
    f1_data/grid_2026.csv - the 2026 driver/constructor/engine mapping
"""

import pandas as pd
import os

DATA_DIR = "f1_data"

# Confirmed 2026 lineup (as of Dec 2025) - constructor_id values chosen to
# either match existing IDs in our data (continuity) or introduce new ones
# (genuinely new entities), per the category logic above.
GRID_2026 = [
    # (driver_id matching our existing data where applicable, driver_name, constructor_id, situation_category)
    {"driver_id": "piastri", "driver_name": "Oscar Piastri", "constructor_id": "mclaren", "category": "normal"},
    {"driver_id": "norris", "driver_name": "Lando Norris", "constructor_id": "mclaren", "category": "normal"},

    {"driver_id": "hamilton", "driver_name": "Lewis Hamilton", "constructor_id": "ferrari", "category": "transition_now_settled"},
    {"driver_id": "leclerc", "driver_name": "Charles Leclerc", "constructor_id": "ferrari", "category": "normal"},

    {"driver_id": "max_verstappen", "driver_name": "Max Verstappen", "constructor_id": "red_bull", "category": "normal"},
    {"driver_id": "hadjar", "driver_name": "Isack Hadjar", "constructor_id": "red_bull", "category": "partial_history"},

    {"driver_id": "russell", "driver_name": "George Russell", "constructor_id": "mercedes", "category": "normal"},
    {"driver_id": "antonelli", "driver_name": "Kimi Antonelli", "constructor_id": "mercedes", "category": "partial_history"},

    {"driver_id": "albon", "driver_name": "Alex Albon", "constructor_id": "williams", "category": "normal"},
    {"driver_id": "sainz", "driver_name": "Carlos Sainz", "constructor_id": "williams", "category": "normal"},

    {"driver_id": "hulkenberg", "driver_name": "Nico Hulkenberg", "constructor_id": "audi", "category": "constructor_rebrand"},
    {"driver_id": "bortoleto", "driver_name": "Gabriel Bortoleto", "constructor_id": "audi", "category": "constructor_rebrand"},

    {"driver_id": "alonso", "driver_name": "Fernando Alonso", "constructor_id": "aston_martin", "category": "engine_change"},
    {"driver_id": "stroll", "driver_name": "Lance Stroll", "constructor_id": "aston_martin", "category": "engine_change"},

    {"driver_id": "gasly", "driver_name": "Pierre Gasly", "constructor_id": "alpine", "category": "normal"},
    {"driver_id": "colapinto", "driver_name": "Franco Colapinto", "constructor_id": "alpine", "category": "partial_history"},

    {"driver_id": "ocon", "driver_name": "Esteban Ocon", "constructor_id": "haas", "category": "normal"},
    {"driver_id": "bearman", "driver_name": "Oliver Bearman", "constructor_id": "haas", "category": "normal"},

    {"driver_id": "lawson", "driver_name": "Liam Lawson", "constructor_id": "rb", "category": "normal"},
    {"driver_id": "lindblad", "driver_name": "Arvid Lindblad", "constructor_id": "rb", "category": "true_rookie"},

    {"driver_id": "perez", "driver_name": "Sergio Perez", "constructor_id": "cadillac", "category": "returning_veteran_new_team"},
    {"driver_id": "bottas", "driver_name": "Valtteri Bottas", "constructor_id": "cadillac", "category": "returning_veteran_new_team"},
]

# Constructor ID continuity mapping: for constructors that are renames of
# an existing team (not genuinely new), map the new ID to the old one so
# constructor-level rolling features inherit real history instead of
# being blanked out.
CONSTRUCTOR_CONTINUITY_MAP = {
    "audi": "sauber",  # Audi 2026 = Sauber renamed, same personnel/continuity
}

# 2026 engine manufacturer updates (only entries that changed from 2025)
ENGINE_UPDATES_2026 = {
    "aston_martin": "Honda",   # switching from Mercedes to Honda power for 2026
    "cadillac": "Ferrari",     # Cadillac confirmed running Ferrari power units for their debut
    "audi": "Audi",            # Audi now runs its own works engine (previously Ferrari-powered as Sauber)
}

# Empirically-derived priors for cold-start cases (computed from our real
# 2018-2025 data - see build_features.py's underlying master_with_features.csv)
ROOKIE_PRIOR_FINISH = 12.1   # mean finish position on rookie debut, n=43
ROOKIE_PRIOR_DNF_RATE = 0.21  # DNF rate on rookie debut

# Conservative new-constructor prior. NOT based on Haas 2016 alone - that
# season was a genuine historical outlier (first team to score points on
# debut since 2002), so treating it as "typical" would be misleading.
# We instead assume new teams perform similarly to a typical ROOKIE
# DRIVER (slightly back of midfield), which is a more conservative and
# defensible assumption than extrapolating from a single outlier season.
NEW_CONSTRUCTOR_PRIOR_FINISH = 13.0
NEW_CONSTRUCTOR_PRIOR_DNF_RATE = 0.18


def build_grid_2026_df():
    return pd.DataFrame(GRID_2026)


def summarize_categories(df):
    print("=== 2026 GRID - SITUATION BREAKDOWN ===\n")
    for category, group in df.groupby("category"):
        print(f"{category} ({len(group)} drivers):")
        for _, row in group.iterrows():
            print(f"  {row['driver_name']} ({row['constructor_id']})")
        print()


if __name__ == "__main__":
    df = build_grid_2026_df()
    summarize_categories(df)

    out_path = os.path.join(DATA_DIR, "grid_2026.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} drivers)")