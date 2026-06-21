"""
Predict the 2026 Austrian Grand Prix (Round 8) using real 2026 season-to-date
data (rounds 1-7) for rolling features, fed through our trained LightGBM
ranking model.

IMPORTANT LIMITATION, stated explicitly: we don't know Austria's actual
qualifying positions yet (the session hasn't happened as of this writing).
So this prediction uses each driver's PRE-RACE feature baseline (their
most recent real rolling form, carried forward) rather than true
race-specific inputs like grid position. This means:
    - We CAN predict relative driver/constructor strength heading into
      the race, based on real current-season form.
    - We CANNOT predict the exact qualifying-dependent grid order, since
      that genuinely doesn't exist yet.

To approximate qualifying_position/grid for this prediction, we use each
driver's average qualifying position from the 2026 season so far as a
reasonable proxy - not a guess, but also not the real Austria-specific
number, which we'll only know after Saturday's qualifying session.

Output: ranked prediction for Austria 2026, with this limitation stated
again in the printed output so it's never presented as more certain than
it is.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from train_model import FEATURE_COLS, CATEGORICAL_FEATURES

DATA_DIR = "f1_data"


def build_austria_prediction_rows():
    master = pd.read_csv(f"{DATA_DIR}/master_with_features.csv")
    season_2026 = master[master["season"] == 2026].sort_values(["driver_id", "round"])

    rows = []
    for driver_id, driver_df in season_2026.groupby("driver_id"):
        latest = driver_df.iloc[-1]  # most recent race (round 7, Barcelona, for everyone still active)

        avg_quali_2026 = driver_df["qualifying_position"].mean()

        row = {
            "driver_id": driver_id,
            "driver_name": latest["driver_name"],
            "constructor_id": latest["constructor_id"],
            "grid": avg_quali_2026,                  # proxy - real grid unknown until Saturday
            "qualifying_position": avg_quali_2026,    # same proxy
            "lat": latest["lat"], "long": latest["long"],
            "engine_tier": latest["engine_tier"],
            "engine_manufacturer": latest["engine_manufacturer"],
            "country": latest["country"],
            # weather genuinely unknown for a race 6 days out - leave NaN,
            # model handles missing values natively
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
            "q1_time_sec": np.nan, "q2_time_sec": np.nan, "q3_time_sec": np.nan,  # unknown until Saturday
        }
        rows.append(row)

    return pd.DataFrame(rows)


def predict_austria(df, model_path=f"{DATA_DIR}/f1_model.txt"):
    model = lgb.Booster(model_file=model_path)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    scores = model.predict(df[FEATURE_COLS])
    df["model_score"] = scores
    df["predicted_rank"] = df["model_score"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("predicted_rank")


if __name__ == "__main__":
    print("=" * 70)
    print("2026 AUSTRIAN GRAND PRIX - PRE-RACE PREDICTION")
    print("=" * 70)
    print("""
LIMITATION: Austria qualifying hasn't happened yet (as of this writing).
This prediction uses each driver's 2026 season-to-date average qualifying
position as a PROXY for grid position, combined with their real current
rolling form (rounds 1-7). It reflects "who looks strongest heading into
Austria based on real 2026 form" - NOT a true grid-aware prediction,
which would require Saturday's actual qualifying results.
""")

    df = build_austria_prediction_rows()
    ranked = predict_austria(df)

    print(f"{'Rank':<5}{'Driver':<22}{'Team':<15}{'Avg Quali 2026':<16}{'Form (L5)':<12}")
    print("-" * 70)
    for _, row in ranked.head(10).iterrows():
        print(f"{row['predicted_rank']:<5}{row['driver_name']:<22}{row['constructor_id']:<15}"
              f"{row['grid']:<16.1f}{row['driver_form_last5']:<12.2f}")