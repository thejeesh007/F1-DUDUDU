"""
Error analysis for the F1 race result predictor.

Goal: understand WHERE and WHY the model gets things wrong, not just
"what's the overall accuracy". This is broken into several angles:

    1. Worst-predicted races overall - which specific races did we botch?
    2. Worst-predicted drivers - is error concentrated in certain drivers
       (e.g. inconsistent midfield drivers) vs spread evenly?
    3. Performance by race characteristic - wet vs dry, by circuit type -
       does the model struggle specifically with rain races, as we might
       expect given the smaller wet-weather sample size in training?
    4. Calibration check - are our podium PREDICTIONS systematically
       biased toward certain drivers/teams (e.g. always predicting the
       reigning champion even when form suggests otherwise)?

All analysis runs on the TEST set only (2024-2025) - we want to
understand real generalization error, not training fit.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from train_model import FEATURE_COLS, CATEGORICAL_FEATURES, predict
from baseline_model import load_split, convert_rank_to_position, spearman_per_race

DATA_DIR = "f1_data"


def load_model_and_test():
    model = lgb.Booster(model_file=f"{DATA_DIR}/f1_model.txt")
    _, test = load_split()
    test_scored = predict(model, test)
    test_scored = convert_rank_to_position(test_scored, "model_pred")
    test_scored["position_error"] = (
        test_scored["_predicted_position"] - test_scored["finish_position"]
    )
    test_scored["abs_error"] = test_scored["position_error"].abs()
    return test_scored


def worst_races(df, n=10):
    print(f"=== TOP {n} WORST-PREDICTED RACES (by avg absolute error) ===\n")
    race_errors = (
        df.groupby(["season", "round", "race_name"])["abs_error"]
        .mean()
        .sort_values(ascending=False)
        .head(n)
    )
    for (season, rnd, name), err in race_errors.items():
        rained = df[(df["season"] == season) & (df["round"] == rnd)]["rained_during_race"].iloc[0]
        print(f"  {season} R{rnd} {name:30s} avg error: {err:.2f} positions  (rain: {rained})")
    print()
    return race_errors


def worst_drivers(df, n=10, min_races=5):
    print(f"=== TOP {n} WORST-PREDICTED DRIVERS (avg abs error, min {min_races} races) ===\n")
    driver_errors = (
        df.groupby("driver_id")
        .agg(avg_error=("abs_error", "mean"), n_races=("abs_error", "count"))
        .query(f"n_races >= {min_races}")
        .sort_values("avg_error", ascending=False)
        .head(n)
    )
    print(driver_errors.to_string())
    print()
    return driver_errors


def best_drivers(df, n=10, min_races=5):
    print(f"=== TOP {n} BEST-PREDICTED DRIVERS (avg abs error, min {min_races} races) ===\n")
    driver_errors = (
        df.groupby("driver_id")
        .agg(avg_error=("abs_error", "mean"), n_races=("abs_error", "count"))
        .query(f"n_races >= {min_races}")
        .sort_values("avg_error", ascending=True)
        .head(n)
    )
    print(driver_errors.to_string())
    print()
    return driver_errors


def wet_vs_dry_performance(df):
    print("=== WET vs DRY RACE PERFORMANCE ===\n")
    for condition, label in [(True, "WET races"), (False, "DRY races")]:
        subset = df[df["rained_during_race"] == condition]
        if len(subset) == 0:
            continue
        spearman, n = spearman_per_race(subset, "model_pred")
        mae = subset["abs_error"].mean()
        n_races = subset[["season", "round"]].drop_duplicates().shape[0]
        print(f"  {label}: {n_races} races, Spearman={spearman:.3f}, MAE={mae:.2f}")
    print()


def podium_prediction_bias(df):
    """
    Checks whether the model OVER-predicts or UNDER-predicts certain
    drivers/teams for the podium - i.e. does it predict someone for
    podium more often than they actually achieve it (overconfidence),
    or the reverse (underrating them)?
    """
    print("=== PODIUM PREDICTION BIAS (test set) ===\n")
    rows = []
    for (season, rnd), race_df in df.groupby(["season", "round"]):
        predicted_top3 = set(race_df.nsmallest(3, "model_pred")["driver_id"])
        actual_top3 = set(race_df.nsmallest(3, "finish_position")["driver_id"])
        for driver in predicted_top3:
            rows.append({"driver_id": driver, "predicted_podium": True,
                         "actual_podium": driver in actual_top3})

    bias_df = pd.DataFrame(rows)
    summary = (
        bias_df.groupby("driver_id")
        .agg(times_predicted_podium=("predicted_podium", "sum"),
             times_actually_podiumed=("actual_podium", "sum"))
        .assign(hit_rate=lambda d: d["times_actually_podiumed"] / d["times_predicted_podium"])
        .sort_values("times_predicted_podium", ascending=False)
    )
    print("Drivers most often PREDICTED for podium, and how often they actually delivered:")
    print(summary.head(12).to_string())
    print()

    overrated = summary[summary["times_predicted_podium"] >= 5].sort_values("hit_rate").head(5)
    print("Most OVER-predicted (predicted podium often, rarely delivered):")
    print(overrated.to_string())
    print()


if __name__ == "__main__":
    df = load_model_and_test()

    worst_races(df)
    worst_drivers(df)
    best_drivers(df)
    wet_vs_dry_performance(df)
    podium_prediction_bias(df)