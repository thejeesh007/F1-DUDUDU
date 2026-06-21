"""
Baseline model + shared evaluation metrics for the F1 race result predictor.

BASELINE: predicted finish position = grid position (i.e. "assume qualifying
order holds throughout the race, nobody overtakes"). This sounds naive but
is actually a strong baseline in F1, since grid position correlates heavily
with finish position. Any real model needs to beat this to prove it's
adding value, not just rediscovering "grid position matters."

METRICS USED (and why):
    1. Spearman rank correlation (per race, averaged across races)
       - Measures how well we predicted the RELATIVE ORDER of drivers
         within a race, which is what actually matters for a ranking
         problem (we don't care if we said someone finishes "5.2" vs
         "5.8", we care if we got the order right).
    2. Podium accuracy (top-3 hit rate)
       - Of the 3 drivers we predicted would podium, how many actually did?
         This is the headline number most people will care about.
    3. Mean Absolute Error (MAE) on finish position
       - Simple, interpretable: "on average, how many positions off were we."
         Included mainly for intuition, not as the primary metric, since
         it doesn't directly capture ranking quality.

Output: prints baseline metrics, saves nothing (this is a reference point,
not a model artifact).
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

DATA_DIR = "f1_data"


def load_split():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    return train, test


def spearman_per_race(df, pred_col, actual_col="finish_position", race_keys=("season", "round")):
    """
    Compute Spearman rank correlation between predicted and actual finish
    order, separately for EACH race, then average across races.

    We do this per-race (not globally) because the actual task is "rank
    these ~20 drivers in THIS race correctly" - a global correlation
    across all races mixed together would be misleading, since finish
    position 5 in one race has no real relationship to finish position 5
    in a different race except its rank.
    """
    correlations = []
    for _, race_df in df.groupby(list(race_keys)):
        if race_df[pred_col].nunique() < 2 or race_df[actual_col].nunique() < 2:
            continue  # can't compute correlation with no variance
        corr, _ = spearmanr(race_df[pred_col], race_df[actual_col])
        if not np.isnan(corr):
            correlations.append(corr)
    return np.mean(correlations), len(correlations)


def podium_accuracy(df, pred_col, actual_col="finish_position", race_keys=("season", "round")):
    """
    For each race: take the 3 drivers with the BEST predicted value (lowest
    predicted finish position / highest predicted rank) and check how many
    of them actually finished in the real top 3.

    Returns average hit rate (0.0-1.0), e.g. 0.67 means we got 2 of 3
    podium spots right on average per race.
    """
    hit_rates = []
    for _, race_df in df.groupby(list(race_keys)):
        if len(race_df) < 3:
            continue
        predicted_top3 = set(race_df.nsmallest(3, pred_col)["driver_id"])
        actual_top3 = set(race_df.nsmallest(3, actual_col)["driver_id"])
        hits = len(predicted_top3 & actual_top3)
        hit_rates.append(hits / 3)
    return np.mean(hit_rates), len(hit_rates)


def mean_absolute_error_metric(df, pred_col, actual_col="finish_position"):
    """
    NOTE: this metric only makes sense when pred_col is on the SAME SCALE
    as actual finish positions (e.g. our baseline, which predicts grid
    position directly). For the LightGBM ranking model, pred_col is a
    relative rank SCORE, not a predicted position - comparing it directly
    to finish_position via absolute difference is not meaningful and will
    produce a misleadingly large number. Use convert_rank_to_position()
    first if you want a genuine MAE for a ranking model's output.
    """
    return np.mean(np.abs(df[pred_col] - df[actual_col]))


def convert_rank_to_position(df, pred_col, race_keys=("season", "round")):
    """
    Converts a prediction column into an actual predicted POSITION
    (1, 2, 3... within each race) by ranking cars within each race.

    IMPORTANT: assumes LOWER pred_col = BETTER (matching finish_position's
    own convention, where 1st place = 1, the lowest number). This is true
    for:
      - the baseline (predicted finish = grid, where grid 1 = pole = best)
      - the LightGBM model's output AFTER negation in train_model.py's
        predict() function (which deliberately flips the raw rank score
        so it follows this same "lower = better" convention)

    If you pass in a raw score where HIGHER = better (e.g. the LightGBM
    model's un-negated relevance score), this will silently invert the
    result - the negation in predict() exists specifically to avoid that.
    """
    df = df.copy()
    df["_predicted_position"] = (
        df.groupby(list(race_keys))[pred_col]
        .rank(ascending=True, method="first")
    )
    return df


def evaluate(df, pred_col, label=""):
    spearman, n_races_s = spearman_per_race(df, pred_col)
    podium, n_races_p = podium_accuracy(df, pred_col)

    # Convert to actual predicted positions first so MAE is meaningful
    # regardless of whether pred_col is a raw score or an actual position.
    df_with_pos = convert_rank_to_position(df, pred_col)
    mae = mean_absolute_error_metric(df_with_pos, "_predicted_position")

    print(f"--- {label} ---")
    print(f"  Spearman rank correlation (avg per race): {spearman:.3f}  (n={n_races_s} races)")
    print(f"  Podium accuracy (avg hit rate):           {podium:.1%}  (n={n_races_p} races)")
    print(f"  Mean Absolute Error (positions):          {mae:.2f}")
    print()
    return {"spearman": spearman, "podium_accuracy": podium, "mae": mae}


if __name__ == "__main__":
    train, test = load_split()

    print("=== BASELINE: predicted finish = grid position ===\n")
    train["baseline_pred"] = train["grid"]
    test["baseline_pred"] = test["grid"]

    print("On TRAIN set (2018-2023):")
    evaluate(train, "baseline_pred", label="Baseline (train)")

    print("On TEST set (2024-2025) - this is the number that matters:")
    evaluate(test, "baseline_pred", label="Baseline (test)")