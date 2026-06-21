"""
LightGBM ranking model for F1 race result prediction.

WHY A RANKING OBJECTIVE (lambdarank), NOT PLAIN REGRESSION:
The actual task is "order these ~20 drivers correctly within this race",
not "predict driver X's exact finish position as an absolute number".
A regression model treats "predicted 5.2 vs actual 6" and "predicted 15.2
vs actual 16" as equally good (both off by 0.8), but ranking cares about
RELATIVE order between drivers in the SAME race, which is what we
actually care about for predicting who beats whom / who podiums.

lambdarank needs each race treated as its own "group" - the model learns
to rank items WITHIN a group, not across different groups. We tell it
the group boundaries via the `group` parameter (number of rows per race,
in order).

LABEL CONSTRUCTION FOR RANKING:
lambdarank wants HIGHER label = BETTER. Our finish_position has LOWER =
better (1st place = 1). So we convert: relevance = (max_position_in_race
- finish_position + 1) - meaning whoever finished 1st gets the highest
relevance score in that race, last place gets the lowest. This is
recomputed per race since field size varies (DNFs, race-specific entries).

CATEGORICAL FEATURES:
LightGBM can handle categorical features (country, engine_manufacturer)
natively without one-hot encoding, IF we tell it which columns are
categorical and convert them to pandas 'category' dtype first.

Output:
    f1_model.txt - the trained LightGBM model file
    Prints evaluation metrics on train and test, using the same metrics
    module as the baseline for a fair side-by-side comparison.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from baseline_model import evaluate, load_split

DATA_DIR = "f1_data"

CATEGORICAL_FEATURES = ["country", "engine_manufacturer"]

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


def prepare_for_lightgbm(df):
    """
    - Sort by (season, round) so rows for the same race are contiguous -
      lambdarank's `group` parameter requires this (it just takes group
      SIZES, assuming rows are already grouped together in order).
    - Convert categorical columns to pandas 'category' dtype so LightGBM
      recognizes and handles them natively.
    - Build the ranking label (higher = better, see module docstring).
    - Build the `group` array: how many rows belong to each race, in order.
    """
    df = df.sort_values(["season", "round"]).reset_index(drop=True)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    # ranking label: within each race, 1st place gets the highest score
    df["_field_size"] = df.groupby(["season", "round"])["finish_position"].transform("max")
    df["_relevance"] = df["_field_size"] - df["finish_position"] + 1

    group_sizes = df.groupby(["season", "round"], sort=False).size().values

    return df, group_sizes


def train_model(train_df):
    """
    IMPORTANT - early stopping via an internal validation split:
    We carve the LAST season of TRAINING data (2023) out as a validation
    set, used only to decide when to stop adding more boosting rounds
    (early stopping). This is NOT the test set (2024-2025) - using the
    real test set for this would be a subtle form of leakage, since we'd
    be tuning the model based on test performance, defeating the purpose
    of having a held-out test set at all.
    """
    train_df, _ = prepare_for_lightgbm(train_df)

    # carve out 2023 as internal validation (still chronologically before
    # the real test set, so no leakage relative to 2024-2025)
    fit_df = train_df[train_df["season"] <= 2022].copy()
    val_df = train_df[train_df["season"] == 2023].copy()

    fit_group_sizes = fit_df.groupby(["season", "round"], sort=False).size().values
    val_group_sizes = val_df.groupby(["season", "round"], sort=False).size().values

    X_fit, y_fit = fit_df[FEATURE_COLS], fit_df["_relevance"]
    X_val, y_val = val_df[FEATURE_COLS], val_df["_relevance"]

    fit_data = lgb.Dataset(
        X_fit, label=y_fit, group=fit_group_sizes,
        categorical_feature=CATEGORICAL_FEATURES,
    )
    val_data = lgb.Dataset(
        X_val, label=y_val, group=val_group_sizes,
        categorical_feature=CATEGORICAL_FEATURES, reference=fit_data,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3, 5, 10],
        "learning_rate": 0.04,
        "num_leaves": 31,                # restored from 15 - smaller trees were choking out
                                          # secondary features in favor of qualifying_position alone
                                          # (confirmed via a plain-regression diagnostic showing
                                          # track_history and form features carry real signal)
        "min_data_in_leaf": 15,          # loosened from 40
        "feature_fraction": 0.8,         # loosened from 0.7
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 0.3,                # loosened from 1.0
        "verbose": -1,
    }

    model = lgb.train(
        params, fit_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    print(f"  (stopped at {model.best_iteration} boosting rounds, "
          f"out of 500 max, based on 2023 validation performance)\n")

    # Refit on the FULL training set (2018-2023, including validation
    # data) using the now-known best iteration count, so the final model
    # doesn't waste the 2023 data it used only for early stopping above.
    full_group_sizes = train_df.groupby(["season", "round"], sort=False).size().values
    full_data = lgb.Dataset(
        train_df[FEATURE_COLS], label=train_df["_relevance"], group=full_group_sizes,
        categorical_feature=CATEGORICAL_FEATURES,
    )
    final_model = lgb.train(params, full_data, num_boost_round=model.best_iteration)

    return final_model


def predict(model, df):
    """
    Returns predicted RANK SCORES (higher = better, matching training
    label convention) - NOT predicted finish positions directly. We
    negate scores so they sort the same direction as finish_position
    (lower = better) for compatibility with our existing metrics module,
    which was written assuming "lower predicted value = better".
    """
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    scores = model.predict(df[FEATURE_COLS])
    df["model_pred"] = -scores  # negate: higher rank score -> lower (better) pseudo-position
    return df


if __name__ == "__main__":
    train, test = load_split()

    print("Training LightGBM ranking model...\n")
    model = train_model(train.copy())

    train_scored = predict(model, train)
    test_scored = predict(model, test)

    print("=== MODEL EVALUATION ===\n")
    print("On TRAIN set (2018-2023):")
    evaluate(train_scored, "model_pred", label="LightGBM ranker (train)")

    print("On TEST set (2024-2025) - this is the number that matters:")
    evaluate(test_scored, "model_pred", label="LightGBM ranker (test)")

    print("=== FEATURE IMPORTANCE (top 15) ===")
    importance = pd.Series(model.feature_importance(importance_type="gain"), index=FEATURE_COLS)
    print(importance.sort_values(ascending=False).head(15))

    model.save_model(f"{DATA_DIR}/f1_model.txt")
    print(f"\nModel saved to {DATA_DIR}/f1_model.txt")