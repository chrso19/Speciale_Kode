"""Compare precomputed feature forecasts against live-computed feature forecasts.

This script checks whether Data/feature_predictions_{PRICE_ZONE}_2024-2025_new.csv
matches the feature blocks produced live by Modules.week_predictions2 for the same
zone and period.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
USER = "Nikolaj"
PRICE_ZONE = "DK1"  # "DK1" or "DK2"
SOURCE_FILE = "combined_data_cleaned_v5.csv"
PRECOMPUTED_TEMPLATE = "feature_predictions_{zone}_2024-2025_new.csv"

CMP_START = pd.Timestamp("2024-01-01 00:00:00")
CMP_END = pd.Timestamp("2025-12-31 23:00:00")
FORECAST_HORIZON = 168
DIFF_TOL = 1e-6

# Keep this aligned with the setting you use in the actual validation/test calls.
USE_FORECASTED_HISTORY = True


# -----------------------------------------------------------------------------
# Project setup
# -----------------------------------------------------------------------------
def find_project_root(start: Path | None = None) -> Path:
    """Find the Speciale_Kode directory by searching upward from the script file first,
    then from the current working directory."""
    for candidate in (Path(__file__).resolve(), (start or Path.cwd()).resolve()):
        path = candidate
        while path.name != "Speciale_Kode":
            if path.parent == path:
                break
            path = path.parent
        if path.name == "Speciale_Kode":
            return path

    raise FileNotFoundError(
        "Could not find a parent folder named 'Speciale_Kode'. "
        "Run this script from inside the project, or adjust project_root manually."
    )


project_root = find_project_root()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from Modules.read_data import read_data
from Modules.Load_RF_forecast_models import load_rf_models
from Modules import week_predictions2 as wp

# Support either function name, in case you renamed get_predictions to get_predictions2.
try:
    live_get_predictions = wp.get_predictions2
except AttributeError:
    live_get_predictions = wp.get_predictions


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_zone_data(zone: str) -> pd.DataFrame:
    """Load the selected zone's full 2024-2025 comparison context."""
    if zone not in {"DK1", "DK2"}:
        raise ValueError("PRICE_ZONE must be 'DK1' or 'DK2'.")

    DK1_train, DK1_test, DK2_train, DK2_test, *_ = read_data(SOURCE_FILE)

    if zone == "DK1":
        train, test = DK1_train, DK1_test
    else:
        train, test = DK2_train, DK2_test

    full_data = (
        pd.concat([train, test], ignore_index=True)
        .sort_values("Time")
        .drop_duplicates(subset=["Time"], keep="last")
        .reset_index(drop=True)
    )

    # Ensure the target is first and Time second, matching the assumptions in week_predictions2.
    target_col = full_data.columns[0]
    full_data = full_data[
        [target_col, "Time"] + [c for c in full_data.columns if c not in (target_col, "Time")]
    ].copy()

    return full_data


def load_precomputed_csv(zone: str) -> pd.DataFrame:
    """Load the new precomputed forecast CSV for the selected zone."""
    path = project_root / "Data" / PRECOMPUTED_TEMPLATE.format(zone=zone)
    if not path.exists():
        raise FileNotFoundError(f"Precomputed CSV not found: {path}")

    csv_df = pd.read_csv(path, decimal=",", parse_dates=["Time"])
    csv_df["Time"] = pd.to_datetime(csv_df["Time"])

    if "DKZone" in csv_df.columns:
        csv_df = csv_df.loc[csv_df["DKZone"] == zone].copy()

    csv_df = (
        csv_df.loc[(csv_df["Time"] >= CMP_START) & (csv_df["Time"] <= CMP_END)]
        .sort_values("Time")
        .reset_index(drop=True)
    )

    return csv_df


class ZeroModel:
    """Dummy model. Predictions are irrelevant; only feature blocks are inspected."""

    sequence_length = 1
    use_target_history = False

    def predict(self, X):
        return np.array([0.0])


def build_live_feature_blocks(full_data: pd.DataFrame, zone: str) -> pd.DataFrame:
    """Run live feature construction and collect cached feature blocks."""
    print("Loading RF forecast models...")
    rf_models = load_rf_models(user=USER)

    wp.clear_forecast_feature_cache()

    print("Building live feature forecasts...")
    live_get_predictions(
        model=ZeroModel(),
        dataset=full_data,
        val_start=CMP_START,
        val_end=CMP_END,
        forecast_horizon=FORECAST_HORIZON,
        fitted_scaler=None,
        dk_zone=zone,
        rf_models=rf_models,
        use_precomputed_feature_values=False,
        precomputed_feature_predictions=None,
        use_forecasted_history=USE_FORECASTED_HISTORY,
    )

    if not wp._FORECAST_FEATURE_BLOCK_CACHE:
        raise RuntimeError("No live feature blocks were cached by week_predictions2.")

    # Cache was cleared before the call, so there should be exactly one cache entry.
    cached_blocks = copy.deepcopy(next(iter(wp._FORECAST_FEATURE_BLOCK_CACHE.values())))

    live_df = (
        pd.concat(list(cached_blocks.values()), ignore_index=True)
        .sort_values("Time")
        .reset_index(drop=True)
    )

    return live_df


def compare_forecasts(live_df: pd.DataFrame, csv_df: pd.DataFrame) -> pd.DataFrame:
    """Compare forecastable feature values and print summary statistics."""
    forecast_features = [
        feature
        for feature in wp.FORECASTABLE_FEATURES
        if feature in live_df.columns
        and feature in csv_df.columns
        and feature not in wp.CAPACITY_FEATURES
    ]

    if not forecast_features:
        raise ValueError("No overlapping forecastable features found to compare.")

    merged = live_df[["Time"] + forecast_features].merge(
        csv_df[["Time"] + forecast_features].rename(
            columns={feature: f"{feature}__csv" for feature in forecast_features}
        ),
        on="Time",
        how="inner",
    )

    print(f"\n=== Feature Forecast Comparison ({PRICE_ZONE}, 2024-2025) ===")
    print(f"Live rows       : {len(live_df):,}")
    print(f"CSV rows        : {len(csv_df):,}")
    print(f"Matched rows    : {len(merged):,}")
    print(f"Features checked: {len(forecast_features)}")

    if len(merged) != len(live_df) or len(merged) != len(csv_df):
        missing_in_csv = set(live_df["Time"]) - set(csv_df["Time"])
        missing_in_live = set(csv_df["Time"]) - set(live_df["Time"])
        print("\nWARNING: Row coverage mismatch.")
        print(f"Missing in CSV : {len(missing_in_csv):,}")
        print(f"Missing in live: {len(missing_in_live):,}")

    stats = []
    for feature in forecast_features:
        live_values = merged[feature].to_numpy(dtype=float)
        csv_values = merged[f"{feature}__csv"].to_numpy(dtype=float)
        abs_diff = np.abs(live_values - csv_values)

        stats.append(
            {
                "Feature": feature,
                "N_different": int(np.sum(abs_diff > DIFF_TOL)),
                "Avg_abs_diff": float(np.mean(abs_diff)),
                "Max_abs_diff": float(np.max(abs_diff)),
            }
        )

    stats_df = (
        pd.DataFrame(stats)
        .sort_values(["N_different", "Avg_abs_diff"], ascending=[False, False])
        .reset_index(drop=True)
    )

    width = 36
    print(f"\n  {'Feature':<{width}} {'N rows diff':>11} {'Avg |diff|':>12} {'Max |diff|':>12}")
    print("  " + "-" * (width + 37))
    for _, row in stats_df.iterrows():
        print(
            f"  {row['Feature']:<{width}} "
            f"{int(row['N_different']):>11d} "
            f"{row['Avg_abs_diff']:>12.6f} "
            f"{row['Max_abs_diff']:>12.6f}"
        )

    n_diff_features = int((stats_df["N_different"] > 0).sum())
    print()
    if n_diff_features == 0:
        print(f"All values match within tolerance ({DIFF_TOL}).")
    else:
        print(f"{n_diff_features} feature(s) have at least one differing row.")

    return stats_df


def main() -> None:
    print(f"Project root: {project_root}")
    print(f"Zone        : {PRICE_ZONE}")
    print(f"Period      : {CMP_START} -> {CMP_END}")

    csv_df = load_precomputed_csv(PRICE_ZONE)
    print(f"Loaded precomputed CSV: {len(csv_df):,} rows, {len(csv_df.columns)} columns")

    full_data = load_zone_data(PRICE_ZONE)
    print(f"Loaded full zone data : {len(full_data):,} rows, {len(full_data.columns)} columns")

    live_df = build_live_feature_blocks(full_data, PRICE_ZONE)
    print(f"Live feature blocks   : {len(live_df):,} rows, {len(live_df.columns)} columns")

    compare_forecasts(live_df, csv_df)


if __name__ == "__main__":
    main()
