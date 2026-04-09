"""Predict feature inputs used by Modules.week_predictions.get_predictions.

This script reproduces the feature-forecasting step for the period
2024-01-01 00:00:00 to 2025-12-31 23:00:00 and saves one CSV per zone
(DK1 and DK2) in the Data folder.
"""

from __future__ import annotations

import sys
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd


START = pd.Timestamp("2024-01-01 00:00:00")
END = pd.Timestamp("2025-12-31 23:00:00")
FORECAST_HORIZON = 168
USER = "Nikolaj"
SOURCE_FILE = "combined_data_cleaned_v5.csv"
OUTPUT_TEMPLATE = "feature_predictions_{zone}_2024-2025.csv"


def format_elapsed_minutes(start_time: float) -> str:
    return f"{(time.perf_counter() - start_time) / 60:.1f} minutes running"


def get_speciale_kode_root() -> Path:
    """Resolve Delte scripts/Speciale_Kode from the repository root."""
    speciale_kode_root = Path(__file__).resolve().parents[1]
    if not speciale_kode_root.exists():
        raise FileNotFoundError(f"Could not find Speciale_Kode at: {speciale_kode_root}")
    return speciale_kode_root


def build_feature_blocks(
    data: pd.DataFrame,
    dk_zone: str,
    wp,
    rf_models: dict,
    script_start_time: float,
) -> pd.DataFrame:
    """Rebuild the same exogenous feature blocks that get_predictions() constructs."""

    data = data.copy().sort_values("Time").reset_index(drop=True)
    data.attrs["dk_zone"] = dk_zone

    target_col = data.columns[0]
    feature_columns = [col for col in data.columns[1:] if col != "Time"]
    wp._categorize_feature_columns(feature_columns, target_col)

    horizon_mask = (data["Time"] >= START) & (data["Time"] <= END)
    horizon_hours = data.loc[horizon_mask, "Time"].tolist()
    if not horizon_hours:
        raise ValueError(f"No timestamps found between {START} and {END} for {dk_zone}.")

    total_weeks = math.ceil(len(horizon_hours) / FORECAST_HORIZON)

    simulated_history = data.loc[data["Time"] < START].copy().reset_index(drop=True)
    if simulated_history.empty:
        raise ValueError(f"Need historical data before {START} to build feature forecasts for {dk_zone}.")

    rf_enabled = rf_models is not None and bool(rf_models)
    cache_key = wp._build_feature_cache_key(
        data=data,
        val_start=START,
        val_end=END,
        forecast_horizon=FORECAST_HORIZON,
        dk_zone=dk_zone,
        feature_columns=feature_columns,
        rf_enabled=rf_enabled,
    )

    cached_blocks = wp._FORECAST_FEATURE_BLOCK_CACHE.get(cache_key)
    if cached_blocks is None:
        cached_blocks = {}
        wp._FORECAST_FEATURE_BLOCK_CACHE[cache_key] = cached_blocks

    present_forecast_features = [
        feat for feat in wp.FORECASTABLE_FEATURES if feat in data.columns and feat != target_col
    ]
    non_rf_features = [
        feat
        for feat in present_forecast_features
        if wp.FORECAST_METHODS[dk_zone].get(feat) in {"lag24", "lag168", "last_known_value", "skforecast"}
    ]
    rf_features = [
        feat for feat in present_forecast_features if wp.FORECAST_METHODS[dk_zone].get(feat) == "rf"
    ]

    block_predictions: dict[int, pd.DataFrame] = {}
    block_no = 1
    block_start_idx = 0

    while block_start_idx < len(horizon_hours):
        block_hours = horizon_hours[block_start_idx : block_start_idx + FORECAST_HORIZON]
        block_true = data.loc[data["Time"].isin(block_hours)].copy().reset_index(drop=True)

        if block_no in cached_blocks:
            block_df = cached_blocks[block_no].copy()
        else:
            print(
                f"Forecasting week {block_no}/{total_weeks} for {dk_zone} | {format_elapsed_minutes(script_start_time)}"
            )
            block_df = pd.DataFrame({"Time": block_true["Time"]})
            block_df["DKZone"] = dk_zone

            for col in wp.KNOWN_FUTURE_FEATURES + wp.WEATHER_FEATURES + wp.CAPACITY_FEATURES:
                if col in block_true.columns:
                    block_df[col] = block_true[col].to_numpy()

            for feature_name in non_rf_features:
                method = wp.FORECAST_METHODS[dk_zone][feature_name]
                if method == "lag24":
                    block_df[feature_name] = wp._forecast_lag_feature_for_week(
                        simulated_history, block_df, feature_name, 24
                    )
                elif method == "lag168":
                    block_df[feature_name] = wp._forecast_lag_feature_for_week(
                        simulated_history, block_df, feature_name, 168
                    )
                elif method == "last_known_value":
                    block_df[feature_name] = wp._forecast_last_known_value_for_week(
                        simulated_history, feature_name, len(block_df)
                    )
                elif method == "skforecast":
                    block_df[feature_name] = wp._forecast_skforecast_feature_for_week(
                        simulated_history,
                        block_df,
                        feature_name,
                        offset=168,
                        n_offsets=3,
                        agg_func=np.mean,
                    )
                else:
                    raise ValueError(f"Unsupported forecast method '{method}' for {feature_name}.")

            for feature_index, feature_name in enumerate(rf_features, start=1):
                if not rf_models:
                    raise ValueError(
                        f"RF models are required to forecast '{feature_name}' for {dk_zone}."
                    )
                block_df[feature_name] = wp._forecast_rf_feature_for_week(
                    simulated_history,
                    block_df,
                    feature_name,
                    dk_zone,
                    rf_models,
                )

            ordered_columns = ["Time", "DKZone"]
            ordered_columns += [col for col in wp.KNOWN_FUTURE_FEATURES if col in block_df.columns]
            ordered_columns += [col for col in wp.WEATHER_FEATURES if col in block_df.columns]
            ordered_columns += [col for col in wp.CAPACITY_FEATURES if col in block_df.columns]
            ordered_columns += [col for col in present_forecast_features if col in block_df.columns]
            ordered_columns += [col for col in block_df.columns if col not in ordered_columns]
            block_df = block_df[ordered_columns]

            cached_blocks[block_no] = block_df.copy()

        block_predictions[block_no] = block_df.copy()

        simulated_history = simulated_history.loc[:, ~simulated_history.columns.duplicated()].copy()
        history_append = block_df.loc[:, ~block_df.columns.duplicated()].copy()
        for col in simulated_history.columns:
            if col not in history_append.columns:
                history_append[col] = np.nan
        history_append = history_append[simulated_history.columns]
        simulated_history = pd.concat([simulated_history, history_append], ignore_index=True)

        block_no += 1
        block_start_idx += FORECAST_HORIZON

    return pd.concat(block_predictions.values(), ignore_index=True)


def main() -> None:
    script_start_time = time.perf_counter()
    speciale_kode_root = get_speciale_kode_root()
    if str(speciale_kode_root) not in sys.path:
        sys.path.append(str(speciale_kode_root))

    from Modules.Load_RF_forecast_models import load_rf_models
    from Modules.read_data import read_data
    from Modules import week_predictions as wp

    print("Loading RF feature forecast models...")
    rf_models = load_rf_models(user=USER)

    print(f"Reading data: {SOURCE_FILE}")
    DK1_train, DK1_test, DK2_train, DK2_test, *_ = read_data(SOURCE_FILE)

    zone_data = {
        "DK1": pd.concat([DK1_train, DK1_test], ignore_index=True),
        "DK2": pd.concat([DK2_train, DK2_test], ignore_index=True),
    }

    output_dir = speciale_kode_root / "Data"
    for zone, data in zone_data.items():
        print(f"Forecasting feature inputs for {zone} from {START} to {END}...")
        zone_predictions = build_feature_blocks(
            data=data,
            dk_zone=zone,
            wp=wp,
            rf_models=rf_models,
            script_start_time=script_start_time,
        ).sort_values("Time").reset_index(drop=True)

        output_path = output_dir / OUTPUT_TEMPLATE.format(zone=zone)
        zone_predictions.to_csv(output_path, index=False, decimal=",")

        print(f"Saved {zone} predictions: {output_path}")
        print(f"Rows: {len(zone_predictions):,}")


if __name__ == "__main__":
    main()
