import sys
import os

# Find the project root (Speciale_Kode)
current_dir = os.getcwd()
project_root = current_dir

# Looks for "Speciale_Kode" folder:
while os.path.basename(project_root) != "Speciale_Kode":
    project_root = os.path.dirname(project_root)

# Add to Python path
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from pathlib import Path
from Modules.read_data import read_data

USER = "Nikolaj"
use_precomputed_feature_values = True
PRICE_ZONE = "DK1"  # "DK1" or "DK2"
TRAIN_WINDOW = 2 * 8760
VAL_START = "2024-01-01 00:00:00"
VAL_WINDOW = 0
PREDICT_PERIOD = 52 * 168
STRIDE = 13 * 168                # Stride is measured from the start of the previous fold.
POST_VALIDATION_EXCLUDE_HOURS = 168  # Exclude first 168h after each validation window from remainder_2024_for_train
INCLUDE_REMAINING_2024_DURING_TRAINING = True
INCLUDE_PRICE_HISTORY_AS_INPUT = True
INCLUDE_LAGS = False             # Include lag features (Price_lag1, Price_lag24, etc.) in training input
USE_FORECASTED_HISTORY = True    # Use model-predicted prices to compute lag features during prediction

(
    DK1_train,
    DK1_test,
    DK2_train,
    DK2_test,
    DK1_train_weather,
    DK1_test_weather,
    DK2_train_weather,
    DK2_test_weather
) = read_data("combined_data_cleaned_v5.csv")

if PRICE_ZONE == "DK1":
    dataset_train = DK1_train.copy()
    dataset_test = DK1_test.copy()
elif PRICE_ZONE == "DK2":
    dataset_train = DK2_train.copy()
    dataset_test = DK2_test.copy()
else:
    raise ValueError("PRICE_ZONE must be 'DK1' or 'DK2'.")

# read_data already returns all of 2024 in dataset_train and all of 2025 in dataset_test.
# Build the custom 2024 rolling validation split entirely from dataset_train.
dataset_train = dataset_train.sort_values("Time").reset_index(drop=True)
dataset_test = dataset_test.sort_values("Time").reset_index(drop=True)

val_start_ts = pd.Timestamp(VAL_START)
year_2024_start = pd.Timestamp("2024-01-01 00:00:00")
year_2025_start = pd.Timestamp("2025-01-01 00:00:00")

# Keep legacy full timeline variable before redefining dataset_train below.
df = pd.concat([dataset_train, dataset_test], ignore_index=True).sort_values("Time").reset_index(drop=True)

# Fixed history block: TRAIN_WINDOW ending at VAL_START.
history = dataset_train.loc[dataset_train["Time"] < val_start_ts].copy().iloc[-TRAIN_WINDOW:]
if len(history) < TRAIN_WINDOW:
    raise ValueError(
        f"Not enough history for TRAIN_WINDOW={TRAIN_WINDOW}. Got {len(history)} rows before {VAL_START}."
    )

data_2024 = dataset_train.loc[
    (dataset_train["Time"] >= year_2024_start) & (dataset_train["Time"] < year_2025_start)
].copy()

validation_idx = []
validation_windows = []
window_start = val_start_ts

# Validation windows in 2024: next fold starts STRIDE hours after the current fold start.
# Only full windows are allowed; trailing partial windows are skipped.
while (window_start + pd.Timedelta(hours=PREDICT_PERIOD)) <= year_2025_start:
    window_end = window_start + pd.Timedelta(hours=PREDICT_PERIOD)
    if window_end <= window_start:
        break

    mask = (data_2024["Time"] >= window_start) & (data_2024["Time"] < window_end)
    if mask.any():
        validation_idx.extend(data_2024.index[mask].tolist())
        validation_windows.append((window_start, window_end))

    window_start = window_start + pd.Timedelta(hours=STRIDE)

validation_idx = sorted(set(validation_idx))

# Exclude first POST_VALIDATION_EXCLUDE_HOURS after each validation window from train remainder.
post_validation_exclusion_idx = []
for _, window_end in validation_windows:
    exclusion_end = min(window_end + pd.Timedelta(hours=POST_VALIDATION_EXCLUDE_HOURS), year_2025_start)
    if exclusion_end <= window_end:
        continue

    exclusion_mask = (data_2024["Time"] >= window_end) & (data_2024["Time"] < exclusion_end)
    if exclusion_mask.any():
        post_validation_exclusion_idx.extend(data_2024.index[exclusion_mask].tolist())

post_validation_exclusion_idx = sorted(set(post_validation_exclusion_idx))
excluded_from_remainder_idx = sorted(set(validation_idx).union(post_validation_exclusion_idx))

dataset_validation = data_2024.loc[validation_idx].copy().sort_values("Time").reset_index(drop=True)
remainder_2024_for_train = data_2024.drop(index=excluded_from_remainder_idx).copy().sort_values("Time").reset_index(drop=True)

# Load cell is the only place that decides whether 2024 remainder is included in training.
if INCLUDE_REMAINING_2024_DURING_TRAINING:
    dataset_train = (
        pd.concat([history, remainder_2024_for_train], ignore_index=True)
        .sort_values("Time")
        .drop_duplicates(subset=["Time"], keep="last")
        .reset_index(drop=True)
    )
else:
    dataset_train = history.copy().sort_values("Time").reset_index(drop=True)

# Full context dataset: pre-2024 history + all of 2024.
# Used by get_predictions for lag computation regardless of training flags.
dataset_context = (
    pd.concat([history, data_2024], ignore_index=True)
    .sort_values("Time")
    .drop_duplicates(subset=["Time"], keep="last")
    .reset_index(drop=True)
)

# Preserve full target-bearing datasets for training/evaluation and create input views for later cells.
dataset_train_full = dataset_train.copy()
dataset_validation_full = dataset_validation.copy()
dataset_train_input = dataset_train_full.copy()
dataset_validation_input = dataset_validation_full.copy()

if not INCLUDE_PRICE_HISTORY_AS_INPUT:
    dataset_train_input = dataset_train_input.drop(columns=["DKPrice"])
    dataset_validation_input = dataset_validation_input.drop(columns=["DKPrice"])

lag_columns = [c for c in dataset_train_input.columns if '_lag' in c]
if not INCLUDE_LAGS:
    dataset_train_input = dataset_train_input.drop(columns=lag_columns, errors='ignore')
    dataset_validation_input = dataset_validation_input.drop(
        columns=[c for c in dataset_validation_input.columns if '_lag' in c], errors='ignore'
    )

# Add DKPrice_lag1 as a feature when price history is enabled.
# Computed from the full timeline (df) to handle training-set gaps correctly.
if INCLUDE_PRICE_HISTORY_AS_INPUT:
    price_lag_full = df[["Time", "DKPrice"]].copy().sort_values("Time").reset_index(drop=True)
    price_lag_full["DKPrice_lag1"] = price_lag_full["DKPrice"].shift(1)
    price_lag_full = price_lag_full[["Time", "DKPrice_lag1"]]
    dataset_train_input = dataset_train_input.merge(price_lag_full, on="Time", how="left")
    dataset_validation_input = dataset_validation_input.merge(price_lag_full, on="Time", how="left")

# Keep 2025 as test set.
dataset_test = dataset_test.copy().reset_index(drop=True)

target_time = val_start_ts
prices = history["DKPrice"].astype(float).values.reshape(-1, 1)

def _load_feature_predictions_for_zone(zone):
    prediction_path = Path(project_root) / "Data" / f"feature_predictions_{zone}_2024-2025_new.csv"
    if not prediction_path.exists():
        print("No precomputed forecasts found.")
        return None

    predictions = pd.read_csv(prediction_path, sep=";", decimal=".", parse_dates=["Time"], dayfirst=True)
    predictions = predictions.loc[:, ~predictions.columns.duplicated()].copy()
    if "DKZone" in predictions.columns:
        predictions = predictions.loc[predictions["DKZone"] == zone].copy()
        print(f"Loaded {len(predictions)} forecasts for zone {zone}.")
        print(f"Forecast features: {len(predictions.columns)} {predictions.columns.tolist()}")
    return predictions

print(f"Using zone: {PRICE_ZONE}")
print(f"Train source shape (all of 2024): {DK1_train.shape if PRICE_ZONE == 'DK1' else DK2_train.shape}")
print(f"Test source shape (all of 2025): {DK1_test.shape if PRICE_ZONE == 'DK1' else DK2_test.shape}")
print(f"Include remainder_2024_for_train in training: {INCLUDE_REMAINING_2024_DURING_TRAINING}")
print(f"Include DKPrice in training input dataset: {INCLUDE_PRICE_HISTORY_AS_INPUT}")
print(f"Include lag features in training input: {INCLUDE_LAGS}")
print(f"Use forecasted prices for lag computation during prediction: {USE_FORECASTED_HISTORY}")
print(f"Train shape (prepared training dataset): {dataset_train.shape}")
print(f"Training input shape: {dataset_train_input.shape}")
print(f"2024 remainder rows included in training: {len(remainder_2024_for_train) if INCLUDE_REMAINING_2024_DURING_TRAINING else 0}")
print(f"Validation shape (rolling 2024 windows): {dataset_validation.shape}")
print(f"Validation input shape: {dataset_validation_input.shape}")
print(f"Context shape (history + all 2024): {dataset_context.shape}")
print(f"Test shape (2025): {dataset_test.shape}")
print(f"Validation windows created: {len(validation_windows)}")
print(f"Post-validation exclusion hours: {POST_VALIDATION_EXCLUDE_HOURS}")
print(f"Rows excluded from remainder after validation windows: {len(post_validation_exclusion_idx)}")
if validation_windows:
    print("All validation windows:")
    for idx, (window_start, window_end) in enumerate(validation_windows, start=1):
        print(f"  {idx:02d}. {window_start} -> {window_end}")
print(f"Training dataset columns: {dataset_train.columns.tolist()}")
print(f"Training input columns: {dataset_train_input.columns.tolist()}")

feature_predictions = _load_feature_predictions_for_zone(PRICE_ZONE)    
if feature_predictions is not None:
    print("\nPrecomputed forecasts loaded.")

from Modules.Load_RF_forecast_models import load_rf_models

rf_models = None
if not use_precomputed_feature_values:
    # load_rf_models currently supports only the optional timeout argument.
    rf_models = load_rf_models(user=USER)      # set user to "Nikolaj" or "Christine"

import numpy as np
from Modules.week_predictions2 import get_predictions

# =============================================================================
# NEW CSV vs Live-Computed Comparison: 2024 + 2025 (full range)
# Compares feature_predictions_{PRICE_ZONE}_2024-2025_new.csv against
# get_predictions() with use_precomputed_feature_values=False.
# =============================================================================
import copy as _copy2
from Modules.week_predictions2 import (
    clear_forecast_feature_cache as _clear_cache2,
    _FORECAST_FEATURE_BLOCK_CACHE as _FEAT_CACHE2,
    FORECASTABLE_FEATURES as _FORECASTABLE2,
    CAPACITY_FEATURES as _CAPACITY2,
)

_NEW_CSV_PATH = Path(project_root) / "Data" / f"feature_predictions_{PRICE_ZONE}_2024-2025_new.csv"

if not _NEW_CSV_PATH.exists():
    print(f"\n[Comparison skipped] New CSV not found: {_NEW_CSV_PATH}")
else:
    _CMP_START2 = pd.Timestamp("2024-01-01 00:00:00")
    _CMP_END2   = pd.Timestamp("2025-12-31 23:00:00")
    _CMP_HOR2   = 168

    # Load the new CSV (saved with decimal=",", sep=",")
    _new_csv = pd.read_csv(_NEW_CSV_PATH, decimal=",", parse_dates=["Time"])
    _new_csv["Time"] = pd.to_datetime(_new_csv["Time"])
    if "DKZone" in _new_csv.columns:
        _new_csv = _new_csv.loc[_new_csv["DKZone"] == PRICE_ZONE].copy()
    _new_csv = _new_csv.sort_values("Time").reset_index(drop=True)
    print(f"\nLoaded new CSV: {len(_new_csv)} rows  ({_CMP_START2.date()} – {_CMP_END2.date()})")

    # Build full history + 2024 + 2025 dataset so block_history has true data for every block.
    if PRICE_ZONE == "DK1":
        _src_tr2, _src_te2 = DK1_train.copy(), DK1_test.copy()
    else:
        _src_tr2, _src_te2 = DK2_train.copy(), DK2_test.copy()

    _full_cmp = (
        pd.concat([_src_tr2, _src_te2], ignore_index=True)
        .sort_values("Time")
        .drop_duplicates(subset=["Time"], keep="last")
        .reset_index(drop=True)
    )
    _tgt2 = _full_cmp.columns[0]  # first column = target (e.g. DKPrice)
    _full_cmp = _full_cmp[[_tgt2, "Time"] + [c for c in _full_cmp.columns if c not in (_tgt2, "Time")]].copy()

    # RF models
    if "rf_models" not in dir() or rf_models is None:
        from Modules.Load_RF_forecast_models import load_rf_models as _lrf2
        print("Loading RF forecast models for 2024-2025 comparison ...")
        _rf2 = _lrf2(user=USER)
    else:
        _rf2 = rf_models

    # Dummy model – returns 0 for every prediction; we only care about the feature blocks.
    class _ZeroModel:
        sequence_length = 1
        use_target_history = False
        def predict(self, X):
            return np.array([0.0])

    _clear_cache2()
    print("Building live feature forecasts for 2024-2025 (use_precomputed=False) …")
    get_predictions(
        model=_ZeroModel(),
        dataset=_full_cmp,
        val_start=_CMP_START2,
        val_end=_CMP_END2,
        forecast_horizon=_CMP_HOR2,
        fitted_scaler=None,
        dk_zone=PRICE_ZONE,
        rf_models=_rf2,
        use_precomputed_feature_values=False,
        precomputed_feature_predictions=None,
        use_forecasted_history=USE_FORECASTED_HISTORY,
    )

    # Collect all blocks from cache into a single DataFrame.
    _live_blocks2 = _copy2.deepcopy(list(_FEAT_CACHE2.values())[0])
    _live_combined2 = (
        pd.concat(list(_live_blocks2.values()), ignore_index=True)
        .sort_values("Time")
        .reset_index(drop=True)
    )
    print(f"Live feature DataFrame: {len(_live_combined2)} rows, {len(_live_combined2.columns)} columns")

    # Only compare features that are actually forecasted (not copied from true data).
    _forecast_feats2 = [
        f for f in _FORECASTABLE2
        if f in _live_combined2.columns
        and f in _new_csv.columns
        and f not in _CAPACITY2
    ]

    # Inner join on Time.
    _m2 = _live_combined2[["Time"] + _forecast_feats2].merge(
        _new_csv[["Time"] + _forecast_feats2].rename(columns={f: f"{f}__csv" for f in _forecast_feats2}),
        on="Time",
        how="inner",
    )
    _n_rows = len(_m2)

    print(f"\n=== Feature Forecast Comparison: _new CSV vs Live-Computed ({PRICE_ZONE}, 2024–2025) ===")
    print(f"  Matched rows : {_n_rows}  |  Features compared : {len(_forecast_feats2)}")
    print()

    _DIFF_TOL2 = 1e-6
    _stats2 = []
    for _f2 in _forecast_feats2:
        _lv = _m2[_f2].to_numpy(dtype=float)
        _cv = _m2[f"{_f2}__csv"].to_numpy(dtype=float)
        _ad = np.abs(_lv - _cv)
        _stats2.append({
            "Feature":      _f2,
            "N_different":  int(np.sum(_ad > _DIFF_TOL2)),
            "Avg_abs_diff": float(np.mean(_ad)),
            "Max_abs_diff": float(np.max(_ad)),
        })

    _stats2_df = (
        pd.DataFrame(_stats2)
        .sort_values("Avg_abs_diff", ascending=False)
        .reset_index(drop=True)
    )

    _W = 36
    print(f"  {'Feature':<{_W}} {'N rows diff':>11} {'Avg |diff|':>12} {'Max |diff|':>12}")
    print("  " + "-" * (_W + 37))
    for _, _r2 in _stats2_df.iterrows():
        print(
            f"  {_r2['Feature']:<{_W}} "
            f"{int(_r2['N_different']):>11d} "
            f"{_r2['Avg_abs_diff']:>12.6f} "
            f"{_r2['Max_abs_diff']:>12.6f}"
        )

    _total_diff_feats = int((_stats2_df["N_different"] > 0).sum())
    print()
    if _total_diff_feats == 0:
        print("  All values match within tolerance (1e-6). CSV is consistent with live-computed forecasts.")
    else:
        print(f"  {_total_diff_feats} feature(s) have at least one differing row.")

