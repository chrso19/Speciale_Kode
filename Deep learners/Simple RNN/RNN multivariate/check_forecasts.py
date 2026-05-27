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
VAL_WINDOW = 8784
PREDICT_PERIOD = 4 * 168
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
    prediction_path = Path(project_root) / "Data" / f"feature_predictions_{zone}_2024-2025.csv"
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
import pandas as pd
from pathlib import Path
import wandb
import joblib
from sklearn.preprocessing import StandardScaler
from Modules.Validation3 import _build_validation_folds
from Modules.week_predictions2 import get_predictions

# =====================================================================================
# Test final model on 2025 only, week-by-week (168h blocks)
# =====================================================================================
WANDB_PROJECT = "LSTM final"
WANDB_RUN_NAME = "DK1_LSTM_2y_pricemaskexcl_2024price_lag1inclincl_4val"  # Must match final training run name
WANDB_TEST_RUN_NAME = f"{WANDB_RUN_NAME}_test"
WANDB_ARTIFACT_NAME = f"{WANDB_RUN_NAME}_model"

FORECAST_HORIZON = 168  # 1 week
PREDICT_PERIOD = 168    # one week per validation fold
STRIDE = 168            # next fold starts next week
TEST_START = pd.Timestamp("2025-01-01 00:00:00")
TEST_WINDOW = 8760
TEST_END = TEST_START + pd.Timedelta(hours=TEST_WINDOW - 1)

# Retrieve zone data
if PRICE_ZONE == "DK1":
    test_source = DK1_test.copy() if "DK1_test" in globals() else None
    history_source = DK1_train.copy() if "DK1_train" in globals() else None
elif PRICE_ZONE == "DK2":
    test_source = DK2_test.copy() if "DK2_test" in globals() else None
    history_source = DK2_train.copy() if "DK2_train" in globals() else None
else:
    raise ValueError("PRICE_ZONE must be 'DK1' or 'DK2'.")

if test_source is None or history_source is None:
    raise ValueError(f"Missing train/test source data for {PRICE_ZONE}. Run data loading first.")

test_source = test_source.sort_values("Time").reset_index(drop=True)
history_source = history_source.sort_values("Time").reset_index(drop=True)

# Test only on 2025 window requested by user
test_set = test_source.loc[
    (test_source["Time"] >= TEST_START) & (test_source["Time"] <= TEST_END)
].copy().sort_values("Time").reset_index(drop=True)

if len(test_set) != TEST_WINDOW:
    raise ValueError(
        f"Expected exactly TEST_WINDOW={TEST_WINDOW} rows from {TEST_START} to {TEST_END}, got {len(test_set)}."
    )

# Use the same feature columns the model was trained on (respects INCLUDE_LAGS and INCLUDE_PRICE_HISTORY_AS_INPUT)
feature_columns = [c for c in dataset_train_input.columns if c not in ["Time", "DKPrice"]]
if not feature_columns:
    raise ValueError("No feature columns found in dataset_train_input. Run the load data cell first.")

# Include end-of-2024 history so first 2025 prediction can build sequence/lag features.
# get_predictions expects DKPrice as the first column (target), followed by Time and features.
full_eval_dataset = (
    pd.concat([history_source, test_set], ignore_index=True)
    .sort_values("Time")
    .drop_duplicates(subset=["Time"], keep="last")
    .reset_index(drop=True)
)
# Compute DKPrice_lag1 if the model was trained with it as a feature.
if "DKPrice_lag1" in feature_columns:
    full_eval_dataset["DKPrice_lag1"] = full_eval_dataset["DKPrice"].shift(1)
missing_in_eval = [c for c in feature_columns if c not in full_eval_dataset.columns]
if missing_in_eval:
    raise ValueError(f"full_eval_dataset is missing feature columns from dataset_train_input: {missing_in_eval}")
# DKPrice must be the first column so get_predictions treats it as the target
full_eval_dataset = full_eval_dataset[["DKPrice", "Time"] + feature_columns].copy()

# Load trained model
api = wandb.Api(timeout=60)
model_artifact = api.artifact(WANDB_ARTIFACT_NAME, type="model")
artifact_dir = Path(model_artifact.download())
model_path = artifact_dir / "model.joblib"
if not model_path.exists():
    raise ValueError(f"Could not find model.joblib in downloaded artifact: {artifact_dir}")

model = joblib.load(model_path)
print(f"Loaded model artifact: {WANDB_ARTIFACT_NAME}:latest")
print(f"Model path: {model_path}")
print(f"Test window: {TEST_START} -> {TEST_END} ({len(test_set)} rows)")
print(f"Feature columns ({len(feature_columns)}): {feature_columns}")

# Fit scaler on the same training tail used during training.
# Compute target-lag columns on the full sorted history before taking the tail
# so that the first row of the tail has a valid lag value.
train_for_scaler = history_source.sort_values("Time").copy()
if "DKPrice_lag1" in feature_columns and "DKPrice_lag1" not in train_for_scaler.columns:
    train_for_scaler["DKPrice_lag1"] = train_for_scaler["DKPrice"].shift(1)
train_for_scaler = train_for_scaler.tail(TRAIN_WINDOW).reset_index(drop=True)
X_scaler = train_for_scaler[feature_columns].astype(np.float32)
scaler = StandardScaler()
scaler.fit(X_scaler)

# Build 2025 weekly folds
folds = _build_validation_folds(
    data=full_eval_dataset,
    val_window=TEST_WINDOW,
    val_start=str(TEST_START),
    predict_period=PREDICT_PERIOD,
    stride=STRIDE,
    validation_reference=test_set,
)
print(f"Generated {len(folds)} weekly folds with predict_period={PREDICT_PERIOD}, stride={STRIDE}.")

# Metric helper
def smape_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    vals = np.where(denom == 0, 0.0, 200.0 * np.abs(y_pred - y_true) / denom)
    return float(np.mean(vals))

# Evaluate fold-by-fold (week-by-week)
all_week_evals = []
day_smapes_per_week = {}  # (fold_no, week_no) -> {day_num -> smape}

for fold in folds:
    fold_no = int(fold["fold"])
    fold_val_data = test_set.loc[
        (test_set["Time"] >= fold["val_start"]) & (test_set["Time"] <= fold["val_end"])
    ].copy()

    preds_by_week = get_predictions(
        model=model,
        dataset=full_eval_dataset,
        val_start=fold["val_start"],
        val_end=fold["val_end"],
        forecast_horizon=FORECAST_HORIZON,
        fitted_scaler=scaler,
        dk_zone=PRICE_ZONE,
        rf_models=rf_models if "rf_models" in globals() else None,
        use_precomputed_feature_values=use_precomputed_feature_values if "use_precomputed_feature_values" in globals() else False,
        precomputed_feature_predictions=feature_predictions if "feature_predictions" in globals() else None,
        use_forecasted_history=USE_FORECASTED_HISTORY,
    )

    for week_no, week_pred_df in preds_by_week.items():
        week_eval = week_pred_df.merge(
            fold_val_data[["Time", "DKPrice"]],
            on="Time",
            how="left",
        )
        week_eval = week_eval.dropna(subset=["DKPrice"]).copy()
        if week_eval.empty:
            continue

        week_eval["fold"] = fold_no
        week_eval["week_in_fold"] = int(week_no)
        all_week_evals.append(week_eval)

        week_key = (fold_no, int(week_no))
        day_smapes_per_week[week_key] = {}
        week_eval_sorted = week_eval.sort_values("Time").reset_index(drop=True)

        for day_num in range(1, 8):
            start_hour = (day_num - 1) * 24
            end_hour = day_num * 24
            day_data = week_eval_sorted.iloc[start_hour:end_hour]
            if len(day_data) > 0:
                day_smapes_per_week[week_key][day_num] = smape_mean(
                    day_data["DKPrice"].values,
                    day_data["Prediction"].values,
                )

if not all_week_evals:
    raise RuntimeError("No aligned predictions produced for 2025 weekly folds.")

results_df = pd.concat(all_week_evals, ignore_index=True).sort_values("Time").reset_index(drop=True)
results_df = results_df.rename(columns={"DKPrice": "Actual"})
results_df["Error"] = results_df["Actual"] - results_df["Prediction"]
results_df["Abs_Error"] = np.abs(results_df["Error"])

y_actual = results_df["Actual"].values
y_pred = results_df["Prediction"].values

# Overall metrics
overall_mse = float(np.mean((y_actual - y_pred) ** 2))
overall_smape = smape_mean(y_actual, y_pred)
overall_mae = float(np.mean(np.abs(y_actual - y_pred)))
overall_rmse = float(np.sqrt(overall_mse))

print("\n=== Overall Test Results ===")
print(f"MSE Loss:  {overall_mse:.6f}")
print(f"SMAPE:     {overall_smape:.6f}")
print(f"MAE:       {overall_mae:.6f}")
print(f"RMSE:      {overall_rmse:.6f}")

# Weekly metrics (calendar week)
print("\n=== Weekly Metrics ===")
results_df["Week"] = results_df["Time"].dt.isocalendar().week
weekly_results = []
for week, group in results_df.groupby("Week"):
    week_actual = group["Actual"].values
    week_pred = group["Prediction"].values
    week_rmse = float(np.sqrt(np.mean((week_actual - week_pred) ** 2)))
    week_mae = float(np.mean(np.abs(week_actual - week_pred)))
    week_smape = smape_mean(week_actual, week_pred)
    weekly_results.append({
        "week": int(week),
        "rmse": week_rmse,
        "mae": week_mae,
        "smape": week_smape,
        "n_samples": len(group),
    })
    print(f"Week {week:02d}: RMSE={week_rmse:.4f}, MAE={week_mae:.4f}, SMAPE={week_smape:.4f}")

# Daily metrics
print("\n=== Daily Metrics (by Date) ===")
results_df["Date"] = results_df["Time"].dt.date
daily_results = []
for date, group in results_df.groupby("Date"):
    day_actual = group["Actual"].values
    day_pred = group["Prediction"].values
    day_rmse = float(np.sqrt(np.mean((day_actual - day_pred) ** 2)))
    day_mae = float(np.mean(np.abs(day_actual - day_pred)))
    day_smape = smape_mean(day_actual, day_pred)
    daily_results.append({
        "date": str(date),
        "rmse": day_rmse,
        "mae": day_mae,
        "smape": day_smape,
        "n_samples": len(group),
    })
    print(f"{date}: RMSE={day_rmse:.4f}, MAE={day_mae:.4f}, SMAPE={day_smape:.4f}")

# Day-of-forecast metrics (Day 1..7 inside each predicted week)
print("\n=== Day-of-Forecast Metrics (within each 168-hour period) ===")
day_of_forecast_smapes = {}
for day_num in range(1, 8):
    smapes = [
        day_smapes_per_week[w].get(day_num)
        for w in day_smapes_per_week
        if day_num in day_smapes_per_week[w]
    ]
    smapes = [s for s in smapes if s is not None and not np.isnan(s)]
    day_of_forecast_smapes[day_num] = float(np.mean(smapes)) if smapes else float("nan")
    print(f"Day {day_num} (hours {(day_num-1)*24}-{day_num*24-1}): avg SMAPE={day_of_forecast_smapes[day_num]:.4f}")

# Average weekly/daily metrics
avg_weekly_rmse = float(np.mean([r["rmse"] for r in weekly_results]))
avg_weekly_mae = float(np.mean([r["mae"] for r in weekly_results]))
avg_weekly_smape = float(np.mean([r["smape"] for r in weekly_results]))

avg_daily_rmse = float(np.mean([r["rmse"] for r in daily_results]))
avg_daily_mae = float(np.mean([r["mae"] for r in daily_results]))
avg_daily_smape = float(np.mean([r["smape"] for r in daily_results]))

print("\n=== Average Weekly Metrics ===")
print(f"Avg Weekly RMSE: {avg_weekly_rmse:.6f}")
print(f"Avg Weekly MAE:  {avg_weekly_mae:.6f}")
print(f"Avg Weekly SMAPE: {avg_weekly_smape:.6f}")

print("\n=== Average Daily Metrics ===")
print(f"Avg Daily RMSE: {avg_daily_rmse:.6f}")
print(f"Avg Daily MAE:  {avg_daily_mae:.6f}")
print(f"Avg Daily SMAPE: {avg_daily_smape:.6f}")

# Save all hourly predictions to CSV
output_root = Path(project_root) / "Deep learners" / "Simple RNN"
output_root.mkdir(parents=True, exist_ok=True)
predictions_csv_path = output_root / f"{WANDB_RUN_NAME}_test_predictions.csv"
results_df.to_csv(predictions_csv_path, index=False, sep=";", decimal=".")
print(f"\nHourly predictions saved to: {predictions_csv_path}")

weekly_df = pd.DataFrame(weekly_results)
daily_df = pd.DataFrame(daily_results)

