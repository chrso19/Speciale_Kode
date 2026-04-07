import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from Modules.week_predictions import get_predictions
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0

    values = np.zeros_like(y_true, dtype=float)
    values[mask] = 2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]

    return 100 * np.mean(values)


def _build_validation_folds(
        data: pd.DataFrame, 
        train_window: int,
        val_window: int,
        val_start: str, 
        predict_period: int, 
        stride: int
        ) -> list:
    """
    Build validation folds with:
    - fixed train_end at 2023-12-31 23:00:00
    - validation windows inside 2024
    - val_start stepping by `stride`
    """

    if not val_start:
        raise ValueError("val_start must be provided, e.g. '2024-01-01 00:00:00'")
    else:
        val_start = pd.to_datetime(val_start)

    val_window_start = val_start
    val_window_end = val_window_start + pd.Timedelta(hours=val_window - 1)
    train_end = val_start - pd.Timedelta(hours=1)
    train_start = train_end - pd.Timedelta(hours=train_window - 1)

    data_min = data["Time"].min()
    data_max = data["Time"].max()

    if train_start < data_min:
        raise ValueError(
            f"Not enough history for train_window={train_window}. "
            f"Need data from {train_start}, but dataset starts at {data_min}."
        )

    if val_window_start > data_max:
        raise ValueError(
            f"Dataset ends before 2024 validation starts. Dataset max time is {data_max}."
        )

    folds = []
    fold_no = 1

    val_start = val_window_start
    while val_start <= val_window_end:
        val_end = val_start + pd.Timedelta(hours=predict_period - 1)
        if val_end > val_window_end:
            break
        if val_end > data_max:
            break

        folds.append({
            "fold": fold_no,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
        })
        fold_no += 1
        val_start = val_start + pd.Timedelta(hours=stride)

    if not folds:
        raise ValueError("No valid 2024 validation folds could be created with the given horizon/stride.")

    return folds


def run_cross_validation(
    model,
    dataset: pd.DataFrame,
    dk_zone: str,
    split_setup: int,
    train_window: int,
    val_window: int,
    val_start: str,
    predict_period: int,
    stride: int,
    use_scaler: bool = True,
    print_fold_results: bool = True,
    plot: bool = True,
    rf_models = None
):
    """
    Runs cross-validation over all folds.
    
    DIFFERENCE FROM Cross_Validation_runner: This trains the model ONCE on the first fold's 
    training period, then validates on all folds WITHOUT retraining.

    For each fold:
    - get predictions on validation part (using model trained once)
    - calculate daily SMAPE
    - calculate weekly average SMAPE
    - calculate fold average SMAPE

    After all folds:
    - calculate average SMAPE across all weeks in all folds
    - plot daily SMAPE across the full validation timeline
    """

    data = dataset.copy().sort_values("Time").reset_index(drop=True)
    target_col = data.columns[0]
    feature_columns = [col for col in data.columns[1:] if col != "Time"]

    folds = _build_validation_folds(
        data=data,
        train_window=train_window,
        val_window=val_window,
        val_start=val_start,
        predict_period=predict_period,
        stride=stride,
    )

    # split_setup is kept for API compatibility with existing notebooks.
    if split_setup != 2:
        print("Validation2 uses fixed 2024 validation folds; split_setup is ignored.")

    # Extract training data from FIRST fold only
    first_fold = folds[0]
    train_data = data.loc[
        (data["Time"] >= first_fold["train_start"]) &
        (data["Time"] <= first_fold["train_end"])
    ].copy()

    X_train = train_data[feature_columns]
    y_train = train_data[target_col]

    scaler = None
    if use_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    train_hours = int((first_fold["train_end"] - first_fold["train_start"]) / pd.Timedelta(hours=1)) + 1
    print(
        "Training fold details: "
        f"train_start={first_fold['train_start']}, "
        f"train_end={first_fold['train_end']}, "
        f"hours={train_hours}, "
        f"rows={len(y_train)}, "
        f"features={len(feature_columns)}"
    )

    # TRAIN MODEL ONCE
    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    fold_results = []
    weekly_results = []
    daily_results = []
    all_predictions = []

    print(f"Model trained in {fit_seconds:.2f}s. Now validating on {len(folds)} folds...")

    # Validate on all folds WITHOUT retraining
    for fold in folds:
        fold_no = fold["fold"]

        val_data = data.loc[
            (data["Time"] >= fold["val_start"]) &
            (data["Time"] <= fold["val_end"])
        ].copy()

        preds_by_week = get_predictions(
            model=model,
            dataset=data,
            val_start=fold["val_start"],
            val_end=fold["val_end"],
            forecast_horizon=168,
            fitted_scaler=scaler,
            dk_zone=dk_zone,
            rf_models=rf_models
        )

        fold_week_rmse = []
        fold_week_mae = []
        fold_week_smape = []

        for week_no, week_pred_df in preds_by_week.items():
            week_eval = week_pred_df.merge(
                val_data[["Time", target_col]],
                on="Time",
                how="left"
            )

            week_eval["fold"] = fold_no
            week_eval["week"] = week_no
            week_eval["Date"] = week_eval["Time"].dt.floor("D")

            week_rmse = np.sqrt(mean_squared_error(
                week_eval[target_col].values,
                week_eval["Prediction"].values)
                )
            week_mae = mean_absolute_error(
                week_eval[target_col].values,
                week_eval["Prediction"].values
            )
            week_smape = smape(
                week_eval[target_col].values,
                week_eval["Prediction"].values
            )

            fold_week_rmse.append(week_rmse)
            fold_week_mae.append(week_mae)
            fold_week_smape.append(week_smape)

            weekly_results.append({
                "fold": fold_no,
                "week": week_no,
                "week_start": week_eval["Time"].min(),
                "week_end": week_eval["Time"].max(),
                "weekly_rmse": week_rmse,
                "weekly_mae": week_mae,
                "weekly_smape": week_smape,
            })

            daily_rmse_df = (
                week_eval.groupby("Date")
                .apply(lambda g: np.sqrt(mean_squared_error(g[target_col].values, g["Prediction"].values)), include_groups = False)
                .reset_index(name="daily_rmse")
            )
            daily_mae_df = (
                week_eval.groupby("Date")
                .apply(lambda g: mean_absolute_error(g[target_col].values, g["Prediction"].values), include_groups=False)
                .reset_index(name="daily_mae")
            )
            daily_smape_df = (
                week_eval.groupby("Date")
                .apply(lambda g: smape(g[target_col].values, g["Prediction"].values), include_groups=False)
                .reset_index(name="daily_smape")
            )
            daily_rmse_df["fold"] = fold_no
            daily_rmse_df["week"] = week_no
            daily_mae_df["fold"] = fold_no
            daily_mae_df["week"] = week_no
            daily_smape_df["fold"] = fold_no
            daily_smape_df["week"] = week_no

            # Merge all three daily metric frames on Date/fold/week
            # so each row has rmse, mae and smape for the same day
            daily_merged = (
                daily_rmse_df
                .merge(daily_mae_df,   on=["Date", "fold", "week"])
                .merge(daily_smape_df, on=["Date", "fold", "week"])
            )
            daily_results.append(daily_merged)
            all_predictions.append(week_eval)

        fold_avg_rmse = np.mean(fold_week_rmse)
        fold_avg_mae = np.mean(fold_week_mae)
        fold_avg_smape = np.mean(fold_week_smape)

        fold_results.append({
            "fold": fold_no,
            "train_start": fold["train_start"],
            "train_end": fold["train_end"],
            "val_start": fold["val_start"],
            "val_end": fold["val_end"],
            "fold_avg_rmse": fold_avg_rmse,
            "fold_avg_mae": fold_avg_mae,
            "fold_avg_smape": fold_avg_smape
        })

    fold_results_df = pd.DataFrame(fold_results)
    weekly_results_df = pd.DataFrame(weekly_results)
    daily_results_df = pd.concat(daily_results, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    overall_avg_weekly_rmse = weekly_results_df["weekly_rmse"].mean()
    overall_avg_weekly_mae = weekly_results_df["weekly_mae"].mean()
    overall_avg_weekly_smape = weekly_results_df["weekly_smape"].mean()

    overall_daily_rmse_df = (
        daily_results_df.groupby("Date", as_index=False)["daily_rmse"]
        .mean()
        .sort_values("Date")
    )
    overall_daily_mae_df = (
        daily_results_df.groupby("Date", as_index=False)["daily_mae"]
        .mean()
        .sort_values("Date")
    )
    overall_daily_smape_df = (
        daily_results_df.groupby("Date", as_index=False)["daily_smape"]
        .mean()
        .sort_values("Date")
    )

    overall_avg_daily_rmse = overall_daily_rmse_df["daily_rmse"].mean()
    overall_avg_daily_mae = overall_daily_mae_df["daily_mae"].mean()
    overall_avg_daily_smape = overall_daily_smape_df["daily_smape"].mean()

    # Average SMAPE for each day position (1..7) inside the weekly horizon.
    daily_smape_by_day = daily_results_df[["fold", "week", "Date", "daily_smape"]].copy()
    daily_smape_by_day = daily_smape_by_day.sort_values(["fold", "week", "Date"])
    daily_smape_by_day["day_in_week"] = daily_smape_by_day.groupby(["fold", "week"]).cumcount() + 1
    avg_smape_by_day = {
        f"avg_smape_day_{day}": daily_smape_by_day.loc[
            daily_smape_by_day["day_in_week"] == day,
            "daily_smape"
        ].mean()
        for day in range(1, 8)
    }

    if print_fold_results:
        print("\nFold results:")
        print(fold_results_df.to_string(index=False))

        print("\nWeekly results:")
        print(weekly_results_df.to_string(index=False))

    print(f"\nAverage RMSE across all weeks in all folds: {overall_avg_weekly_rmse:.3f}")
    print(f"\nAverage MAE across all weeks in all folds: {overall_avg_weekly_mae:.3f}")
    print(f"\nAverage SMAPE across all weeks in all folds: {overall_avg_weekly_smape:.3f}")

    if plot:
        plt.figure(figsize=(14, 6))
        plt.plot(
            overall_daily_smape_df["Date"],
            overall_daily_smape_df["daily_smape"]
        )
        plt.xlabel("Date")
        plt.ylabel("SMAPE (%)")
        plt.title("Daily SMAPE across all validation folds")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return {
        "model": model,
        "fold_results": fold_results_df,
        "weekly_results": weekly_results_df,
        "daily_results": daily_results_df,
        "overall_daily_rmse": overall_daily_rmse_df,
        "overall_daily_mae": overall_daily_mae_df,
        "overall_daily_smape": overall_daily_smape_df,
        "predictions": predictions_df,
        "overall_avg_weekly_rmse": overall_avg_weekly_rmse,
        "overall_avg_weekly_mae": overall_avg_weekly_mae,
        "overall_avg_weekly_smape": overall_avg_weekly_smape,
        "overall_avg_daily_rmse": overall_avg_daily_rmse,
        "overall_avg_daily_mae": overall_avg_daily_mae,
        "overall_avg_daily_smape": overall_avg_daily_smape,
        **avg_smape_by_day
    }
