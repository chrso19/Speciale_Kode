import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from Modules.week_predictions2 import get_predictions


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
    val_window: int,
    val_start: str,
    predict_period: int,
    stride: int,
    validation_reference: pd.DataFrame | None = None,
) -> list:
    """
    Build validation folds inside the [val_start, val_start + val_window) window.

    If validation_reference is provided (e.g. dataset_validation), only folds that
    have full hourly coverage in that reference are kept.
    """

    if not val_start:
        raise ValueError("val_start must be provided, e.g. '2024-01-01 00:00:00'")

    val_start_ts = pd.to_datetime(val_start)
    val_window_end = val_start_ts + pd.Timedelta(hours=val_window - 1)

    data_max = data["Time"].max()
    if val_start_ts > data_max:
        raise ValueError(
            f"Dataset ends before validation starts. Dataset max time is {data_max}."
        )

    ref = data if validation_reference is None else validation_reference

    folds = []
    fold_no = 1
    current_start = val_start_ts

    while current_start <= val_window_end:
        current_end = current_start + pd.Timedelta(hours=predict_period - 1)
        if current_end > val_window_end:
            break
        if current_end > data_max:
            break

        ref_window = ref.loc[
            (ref["Time"] >= current_start) & (ref["Time"] <= current_end)
        ]

        if len(ref_window) == predict_period:
            folds.append(
                {
                    "fold": fold_no,
                    "val_start": current_start,
                    "val_end": current_end,
                }
            )
            fold_no += 1

        current_start = current_start + pd.Timedelta(hours=stride)

    if not folds:
        raise ValueError(
            "No valid validation folds could be created with the given horizon/stride."
        )

    return folds


def run_cross_validation(
    model,
    dataset_train: pd.DataFrame,
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
    rf_models=None,
    use_precomputed_feature_values: bool = False,
    precomputed_feature_predictions: pd.DataFrame | None = None,
    use_forecasted_history: bool = True,
    dataset_validation: pd.DataFrame | None = None,
    include_remaining_2024: bool = True,
    dataset_context: pd.DataFrame | None = None,
    feature_columns: list | None = None,
):
    """
    Cross-validation with explicit support for split datasets.

    Arguments:
    - dataset_train: Training source data.
      This should contain at least TRAIN_WINDOW history ending at val_start.
    - dataset_validation: Optional explicit validation rows (e.g. selected 2024 windows).
      If provided, folds are only kept when this dataset contains full horizon coverage.
    - include_remaining_2024:
      True  -> training uses TRAIN_WINDOW history + remaining 2024 rows not in validation windows.
      False -> training uses only TRAIN_WINDOW history ending at val_start.
    """

    train_data_source = dataset_train.copy().sort_values("Time").reset_index(drop=True)

    if dataset_validation is not None and not dataset_validation.empty:
        validation_source = dataset_validation.copy().sort_values("Time").reset_index(drop=True)
        data = (
            pd.concat([train_data_source, validation_source], ignore_index=True)
            .sort_values("Time")
            .drop_duplicates(subset=["Time"], keep="last")
            .reset_index(drop=True)
        )
    else:
        validation_source = None
        data = train_data_source.copy()

    # Build a separate context dataset for get_predictions if provided.
    # This allows lag computation across the full 2024 range even when
    # the training set does not include 2024 non-validation data.
    if dataset_context is not None:
        context_source = dataset_context.copy().sort_values("Time").reset_index(drop=True)
        if validation_source is not None:
            context_data = (
                pd.concat([context_source, validation_source], ignore_index=True)
                .sort_values("Time")
                .drop_duplicates(subset=["Time"], keep="last")
                .reset_index(drop=True)
            )
        else:
            context_data = context_source
    else:
        context_data = None

    target_col = data.columns[0]
    if feature_columns is None:
        feature_columns = [col for col in data.columns[1:] if col != "Time"]

    # Bug fix: restrict context_data columns to target + Time + feature_columns so
    # get_predictions sees the same feature set the model was trained on.
    if context_data is not None:
        keep_cols = [c for c in context_data.columns if c == "Time" or c == target_col or c in feature_columns]
        context_data = context_data[keep_cols].copy()

    folds = _build_validation_folds(
        data=data,
        val_window=val_window,
        val_start=val_start,
        predict_period=predict_period,
        stride=stride,
        validation_reference=validation_source,
    )

    if split_setup != 2:
        print("Validation3 uses fixed validation folds; split_setup is ignored.")

    val_start_ts = pd.to_datetime(val_start)
    train_end = val_start_ts - pd.Timedelta(hours=1)
    train_start = train_end - pd.Timedelta(hours=train_window - 1)

    base_train = data.loc[
        (data["Time"] >= train_start) & (data["Time"] <= train_end)
    ].copy()

    if len(base_train) < train_window:
        data_min = data["Time"].min()
        raise ValueError(
            f"Not enough history for train_window={train_window}. "
            f"Need data from {train_start}, but dataset starts at {data_min}."
        )

    train_data = base_train

    if include_remaining_2024:
        validation_mask = pd.Series(False, index=train_data_source.index)
        for fold in folds:
            fold_mask = (
                (train_data_source["Time"] >= fold["val_start"])
                & (train_data_source["Time"] <= fold["val_end"])
            )
            validation_mask = validation_mask | fold_mask

        year_2024_mask = (
            (train_data_source["Time"] >= pd.Timestamp("2024-01-01 00:00:00"))
            & (train_data_source["Time"] < pd.Timestamp("2025-01-01 00:00:00"))
        )

        remaining_2024 = train_data_source.loc[year_2024_mask & (~validation_mask)].copy()

        train_data = (
            pd.concat([base_train, remaining_2024], ignore_index=True)
            .sort_values("Time")
            .drop_duplicates(subset=["Time"], keep="first")
            .reset_index(drop=True)
        )

    X_train = train_data[feature_columns]
    y_train = train_data[target_col]

    scaler = None
    if use_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    train_hours = int((train_end - train_start) / pd.Timedelta(hours=1)) + 1
    # print(
    #     "\nTraining details: "
    #     f"train_start={train_start}, "
    #     f"train_end={train_end}, "
    #     f"base_hours={train_hours}, "
    #     f"include_remaining_2024={include_remaining_2024}, "
    #     f"rows={len(y_train)}, "
    #     f"features={len(feature_columns)}"
    # )

    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    fold_results = []
    weekly_results = []
    daily_results = []
    all_predictions = []

    print(f"Model trained in {fit_seconds:.2f}s. Now validating on {len(folds)} folds...")

    for fold in folds:
        fold_no = fold["fold"]

        if validation_source is not None:
            val_data = validation_source.loc[
                (validation_source["Time"] >= fold["val_start"])
                & (validation_source["Time"] <= fold["val_end"])
            ].copy()
        else:
            val_data = data.loc[
                (data["Time"] >= fold["val_start"]) & (data["Time"] <= fold["val_end"])
            ].copy()

        prediction_context = context_data if dataset_context is not None else data
        preds_by_week = get_predictions(
            model=model,
            dataset=prediction_context,
            val_start=fold["val_start"],
            val_end=fold["val_end"],
            forecast_horizon=168,
            fitted_scaler=scaler,
            dk_zone=dk_zone,
            rf_models=rf_models,
            use_precomputed_feature_values=use_precomputed_feature_values,
            precomputed_feature_predictions=precomputed_feature_predictions,
            use_forecasted_history=use_forecasted_history,
        )

        fold_week_rmse = []
        fold_week_mae = []
        fold_week_smape = []

        for week_no, week_pred_df in preds_by_week.items():
            week_eval = week_pred_df.merge(
                val_data[["Time", target_col]],
                on="Time",
                how="left",
            )

            week_eval = week_eval.dropna(subset=[target_col]).copy()
            if week_eval.empty:
                continue

            week_eval["fold"] = fold_no
            week_eval["week"] = week_no
            week_eval["Date"] = week_eval["Time"].dt.floor("D")

            week_rmse = np.sqrt(
                mean_squared_error(
                    week_eval[target_col].values,
                    week_eval["Prediction"].values,
                )
            )
            week_mae = mean_absolute_error(
                week_eval[target_col].values,
                week_eval["Prediction"].values,
            )
            week_smape = smape(
                week_eval[target_col].values,
                week_eval["Prediction"].values,
            )

            fold_week_rmse.append(week_rmse)
            fold_week_mae.append(week_mae)
            fold_week_smape.append(week_smape)

            weekly_results.append(
                {
                    "fold": fold_no,
                    "week": week_no,
                    "week_start": week_eval["Time"].min(),
                    "week_end": week_eval["Time"].max(),
                    "weekly_rmse": week_rmse,
                    "weekly_mae": week_mae,
                    "weekly_smape": week_smape,
                }
            )

            daily_rmse_df = (
                week_eval.groupby("Date")
                .apply(
                    lambda g: np.sqrt(
                        mean_squared_error(g[target_col].values, g["Prediction"].values)
                    ),
                    include_groups=False,
                )
                .reset_index(name="daily_rmse")
            )
            daily_mae_df = (
                week_eval.groupby("Date")
                .apply(
                    lambda g: mean_absolute_error(g[target_col].values, g["Prediction"].values),
                    include_groups=False,
                )
                .reset_index(name="daily_mae")
            )
            daily_smape_df = (
                week_eval.groupby("Date")
                .apply(
                    lambda g: smape(g[target_col].values, g["Prediction"].values),
                    include_groups=False,
                )
                .reset_index(name="daily_smape")
            )

            daily_rmse_df["fold"] = fold_no
            daily_rmse_df["week"] = week_no
            daily_mae_df["fold"] = fold_no
            daily_mae_df["week"] = week_no
            daily_smape_df["fold"] = fold_no
            daily_smape_df["week"] = week_no

            daily_merged = (
                daily_rmse_df
                .merge(daily_mae_df, on=["Date", "fold", "week"])
                .merge(daily_smape_df, on=["Date", "fold", "week"])
            )
            daily_results.append(daily_merged)
            all_predictions.append(week_eval)

        if not fold_week_smape:
            continue

        fold_avg_rmse = np.mean(fold_week_rmse)
        fold_avg_mae = np.mean(fold_week_mae)
        fold_avg_smape = np.mean(fold_week_smape)

        fold_results.append(
            {
                "fold": fold_no,
                "train_start": train_start,
                "train_end": train_end,
                "val_start": fold["val_start"],
                "val_end": fold["val_end"],
                "fold_avg_rmse": fold_avg_rmse,
                "fold_avg_mae": fold_avg_mae,
                "fold_avg_smape": fold_avg_smape,
            }
        )

    if not fold_results:
        raise ValueError("No fold metrics were computed. Check validation coverage and inputs.")

    fold_results_df = pd.DataFrame(fold_results)
    weekly_results_df = pd.DataFrame(weekly_results)
    daily_results_df = pd.concat(daily_results, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    overall_avg_weekly_rmse = weekly_results_df["weekly_rmse"].mean()
    overall_avg_weekly_mae = weekly_results_df["weekly_mae"].mean()
    overall_avg_weekly_smape = weekly_results_df["weekly_smape"].mean()

    overall_daily_rmse_df = (
        daily_results_df.groupby("Date", as_index=False)["daily_rmse"].mean().sort_values("Date")
    )
    overall_daily_mae_df = (
        daily_results_df.groupby("Date", as_index=False)["daily_mae"].mean().sort_values("Date")
    )
    overall_daily_smape_df = (
        daily_results_df.groupby("Date", as_index=False)["daily_smape"].mean().sort_values("Date")
    )

    overall_avg_daily_rmse = overall_daily_rmse_df["daily_rmse"].mean()
    overall_avg_daily_mae = overall_daily_mae_df["daily_mae"].mean()
    overall_avg_daily_smape = overall_daily_smape_df["daily_smape"].mean()

    daily_smape_by_day = daily_results_df[["fold", "week", "Date", "daily_smape"]].copy()
    daily_smape_by_day = daily_smape_by_day.sort_values(["fold", "week", "Date"])
    daily_smape_by_day["day_in_week"] = daily_smape_by_day.groupby(["fold", "week"]).cumcount() + 1
    avg_smape_by_day = {
        f"avg_smape_day_{day}": daily_smape_by_day.loc[
            daily_smape_by_day["day_in_week"] == day,
            "daily_smape",
        ].mean()
        for day in range(1, 8)
    }

    if print_fold_results:
        print("\nFold results:")
        print(fold_results_df.to_string(index=False))

        print("\nWeekly results:")
        print(weekly_results_df.to_string(index=False))

    # print(f"\nAverage RMSE across all weeks in all folds: {overall_avg_weekly_rmse:.3f}")
    # print(f"\nAverage MAE across all weeks in all folds: {overall_avg_weekly_mae:.3f}")
    print(f"\nAverage SMAPE across all weeks in all folds: {overall_avg_weekly_smape:.3f}")

    if plot:
        plt.figure(figsize=(14, 6))
        plt.plot(overall_daily_smape_df["Date"], overall_daily_smape_df["daily_smape"])
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
        **avg_smape_by_day,
    }
