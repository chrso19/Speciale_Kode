import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# USER SETTINGS
# ============================================================

DATA_FOLDER = Path(".")

TIME_COL = "Time"
ACTUAL_COL = "DKPrice"
PRED_COL = "Prediction"
ZONE_COL = "DKZone"

# Choose plot type: "errors" or "predictions"
PLOT_TYPE = "predictions"

# Choose metric: "SMAPE", "MAPE", "RMSE", or "MAE"
# Only used when PLOT_TYPE = "errors"
METRIC = "MAE"

# Choose frequency: "hourly" or "daily"
# For PLOT_TYPE = "predictions", daily means daily average prices.
# For PLOT_TYPE = "errors", daily means true daily metric values.
FREQUENCY = "daily"

# Choose bidding zone: "DK1", "DK2", or None
DK_ZONE = None

MODEL_1_FILE = "Baseline/MovingAverage/DK1_predictions_MA.csv"
MODEL_2_FILE = "Baseline/Seasonal/DK1_predictions_seasonal.csv"

MODEL_1_NAME = "Moving Average"
MODEL_2_NAME = "Seasonal Naïve"

START_TIME = "2025-06-01 00:00:00"
END_TIME = "2025-08-30 23:00:00"


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0

    if not np.any(mask):
        return np.nan

    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0

    if not np.any(mask):
        return 0.0

    values = np.zeros_like(y_true, dtype=float)
    values[mask] = 2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]

    return 100 * np.mean(values)


def calculate_metric(y_true, y_pred, metric):
    metric = metric.upper()

    if metric == "MAE":
        return mae(y_true, y_pred)

    if metric == "RMSE":
        return rmse(y_true, y_pred)

    if metric == "MAPE":
        return mape(y_true, y_pred)

    if metric == "SMAPE":
        return smape(y_true, y_pred)

    raise ValueError("Metric must be one of: SMAPE, MAPE, RMSE, MAE")


def calculate_pointwise_error(y_true, y_pred, metric):
    """
    Used for hourly error plots.
    For hourly MAE/RMSE, this is pointwise absolute error.
    """
    metric = metric.upper()

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if metric in ["MAE", "RMSE"]:
        return np.abs(y_true - y_pred)

    if metric == "MAPE":
        error = np.full_like(y_true, np.nan, dtype=float)
        mask = y_true != 0
        error[mask] = 100 * np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        return error

    if metric == "SMAPE":
        denominator = np.abs(y_true) + np.abs(y_pred)
        error = np.zeros_like(y_true, dtype=float)
        mask = denominator != 0
        error[mask] = 100 * (
            2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]
        )
        return error

    raise ValueError("Metric must be one of: SMAPE, MAPE, RMSE, MAE")


# ============================================================
# DATA LOADING
# ============================================================

def load_model_data(file_name):
    file_path = DATA_FOLDER / file_name

    df = pd.read_csv(file_path)

    required_cols = [TIME_COL, ACTUAL_COL, PRED_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"{file_name} is missing columns: {missing_cols}. "
            f"Expected columns are: {required_cols}"
        )

    if DK_ZONE is not None:
        if ZONE_COL not in df.columns:
            raise ValueError(
                f"{file_name} does not contain a {ZONE_COL} column, "
                f"but DK_ZONE is set to {DK_ZONE}."
            )

        df = df.loc[df[ZONE_COL] == DK_ZONE].copy()

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    start_time = pd.to_datetime(START_TIME)
    end_time = pd.to_datetime(END_TIME)

    df = df.loc[
        (df[TIME_COL] >= start_time) &
        (df[TIME_COL] <= end_time)
    ].copy()

    if df.empty:
        raise ValueError(
            f"No rows found in {file_name} between {START_TIME} and {END_TIME}."
        )

    return df


# ============================================================
# ERROR AGGREGATION
# ============================================================

def prepare_error_series(df, metric, frequency):
    metric = metric.upper()
    frequency = frequency.lower()

    if frequency == "hourly":
        df = df.copy()
        df["plot_time"] = df[TIME_COL]
        df["error"] = calculate_pointwise_error(
            y_true=df[ACTUAL_COL].to_numpy(dtype=float),
            y_pred=df[PRED_COL].to_numpy(dtype=float),
            metric=metric,
        )

        return df[["plot_time", "error"]]

    if frequency == "daily":
        df = df.copy()
        df["plot_time"] = df[TIME_COL].dt.floor("D")

        daily_errors = (
            df.groupby("plot_time")
            .apply(
                lambda group: calculate_metric(
                    y_true=group[ACTUAL_COL].to_numpy(dtype=float),
                    y_pred=group[PRED_COL].to_numpy(dtype=float),
                    metric=metric,
                )
            )
            .reset_index(name="error")
        )

        return daily_errors

    raise ValueError("FREQUENCY must be either 'hourly' or 'daily'")


# ============================================================
# PREDICTION AGGREGATION
# ============================================================

def prepare_prediction_series(model_1_df, model_2_df, frequency):
    frequency = frequency.lower()

    model_1 = model_1_df[[TIME_COL, ACTUAL_COL, PRED_COL]].copy()
    model_2 = model_2_df[[TIME_COL, ACTUAL_COL, PRED_COL]].copy()

    model_1 = model_1.rename(
        columns={
            ACTUAL_COL: "actual_model_1",
            PRED_COL: "prediction_model_1",
        }
    )

    model_2 = model_2.rename(
        columns={
            ACTUAL_COL: "actual_model_2",
            PRED_COL: "prediction_model_2",
        }
    )

    merged = model_1.merge(model_2, on=TIME_COL, how="inner")

    if merged.empty:
        raise ValueError(
            "The two model files have no overlapping timestamps after filtering."
        )

    actual_difference = (
        merged["actual_model_1"] - merged["actual_model_2"]
    ).abs().max()

    if actual_difference > 1e-8:
        print(
            "Warning: Actual DKPrice values are not identical in the two files. "
            "Using actual values from the first model file."
        )

    merged = merged.rename(
        columns={
            TIME_COL: "plot_time",
            "actual_model_1": "actual",
            "prediction_model_1": "prediction_model_1",
            "prediction_model_2": "prediction_model_2",
        }
    )

    if frequency == "hourly":
        return merged[
            ["plot_time", "actual", "prediction_model_1", "prediction_model_2"]
        ]

    if frequency == "daily":
        merged["plot_time"] = merged["plot_time"].dt.floor("D")

        daily = (
            merged.groupby("plot_time")[
                ["actual", "prediction_model_1", "prediction_model_2"]
            ]
            .mean()
            .reset_index()
        )

        return daily

    raise ValueError("FREQUENCY must be either 'hourly' or 'daily'")


# ============================================================
# PLOTTING WITH PLOTLY
# ============================================================

def plot_errors():
    metric = METRIC.upper()
    frequency = FREQUENCY.lower()

    model_1_df = load_model_data(MODEL_1_FILE)
    model_2_df = load_model_data(MODEL_2_FILE)

    model_1_errors = prepare_error_series(model_1_df, metric, frequency)
    model_2_errors = prepare_error_series(model_2_df, metric, frequency)

    if metric in ["SMAPE", "MAPE"]:
        ylabel = f"{metric} (%)"
        hover_suffix = "%"
    else:
        ylabel = metric
        hover_suffix = ""

    title_frequency = "Hourly pointwise" if frequency == "hourly" else "Daily"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=model_1_errors["plot_time"],
            y=model_1_errors["error"],
            mode="lines",
            name=MODEL_1_NAME,
            hovertemplate=(
                "Time: %{x}<br>"
                f"{metric}: " + "%{y:.3f}" + hover_suffix +
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=model_2_errors["plot_time"],
            y=model_2_errors["error"],
            mode="lines",
            name=MODEL_2_NAME,
            hovertemplate=(
                "Time: %{x}<br>"
                f"{metric}: " + "%{y:.3f}" + hover_suffix +
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"{title_frequency} {metric} error from {START_TIME} to {END_TIME}",
        xaxis_title="Time",
        yaxis_title=ylabel,
        template="plotly_white",
        hovermode="x unified",
        legend_title="Model",
        width=1200,
        height=600,
    )

    fig.show()


def plot_predictions():
    frequency = FREQUENCY.lower()

    model_1_df = load_model_data(MODEL_1_FILE)
    model_2_df = load_model_data(MODEL_2_FILE)

    plot_df = prepare_prediction_series(model_1_df, model_2_df, frequency)

    title_frequency = "Hourly" if frequency == "hourly" else "Daily average"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["plot_time"],
            y=plot_df["actual"],
            mode="lines",
            name="Actual DKPrice",
            hovertemplate=(
                "Time: %{x}<br>"
                "Actual DKPrice: %{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["plot_time"],
            y=plot_df["prediction_model_1"],
            mode="lines",
            name=f"{MODEL_1_NAME} prediction",
            hovertemplate=(
                "Time: %{x}<br>"
                f"{MODEL_1_NAME}: " + "%{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["plot_time"],
            y=plot_df["prediction_model_2"],
            mode="lines",
            name=f"{MODEL_2_NAME} prediction",
            hovertemplate=(
                "Time: %{x}<br>"
                f"{MODEL_2_NAME}: " + "%{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=(
            f"{title_frequency} actual price and predictions "
            f"from {START_TIME} to {END_TIME}"
        ),
        xaxis_title="Time",
        yaxis_title="DKPrice",
        template="plotly_white",
        hovermode="x unified",
        legend_title="Series",
        width=1200,
        height=600,
    )

    fig.show()


def main():
    plot_type = PLOT_TYPE.lower()

    if plot_type == "errors":
        plot_errors()
    elif plot_type == "predictions":
        plot_predictions()
    else:
        raise ValueError("PLOT_TYPE must be either 'errors' or 'predictions'")


if __name__ == "__main__":
    main()