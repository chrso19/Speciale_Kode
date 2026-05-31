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
FREQUENCY = "hourly"

# Choose bidding zone: "DK1", "DK2", or None
DK_ZONE = None

MODEL_FILES = [
    "Deep learners/Simple RNN/RNN multivariate/DK1_RNN_multi_predictions.csv",
    "Deep learners/LSTM Autoencoder/DK1_LSTM_AE_predictions.csv"
]

MODEL_FILES = [
    "Shallow learners/ARIMA-ARIMAX/DK1_ARIMA_predictions_2.csv",
    "Shallow learners/ARIMA-ARIMAX/DK1_ARIMAX_predictions_2.csv",
]

MODEL_FILES = [
    "Shallow learners/Final_eval/Shap/DK1_Lasso_predictions.csv",
    "Shallow learners/SVR/DK1_SVR_predictions.csv",
]

MODEL_FILES = [
    "Shallow learners/LightGBM/DK1_LightGBM_multi_predictions_1.csv",
    "Shallow learners/Final_eval/Shap/DK1_LightGBM_predictions.csv",
]

MODEL_FILES = [
    #"Deep learners/Final_eval/DK1_LSTM_predictions.csv",
    "Deep learners/LSTM/DK1_LSTM_predictions.csv",
    "Deep learners/Final_eval/DK1_LSTM_AE_predictions.csv"
]

MODEL_FILES = [
    "Deep learners/Final_eval/DK1_RNN_predictions.csv",
    "Deep learners/Final_eval/DK1_GRU_predictions.csv"
]

MODELS = [
    "RNN",
    "GRU",
    #"LSTM",
    #"LSTM AE"
]

START_TIME = "2025-01-01 00:00:00"
END_TIME = "2025-01-30 23:00:00"


def get_model_configs():
    if len(MODEL_FILES) != len(MODELS):
        raise ValueError(
            "MODELS and MODEL_FILES must have the same length. "
            f"Got {len(MODELS)} model names and {len(MODEL_FILES)} files."
        )

    if len(MODELS) == 0:
        raise ValueError("Add at least one model to MODELS and MODEL_FILES.")

    return list(zip(MODELS, MODEL_FILES))


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
    import csv as _csv

    file_path = DATA_FOLDER / file_name

    # Detect the actual field separator so that comma-sep, semicolon-sep, and
    # comma-decimal-with-comma-sep files are all handled correctly.
    try:
        with open(file_path, newline="", encoding="utf-8-sig") as _f:
            _sample = _f.read(8192)
        _dialect = _csv.Sniffer().sniff(_sample, delimiters=",;\t|")
        _sep = _dialect.delimiter
    except (_csv.Error, OSError):
        _sep = ","  # safe fallback

    df = pd.read_csv(file_path, sep=_sep, encoding="utf-8-sig")

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

    time_raw = df[TIME_COL].astype(str).str.strip()
    parsed_time = pd.to_datetime(time_raw, format="%Y-%m-%d %H:%M:%S", errors="coerce")

    if parsed_time.isna().any():
        fallback_parsers = [
            {"format": "%Y-%m-%dT%H:%M:%S", "errors": "coerce"},
            {"format": "%d-%m-%Y %H:%M:%S", "errors": "coerce"},
            {"format": "%d/%m/%Y %H:%M:%S", "errors": "coerce"},
            {"format": "mixed", "errors": "coerce"},
            {"format": "mixed", "dayfirst": True, "errors": "coerce"},
        ]

        for parser_kwargs in fallback_parsers:
            fallback_time = pd.to_datetime(time_raw, **parser_kwargs)
            if fallback_time.notna().sum() > parsed_time.notna().sum():
                parsed_time = fallback_time
            if parsed_time.notna().all():
                break

    if parsed_time.isna().any():
        bad_examples = time_raw[parsed_time.isna()].dropna().unique()[:5].tolist()
        raise ValueError(
            f"{file_name} has invalid values in '{TIME_COL}' that could not be parsed. "
            f"Examples: {bad_examples}"
        )

    df[TIME_COL] = parsed_time

    # Force numeric dtypes for metric math and cross-model subtraction.
    # Supports both dot and comma decimal separators from different CSV exports.
    for numeric_col in [ACTUAL_COL, PRED_COL]:
        raw_values = df[numeric_col]
        before_nulls = raw_values.isna().sum()

        normalized = raw_values.astype(str).str.strip()
        normalized = normalized.str.replace("\u00A0", "", regex=False)
        normalized = normalized.str.replace(" ", "", regex=False)

        comma_only_mask = normalized.str.contains(",", na=False) & ~normalized.str.contains(".", regex=False, na=False)
        normalized.loc[comma_only_mask] = normalized.loc[comma_only_mask].str.replace(",", ".", regex=False)

        both_separators_mask = normalized.str.contains(",", na=False) & normalized.str.contains(".", regex=False, na=False)
        normalized.loc[both_separators_mask] = (
            normalized.loc[both_separators_mask]
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )

        df[numeric_col] = pd.to_numeric(normalized, errors="coerce")
        new_nulls = df[numeric_col].isna().sum() - before_nulls

        if new_nulls > 0:
            failed_examples = raw_values[df[numeric_col].isna()].dropna().astype(str).unique()[:5]
            raise ValueError(
                f"{file_name} has {new_nulls} non-numeric values in '{numeric_col}'. "
                f"Examples: {failed_examples.tolist()}"
            )

    df = df.sort_values(TIME_COL).reset_index(drop=True)

    start_time = pd.to_datetime(START_TIME, format="%Y-%m-%d %H:%M:%S")
    end_time = pd.to_datetime(END_TIME, format="%Y-%m-%d %H:%M:%S")

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

def prepare_prediction_series(df, frequency):
    frequency = frequency.lower()

    series = df[[TIME_COL, ACTUAL_COL, PRED_COL]].copy().rename(
        columns={
            TIME_COL: "plot_time",
            ACTUAL_COL: "actual",
            PRED_COL: "prediction",
        }
    )

    if frequency == "hourly":
        return series[["plot_time", "actual", "prediction"]]

    if frequency == "daily":
        series["plot_time"] = series["plot_time"].dt.floor("D")

        daily = (
            series.groupby("plot_time")[["actual", "prediction"]]
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

    model_configs = get_model_configs()

    if metric in ["SMAPE", "MAPE"]:
        ylabel = f"{metric} (%)"
        hover_suffix = "%"
    else:
        ylabel = metric
        hover_suffix = ""

    title_frequency = "Hourly pointwise" if frequency == "hourly" else "Daily"

    fig = go.Figure()

    for model_name, model_file in model_configs:
        model_df = load_model_data(model_file)
        model_errors = prepare_error_series(model_df, metric, frequency)

        fig.add_trace(
            go.Scatter(
                x=model_errors["plot_time"],
                y=model_errors["error"],
                mode="lines",
                name=model_name,
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

    model_configs = get_model_configs()
    model_data = {
        model_name: prepare_prediction_series(load_model_data(model_file), frequency)
        for model_name, model_file in model_configs
    }

    first_model_name = model_configs[0][0]
    actual_df = model_data[first_model_name]

    for model_name in MODELS[1:]:
        actual_difference = (
            actual_df.set_index("plot_time")["actual"]
            .sub(model_data[model_name].set_index("plot_time")["actual"], fill_value=np.nan)
            .abs()
        )

        if actual_difference.dropna().max() > 1e-8:
            print(
                "Warning: Actual DKPrice values are not identical between "
                f"{first_model_name} and {model_name}. "
                "Using actual values from the first model file."
            )

    title_frequency = "Hourly" if frequency == "hourly" else "Daily average"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual_df["plot_time"],
            y=actual_df["actual"],
            mode="lines",
            name="Actual DKPrice",
            hovertemplate=(
                "Time: %{x}<br>"
                "Actual DKPrice: %{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    for model_name in MODELS:
        model_df = model_data[model_name]

        fig.add_trace(
            go.Scatter(
                x=model_df["plot_time"],
                y=model_df["prediction"],
                mode="lines",
                name=f"{model_name} prediction",
                hovertemplate=(
                    "Time: %{x}<br>"
                    f"{model_name}: " + "%{y:.3f}"
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
        width=1600,
        height=500,
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