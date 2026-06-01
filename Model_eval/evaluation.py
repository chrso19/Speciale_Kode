import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

###############################################################################
# CONFIGURATION
###############################################################################

DATA_FOLDER = "DK1"     # change to DK2 when needed

VERSION = 1
# VERSION = 1 -> Distribution Matching
# VERSION = 2 -> Error Distribution

OUTPUT_FILE = f"{DATA_FOLDER}_Model_Evaluation_version{VERSION}.xlsx"

###############################################################################
# MODEL CATEGORIES
###############################################################################

model_categories = {
    "Naive": "Baseline",
    "MovingAverage": "Baseline",
    "SeasonalNaive": "Baseline",


    "Lasso": "Shallow",
    "RandomForest": "Shallow",
    "XGBoost": "Shallow",
    "LightGBM": "Shallow",
    "SVR": "Shallow",
    "ARIMA": "Shallow",
    "ARIMAX": "Shallow",

    "LSTM": "Deep",
    "GRU": "Deep",
    "LSTM_AE": "Deep",
    "RNN": "Deep"
}

###############################################################################
# METRICS
###############################################################################

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0

    values = np.zeros_like(y_true, dtype=float)
    values[mask] = 2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]

    return 100 * np.mean(values)


def calculate_metrics(actual, pred, version):

    rmse = np.sqrt(
        mean_squared_error(actual, pred)
    )

    mae = mean_absolute_error(actual, pred)

    smape_value = smape(actual, pred)

    if version == 1:

        actual_q1 = np.percentile(actual, 25)
        actual_q2 = np.percentile(actual, 50)
        actual_q3 = np.percentile(actual, 75)

        pred_q1 = np.percentile(pred, 25)
        pred_q2 = np.percentile(pred, 50)
        pred_q3 = np.percentile(pred, 75)

        q1_metric = abs(pred_q1 - actual_q1)
        q2_metric = abs(pred_q2 - actual_q2)
        q3_metric = abs(pred_q3 - actual_q3)

    else:

        errors = np.abs(actual - pred)

        q1_metric = np.percentile(errors, 25)
        q2_metric = np.percentile(errors, 50)
        q3_metric = np.percentile(errors, 75)

    return {
        "SMAPE": smape_value,
        "RMSE": rmse,
        "MAE": mae,
        "Q1": q1_metric,
        "Q2": q2_metric,
        "Q3": q3_metric
    }

###############################################################################
# WEIGHTED SUM MODEL
###############################################################################

def weighted_sum_ranking(df):

    metrics = [
        "SMAPE",
        "RMSE",
        "MAE",
        "Q1",
        "Q2",
        "Q3"
    ]

    norm_df = df.copy()

    for metric in metrics:

        min_val = df[metric].min()
        max_val = df[metric].max()

        if max_val == min_val:
            norm_df[metric] = 1.0

        else:
            norm_df[metric] = (
                1
                - (df[metric] - min_val)
                / (max_val - min_val)
            )

    weight = 1 / len(metrics)

    norm_df["CompositeScore"] = (
        norm_df[metrics] * weight
    ).sum(axis=1)

    return norm_df

###############################################################################
# TOPSIS
###############################################################################

def topsis(df):

    metrics = [
        "SMAPE",
        "RMSE",
        "MAE",
        "Q1",
        "Q2",
        "Q3"
    ]

    X = df[metrics].values

    norm = np.sqrt(
        np.sum(X**2, axis=0)
    )

    X_norm = X / norm

    weights = np.repeat(
        1 / len(metrics),
        len(metrics)
    )

    X_weighted = X_norm * weights

    ideal_best = np.min(
        X_weighted,
        axis=0
    )

    ideal_worst = np.max(
        X_weighted,
        axis=0
    )

    d_best = np.sqrt(
        np.sum(
            (X_weighted - ideal_best)**2,
            axis=1
        )
    )

    d_worst = np.sqrt(
        np.sum(
            (X_weighted - ideal_worst)**2,
            axis=1
        )
    )

    score = d_worst / (d_best + d_worst)

    result = df.copy()

    result["TOPSIS_Score"] = score

    return result

###############################################################################
# MAIN EVALUATION
###############################################################################

results = []

os.getcwd()

for file in os.listdir(DATA_FOLDER):

    if not file.endswith(".csv"):
        continue

    model_name = os.path.splitext(file)[0]

    filepath = os.path.join(
        DATA_FOLDER,
        file
    )

    df = pd.read_csv(filepath)

    actual = df["DKPrice"].values
    pred = df["Predictions"].values

    metrics = calculate_metrics(
        actual,
        pred,
        VERSION
    )

    metrics["Model"] = model_name

    metrics["Category"] = (
        model_categories.get(
            model_name,
            "Unknown"
        )
    )

    results.append(metrics)

###############################################################################
# RESULTS DATAFRAME
###############################################################################

results_df = pd.DataFrame(results)

###############################################################################
# WSM LEADERBOARD
###############################################################################

wsm = weighted_sum_ranking(results_df)

wsm = wsm.sort_values(
    "CompositeScore",
    ascending=False
)

###############################################################################
# TOPSIS LEADERBOARD
###############################################################################

topsis_df = topsis(results_df)

topsis_df = topsis_df.sort_values(
    "TOPSIS_Score",
    ascending=False
)

###############################################################################
# CATEGORY WINNERS
###############################################################################

category_winners = (
    wsm.sort_values(
        "CompositeScore",
        ascending=False
    )
    .groupby("Category")
    .first()
    .reset_index()
)

###############################################################################
# METRIC LEADERBOARDS
###############################################################################

metric_leaderboards = {}

for metric in [
    "SMAPE",
    "RMSE",
    "MAE",
    "Q1",
    "Q2",
    "Q3"
]:

    metric_leaderboards[metric] = (
        results_df
        .sort_values(metric)
    )

###############################################################################
# EXPORT
###############################################################################

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    results_df.to_excel(
        writer,
        sheet_name="Raw Metrics",
        index=False
    )

    wsm.to_excel(
        writer,
        sheet_name="WSM Ranking",
        index=False
    )

    topsis_df.to_excel(
        writer,
        sheet_name="TOPSIS Ranking",
        index=False
    )

    category_winners.to_excel(
        writer,
        sheet_name="Category Winners",
        index=False
    )

    for metric, table in metric_leaderboards.items():

        table.to_excel(
            writer,
            sheet_name=f"{metric}_Ranking",
            index=False
        )

print("Evaluation complete.")
print(f"Results saved to {OUTPUT_FILE}")