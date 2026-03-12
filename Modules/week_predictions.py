import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings


def get_predictions(
    model,
    dataset: pd.DataFrame,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    forecast_horizon: int = 168,
    fitted_scaler = None
):
    """
    Predicts from val_start to val_end in blocks of forecast_horizon hours.

    For each block:
    - known future features use their true future values
    - weather features use true future values as simulated forecasts
    - persistence features use the last known value at the forecast origin
    - recursive lag features are updated hour by hour using simulated values

    Parameters
    ----------
    model : fitted model
        Already-trained model with a .predict() method.
    dataset : pd.DataFrame
        Full dataset for one zone from load_data().
    val_start : pd.Timestamp
        Start of validation period.
    val_end : pd.Timestamp
        End of validation period.
    forecast_horizon : int, default 168
        Number of hours predicted per block.

    Returns
    -------
    dict
        Dictionary with one dataframe per block:
        {
            1: pd.DataFrame(...),
            2: pd.DataFrame(...),
            ...
        }
        Each dataframe contains ["Time", "Prediction"].
    """

    data = dataset.copy()

    # Assumptions from your specification
    target_col = data.columns[0]
    feature_columns = [col for col in data.columns[1:] if col != "Time"]

    known_future_features = ["Year", "Month", "Day", "WeekDay", "Hour"]
    weather_features = ["WindSpeed", "Radiation"]
    persist_features = [
        'OffshoreWindPower',
        'OnshoreWindPower',
        'HydroPower',
        'SolarPower',
        'Biomass',
        'Biogas',
        'Waste',
        'FossilGas',
        'FossilOil',
        'FossilHardCoal',
        'ExchangeGreatBelt',
        'ExchangeGermany',
        'ExchangeSweden',
        'ExchangeNorway',
        'ExchangeNetherlands',
        'DEPrice',
        'NO2Price',
        'SE3Price',
        'SE4Price',
        'NLPrice',
        'OffshoreWindCapacity',
        'OnshoreWindCapacity',
        'SolarPowerCapacity',
        'GrossCon',
        'TotalProduction',
    ]
    recursive_features = [
        "Price_lag1", "Price_lag24", "Price_lag168",
        "GrossCon_lag1", "GrossCon_lag24", "GrossCon_lag168"
    ]
    dropped_features = ["Time"]

    # Check that all feature columns are categorized
    categorized_features = (
        known_future_features
        + weather_features
        + persist_features
        + recursive_features
        + dropped_features
    )

    uncategorized = [col for col in feature_columns if col not in categorized_features]

    if uncategorized:
        raise ValueError(
            "The following feature columns are not categorized and would leak "
            "future information into the prediction dataset:\n"
            + "\n".join(uncategorized)
        )
    
    # Check that no categorized features are missing from the dataset
    missing_features = [col for col in categorized_features if col not in feature_columns and col not in dropped_features]

    if missing_features:
        warnings.warn(
        "Some categorized features are not present in the dataset:\n"
        + "\n".join(missing_features)
        + "\nThey will simply be ignored."
    )

    # Validation timestamps
    horizon_mask = (data["Time"] >= val_start) & (data["Time"] <= val_end)
    horizon_hours = data.loc[horizon_mask, "Time"].tolist()

    if len(horizon_hours) == 0:
        raise ValueError("No timestamps found between val_start and val_end.")

    # History available before first validation timestamp
    simulated_history = data.loc[data["Time"] < val_start].copy().reset_index(drop=True)

    if simulated_history.empty:
        raise ValueError("Need historical data before val_start to build lag features.")

    block_predictions = {}
    block_no = 1
    block_start_idx = 0

    while block_start_idx < len(horizon_hours):
        block_hours = horizon_hours[block_start_idx:block_start_idx + forecast_horizon]
        block_rows = []

        # Forecast origin = last known row before current block starts
        origin_history = simulated_history.copy()
        last_known_row = origin_history.iloc[-1]

        # Freeze persistence variables for the whole block
        persist_values = {
            col: last_known_row[col]
            for col in persist_features
        }

        for h in block_hours:
            base_row = data.loc[data["Time"] == h].iloc[0].copy()
            new_row = {}
            new_row["Time"] = h

            # 1. Known future features
            for col in known_future_features:
                new_row[col] = base_row[col]

            # 2. Weather features: use true future values as simulated forecasts
            for col in weather_features:
                new_row[col] = base_row[col]

            # 3. Persistence features: use last known value at block start
            for col, val in persist_values.items():
                new_row[col] = val

            # 4. Recursive features
            temp_history = simulated_history.copy()

            if block_rows:
                temp_future = pd.DataFrame(block_rows)

                # Align columns to temp_history
                for col in temp_history.columns:
                    if col not in temp_future.columns:
                        temp_future[col] = np.nan
                temp_future = temp_future[temp_history.columns]

                temp_history = pd.concat([temp_history, temp_future], ignore_index=True)

            # Price lags
            if "Price_lag1" in data.columns:
                new_row["Price_lag1"] = temp_history[target_col].iloc[-1]
            if "Price_lag24" in data.columns:
                new_row["Price_lag24"] = temp_history[target_col].iloc[-24]
            if "Price_lag168" in data.columns:
                new_row["Price_lag168"] = temp_history[target_col].iloc[-168]

            # GrossCon lags
            current_grosscon = new_row.get("GrossCon", np.nan)

            if "GrossCon_lag1" in data.columns:
                if len(block_rows) >= 1:
                    new_row["GrossCon_lag1"] = block_rows[-1]["GrossCon"]
                else:
                    new_row["GrossCon_lag1"] = temp_history["GrossCon"].iloc[-1]

            if "GrossCon_lag24" in data.columns:
                if len(block_rows) >= 24:
                    new_row["GrossCon_lag24"] = block_rows[-24]["GrossCon"]
                else:
                    remaining = 24 - len(block_rows)
                    new_row["GrossCon_lag24"] = temp_history["GrossCon"].iloc[-remaining]

            if "GrossCon_lag168" in data.columns:
                if len(block_rows) >= 168:
                    new_row["GrossCon_lag168"] = block_rows[-168]["GrossCon"]
                else:
                    remaining = 168 - len(block_rows)
                    new_row["GrossCon_lag168"] = temp_history["GrossCon"].iloc[-remaining]

            # Predict one hour
            X_row = pd.DataFrame([new_row])[feature_columns]

            if fitted_scaler is not None:
                X_row = fitted_scaler.transform(X_row)
            y_pred = model.predict(X_row)[0]

            # Store simulated target and block output
            new_row[target_col] = y_pred
            new_row["Prediction"] = y_pred

            if "GrossCon" in data.columns:
                new_row["GrossCon"] = current_grosscon

            block_rows.append(new_row)

        block_df = pd.DataFrame(block_rows)
        block_predictions[block_no] = block_df[["Time", "Prediction"]].copy()

        # Append simulated block to history for next block
        history_append = block_df.copy()
        for col in data.columns:
            if col not in history_append.columns:
                history_append[col] = np.nan
        history_append = history_append[data.columns]

        simulated_history = pd.concat([simulated_history, history_append], ignore_index=True)

        block_no += 1
        block_start_idx += forecast_horizon

    return block_predictions