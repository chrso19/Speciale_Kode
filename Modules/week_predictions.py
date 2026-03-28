import glob
import os
import re
import tempfile
import warnings
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

RF_MODEL_CACHE_ENV = "WANDB_RF_MODEL_DIR"
DEFAULT_RF_MODEL_CACHE_DIR = r"C:\Users\n_and\Documents\Data Science\Speciale\Shallow_learners\Artifacts"
try:
    from skforecast.recursive import ForecasterEquivalentDate
except ImportError:
    try:
        from skforecast import ForecasterEquivalentDate
    except ImportError:
        ForecasterEquivalentDate = None


KNOWN_FUTURE_FEATURES = ["Year", "Month", "Day", "WeekDay", "Hour"]
WEATHER_FEATURES = ["WindSpeed", "Radiation"]
CAPACITY_FEATURES = [
    "OffshoreWindCapacity",
    "OnshoreWindCapacity",
    "SolarPowerCapacity",
]

FORECAST_METHODS = {
    "DK1": {
        "OffshoreWindPower": "rf",
        "OnshoreWindPower": "rf",
        "HydroPower": "lag24",
        "SolarPower": "rf",
        "Biomass": "lag24",
        "Biogas": "skforecast",
        "Waste": "lag24",
        "FossilGas": "skforecast",
        "FossilOil": "lag24",
        "FossilHardCoal": "lag24",
        "ExchangeGreatBelt": "skforecast",
        "ExchangeGermany": "last_known_value",
        "ExchangeSweden": "lag24",
        "ExchangeNorway": "last_known_value",
        "ExchangeNetherlands": "skforecast",
        "DEPrice": "skforecast",
        "NO2Price": "skforecast",
        "SE3Price": "skforecast",
        "SE4Price": "skforecast",
        "NLPrice": "skforecast",
        "GrossCon": "skforecast",
        "TotalProduction": "rf",
    },
    "DK2": {
        "OffshoreWindPower": "rf",
        "OnshoreWindPower": "rf",
        "HydroPower": "lag168",
        "SolarPower": "rf",
        "Biomass": "lag24",
        "Biogas": "skforecast",
        "Waste": "lag24",
        "FossilGas": "skforecast",
        "FossilOil": "lag24",
        "FossilHardCoal": "lag24",
        "ExchangeGreatBelt": "skforecast",
        "ExchangeGermany": "lag24",
        "ExchangeSweden": "last_known_value",
        "ExchangeNorway": "last_known_value",
        "ExchangeNetherlands": "last_known_value",
        "DEPrice": "skforecast",
        "NO2Price": "skforecast",
        "SE3Price": "skforecast",
        "SE4Price": "skforecast",
        "NLPrice": "skforecast",
        "GrossCon": "skforecast",
        "TotalProduction": "rf",
    },
}

RF_FEATURE_INPUTS = {
    "OffshoreWindPower": [
        "WindSpeed",
        "OffshoreWindCapacity",
        "OffshoreWindPower_lag24",
        "OffshoreWindPower_lag168",
        "Month",
        "WeekDay",
        "Hour",
    ],
    "OnshoreWindPower": [
        "WindSpeed",
        "OnshoreWindCapacity",
        "OnshoreWindPower_lag24",
        "OnshoreWindPower_lag168",
        "Month",
        "WeekDay",
        "Hour",
    ],
    "SolarPower": [
        "Radiation",
        "SolarPowerCapacity",
        "SolarPower_lag168",
        "Month",
        "WeekDay",
        "Hour",
    ],
    "TotalProduction": [
        "WindSpeed",
        "Radiation",
        "OffshoreWindCapacity",
        "OnshoreWindCapacity",
        "SolarPowerCapacity",
        "TotalProduction_lag24",
        "TotalProduction_lag168",
        "Month",
        "WeekDay",
        "Hour",
    ],
}

FORECASTABLE_FEATURES = sorted(
    set(CAPACITY_FEATURES)
    | set(FORECAST_METHODS["DK1"].keys())
    | set(FORECAST_METHODS["DK2"].keys())
)

BASE_ALIASES = {
    "Price": "DKPrice",
}


@lru_cache(maxsize=None)
def _load_wandb_rf_model(feature: str, dk_zone: str):
    import wandb

    artifact_name = f"Energinet_speciale/Shallow_learners/rf_{feature}_{dk_zone}:latest"
    api = wandb.Api(timeout=60)
    artifact = api.artifact(artifact_name)

    base_cache_dir = os.getenv(RF_MODEL_CACHE_ENV)
    if base_cache_dir is None:
        base_cache_dir = DEFAULT_RF_MODEL_CACHE_DIR
    if base_cache_dir:
        cache_dir = os.path.join(base_cache_dir, feature, dk_zone)
    else:
        cache_dir = os.path.join(tempfile.gettempdir(), "wandb_rf_cache", feature, dk_zone)
    os.makedirs(cache_dir, exist_ok=True)
    artifact_dir = artifact.download(root=cache_dir)

    candidates = []
    for pattern in ("*.joblib", "*.pkl", "*.pickle"):
        candidates.extend(glob.glob(os.path.join(artifact_dir, "**", pattern), recursive=True))

    if not candidates:
        raise FileNotFoundError(
            f"No serialized model file found in W&B artifact {artifact_name}."
        )

    return joblib.load(candidates[0])


def _resolve_dk_zone(dataset: pd.DataFrame, dk_zone: str | None) -> str:
    if dk_zone is not None:
        if dk_zone not in {"DK1", "DK2"}:
            raise ValueError("dk_zone must be 'DK1' or 'DK2'.")
        return dk_zone

    inferred = dataset.attrs.get("dk_zone")
    if inferred in {"DK1", "DK2"}:
        return inferred

    raise ValueError(
        "dk_zone could not be inferred from dataset.attrs['dk_zone']. "
        "Pass dk_zone='DK1' or dk_zone='DK2' to get_predictions()."
    )


def _parse_lag_feature(col_name: str):
    match = re.match(r"^(.*)_lag(\d+)$", col_name)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


def _resolve_base_feature(base_feature: str, target_col: str) -> str:
    if base_feature == target_col:
        return target_col
    return BASE_ALIASES.get(base_feature, base_feature)



def _series_up_to_time(
    history_df: pd.DataFrame,
    block_df: pd.DataFrame,
    feature_name: str,
    current_time: pd.Timestamp | None = None,
) -> pd.Series:
    hist = history_df[["Time", feature_name]].copy()

    if feature_name in block_df.columns:
        future = block_df[["Time", feature_name]].copy()
    else:
        future = pd.DataFrame({
            "Time": block_df["Time"],
            feature_name: np.nan
        })

    if current_time is not None:
        future = future.loc[future["Time"] < current_time]

    combined = pd.concat([hist, future], ignore_index=True)
    combined = combined.dropna(subset=[feature_name]).sort_values("Time")
    return combined.set_index("Time")[feature_name]



def _value_from_lag(
    history_df: pd.DataFrame,
    block_df: pd.DataFrame,
    feature_name: str,
    timestamp: pd.Timestamp,
    lag_hours: int,
) -> float:
    series = _series_up_to_time(history_df, block_df, feature_name)
    ref_time = timestamp - pd.Timedelta(hours=lag_hours)
    if ref_time not in series.index:
        raise ValueError(
            f"Cannot compute {feature_name}_lag{lag_hours} for {timestamp}. "
            f"Missing historical value at {ref_time}."
        )
    return float(series.loc[ref_time])




def _forecast_lag_feature_for_week(
    history_df: pd.DataFrame,
    block_df: pd.DataFrame,
    feature_name: str,
    lag_hours: int,
):
    preds = []
    for timestamp in block_df["Time"]:
        preds.append(
            _value_from_lag(history_df, block_df, feature_name, timestamp, lag_hours)
        )
        block_df.loc[block_df["Time"] == timestamp, feature_name] = preds[-1]
    return preds



def _forecast_last_known_value_for_week(history_df: pd.DataFrame, feature_name: str, horizon: int):
    value = float(history_df[feature_name].iloc[-1])
    return [value] * horizon



def _forecast_skforecast_feature_for_week(
    history_df: pd.DataFrame,
    block_df: pd.DataFrame,
    feature_name: str,
    offset: int = 168,
    n_offsets: int = 3,
    agg_func = np.mean,
):
    if ForecasterEquivalentDate is None:
        raise ImportError(
            "skforecast is not installed. Install it with e.g. 'pip install skforecast' "
            "to use method='mean' equivalent-date forecasts."
        )

    hist_series = (
        history_df[["Time", feature_name]]
        .dropna()
        .sort_values("Time")
        .drop_duplicates(subset="Time")
        .set_index("Time")[feature_name]
        .astype(float)
    )

    hist_series = hist_series.asfreq("h")

    required_history = offset * n_offsets
    if len(hist_series) < required_history:
        raise ValueError(
            f"Not enough history to train skforecast equivalent-date model for {feature_name}. "
            f"Need at least {required_history} hourly observations, got {len(hist_series)}."
        )

    forecaster = ForecasterEquivalentDate(
        offset=offset,
        n_offsets=n_offsets,
        agg_func=agg_func,
    )
    forecaster.fit(y=hist_series)

    preds = forecaster.predict(steps=len(block_df))
    preds = np.asarray(preds, dtype=float).reshape(-1).tolist()

    block_df.loc[:, feature_name] = preds
    return preds



def _build_rf_input_row(
    history_df: pd.DataFrame,
    block_df: pd.DataFrame,
    feature_name: str,
    timestamp: pd.Timestamp,
    input_columns: list[str] | None = None,
) -> pd.DataFrame:
    row = {"Time": timestamp}
    current_row = block_df.loc[block_df["Time"] == timestamp].iloc[0]

    columns = input_columns if input_columns is not None else RF_FEATURE_INPUTS[feature_name]

    for col in columns:
        base_feature, lag_hours = _parse_lag_feature(col)
        if lag_hours is not None:
            resolved_base = _resolve_base_feature(base_feature, feature_name)
            row[col] = _value_from_lag(history_df, block_df, resolved_base, timestamp, lag_hours)
        else:
            if col not in current_row.index:
                raise ValueError(f"RF input column '{col}' not available for {feature_name}.")
            row[col] = current_row[col]

    return pd.DataFrame([row])[columns]



def _forecast_rf_feature_for_week(
    history_df: pd.DataFrame,
    block_df: pd.DataFrame,
    feature_name: str,
    dk_zone: str,
    rf_models: dict
):
    """
    Forecast an RF feature for a week using pre-trained RF models.
    
    Parameters
    ----------
    history_df : pd.DataFrame
        Historical data before the forecast period.
    block_df : pd.DataFrame
        Data for the current forecast block.
    feature_name : str
        Name of the feature to forecast.
    dk_zone : str
        DK zone ('DK1' or 'DK2').
    rf_models : dict
        Dictionary mapping feature names to pre-trained RF models.
        E.g., {"OffshoreWindPower_DK1": model, ...}
    """
    preds = []

    # Get the RF model for this feature in this zone
    model_key = f"{feature_name}_{dk_zone}"
    if model_key not in rf_models:
        raise ValueError(
            f"RF model for '{feature_name}' in zone '{dk_zone}' not found in rf_models. "
            f"Available keys: {list(rf_models.keys())}"
        )
    
    rf_model = rf_models[model_key]

    # Prefer model-declared feature order from training; fallback to static defaults.
    predictor = getattr(rf_model, "named_steps", {}).get("model") if hasattr(rf_model, "named_steps") else rf_model
    input_columns = (
        list(getattr(rf_model, "feature_names_in_", []))
        or list(getattr(predictor, "feature_names_in_", []))
        or RF_FEATURE_INPUTS[feature_name]
    )

    for timestamp in block_df["Time"]:
        X_row = _build_rf_input_row(
            history_df=history_df,
            block_df=block_df,
            feature_name=feature_name,
            timestamp=timestamp,
            input_columns=input_columns,
        )
        pred = float(rf_model.predict(X_row)[0])
        preds.append(pred)
        block_df.loc[block_df["Time"] == timestamp, feature_name] = pred

    return preds



def _categorize_feature_columns(feature_columns: list[str], target_col: str):
    uncategorized = []
    for col in feature_columns:
        if col in KNOWN_FUTURE_FEATURES + WEATHER_FEATURES + CAPACITY_FEATURES + FORECASTABLE_FEATURES:
            continue
        base_feature, lag_hours = _parse_lag_feature(col)
        if lag_hours is not None:
            resolved_base = _resolve_base_feature(base_feature, target_col)
            if resolved_base == target_col or resolved_base in FORECASTABLE_FEATURES:
                continue
        uncategorized.append(col)

    if uncategorized:
        raise ValueError(
            "The following feature columns are not categorized and may leak future information:\n"
            + "\n".join(uncategorized)
        )



def get_predictions(
    model,
    dataset: pd.DataFrame,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    forecast_horizon: int = 168,
    fitted_scaler=None,
    dk_zone: str | None = None,
    rf_models=None
):
    """
    Predict from val_start to val_end in week-sized blocks while constructing
    explanatory features the way they would be available in production.

    Rules implemented
    -----------------
    - Calendar features use true future values.
    - Weather features (WindSpeed, Radiation) use future DMI values.
    - Capacity features use their true future values.
    - LastKnownValue features are frozen from the forecast origin.
    - Lag24 / Lag168 features are calculated ad hoc.
    - 'skforecast' features are retrained ad hoc for each week using
      skforecast ForecasterEquivalentDate with offset=168, n_offsets=3,
      and method=mean.
    - RF features are forecasted using pre-trained RF models passed via rf_models parameter.

    Parameters
    ----------
    model : fitted model
        The already-trained target model.
    dataset : pd.DataFrame
        Full dataset for one DK zone.
    val_start : pd.Timestamp
        Validation start timestamp.
    val_end : pd.Timestamp
        Validation end timestamp.
    forecast_horizon : int, default 168
        Number of hours per prediction block.
    fitted_scaler : fitted scaler, optional
        Scaler used for the target model.
    dk_zone : {'DK1', 'DK2'}
        Zone for the supplied dataset.
    rf_models : dict, optional
        Dictionary mapping feature names to pre-trained RF models.
        Keys should be formatted as "{feature_name}_{dk_zone}" (e.g., "OffshoreWindPower_DK1").
        Required if dataset contains RF-forecasted features. If None and RF features are present,
        those features will be skipped with a warning.

    """

    data = dataset.copy().sort_values("Time").reset_index(drop=True)
    dk_zone = _resolve_dk_zone(data, dk_zone)

    target_col = data.columns[0]
    feature_columns = [col for col in data.columns[1:] if col != "Time"]
    _categorize_feature_columns(feature_columns, target_col)

    horizon_mask = (data["Time"] >= val_start) & (data["Time"] <= val_end)
    horizon_hours = data.loc[horizon_mask, "Time"].tolist()
    if not horizon_hours:
        raise ValueError("No timestamps found between val_start and val_end.")

    simulated_history = data.loc[data["Time"] < val_start].copy().reset_index(drop=True)
    if simulated_history.empty:
        raise ValueError("Need historical data before val_start to build lag features.")

    present_forecast_features = [
        feat for feat in FORECASTABLE_FEATURES if feat in data.columns and feat != target_col
    ]
    non_rf_features = [
        feat
        for feat in present_forecast_features
        if FORECAST_METHODS[dk_zone].get(feat) in {"lag24", "lag168", "last_known_value", "skforecast"}
    ]
    rf_features = [
        feat for feat in present_forecast_features if FORECAST_METHODS[dk_zone].get(feat) == "rf"
    ]

    block_predictions = {}
    block_no = 1
    block_start_idx = 0

    while block_start_idx < len(horizon_hours):
        block_hours = horizon_hours[block_start_idx:block_start_idx + forecast_horizon]
        block_true = data.loc[data["Time"].isin(block_hours)].copy().reset_index(drop=True)
        block_df = pd.DataFrame({"Time": block_true["Time"]})

        # Start with independent features.
        for col in KNOWN_FUTURE_FEATURES + WEATHER_FEATURES + CAPACITY_FEATURES:
            if col in block_true.columns:
                block_df[col] = block_true[col].to_numpy()

        # Forecast non-RF features one feature for the whole week before the next.
        for feature_name in non_rf_features:
            method = FORECAST_METHODS[dk_zone][feature_name]
            if method == "lag24":
                block_df[feature_name] = _forecast_lag_feature_for_week(
                    simulated_history, block_df, feature_name, 24
                )
            elif method == "lag168":
                block_df[feature_name] = _forecast_lag_feature_for_week(
                    simulated_history, block_df, feature_name, 168
                )
            elif method == "last_known_value":
                block_df[feature_name] = _forecast_last_known_value_for_week(
                    simulated_history, feature_name, len(block_df)
                )
            elif method == "skforecast":
                block_df[feature_name] = _forecast_skforecast_feature_for_week(
                    simulated_history, block_df, feature_name, offset=168, n_offsets=3, agg_func=np.mean
                )
            else:
                raise ValueError(f"Unsupported forecast method '{method}' for {feature_name}.")

        

        # RF features last.
        for feature_name in rf_features:
            if rf_models is None or not rf_models:
                warn_msg = (
                    f"RF models not provided. Skipping RF feature forecast for '{feature_name}'. "
                    f"Pass rf_models parameter to get_predictions() to forecast RF features."
                )
                warnings.warn(warn_msg)
                continue
            
            block_df[feature_name] = _forecast_rf_feature_for_week(
                simulated_history, block_df, feature_name, dk_zone, rf_models
            )

        # Predict target recursively hour by hour.
        target_rows = []
        for timestamp in block_df["Time"]:
            prepared_row = block_df.loc[block_df["Time"] == timestamp].iloc[0].to_dict()
            new_row = {"Time": timestamp}

            for col in feature_columns:
                if col in prepared_row:
                    new_row[col] = prepared_row[col]
                    continue

                base_feature, lag_hours = _parse_lag_feature(col)
                if lag_hours is not None:
                    resolved_base = _resolve_base_feature(base_feature, target_col)
                    if resolved_base == target_col:
                        temp_history = simulated_history.copy()
                        if target_rows:
                            target_future = pd.DataFrame(target_rows)
                            for req_col in temp_history.columns:
                                if req_col not in target_future.columns:
                                    target_future[req_col] = np.nan
                            target_future = target_future[temp_history.columns]
                            temp_history = pd.concat([temp_history, target_future], ignore_index=True)
                        temp_block = pd.DataFrame({"Time": [timestamp], resolved_base: [np.nan]})
                        new_row[col] = _value_from_lag(temp_history, temp_block, resolved_base, timestamp, lag_hours)
                    else:
                        new_row[col] = _value_from_lag(simulated_history, block_df, resolved_base, timestamp, lag_hours)
                    continue

                raise ValueError(
                    f"Could not build feature column '{col}' for target model prediction."
                )

            X_row = pd.DataFrame([new_row])[feature_columns]
            if fitted_scaler is not None:
                X_row = fitted_scaler.transform(X_row)
            y_pred = float(model.predict(X_row)[0])

            new_row[target_col] = y_pred
            new_row["Prediction"] = y_pred
            target_rows.append(new_row)

        block_out = pd.DataFrame(target_rows)
        block_predictions[block_no] = block_out[["Time", "Prediction"]].copy()

        history_append = block_out.copy()
        for col in data.columns:
            if col not in history_append.columns:
                history_append[col] = np.nan
        history_append = history_append[data.columns]
        simulated_history = pd.concat([simulated_history, history_append], ignore_index=True)

        block_no += 1
        block_start_idx += forecast_horizon

    return block_predictions
