import pandas as pd
import numpy as np
import warnings
from skforecast.recursive import ForecasterEquivalentDate


def _build_persist_forecast(
    history: pd.DataFrame,
    future_times: list,
    features: list,
    offset: int = 168,
    n_offsets: int = 2,
    agg_func=np.mean,
) -> dict:
    """
    Uses ForecasterEquivalentDate to forecast each persist feature over the
    upcoming block. Fits one forecaster per feature on the available history,
    then predicts `len(future_times)` steps ahead.

    Parameters
    ----------
    history      : DataFrame with a "Time" column and all persist feature columns,
                   covering rows up to (not including) the block start.
    future_times : list of pd.Timestamp — the hours to forecast.
    features     : list of persist feature column names to forecast.
    offset       : seasonal period in hours (default 168 = 1 week).
    n_offsets    : number of past equivalent periods to average over (default 2).
    agg_func     : aggregation function applied across the n_offsets
                   (default np.mean; could also use np.median).

    Returns
    -------
    dict  {col: np.ndarray of length len(future_times)}
    """
    steps = len(future_times)

    # ForecasterEquivalentDate requires a DatetimeIndex with a freq set
    hist_indexed = history.set_index("Time")
    hist_indexed.index = pd.DatetimeIndex(hist_indexed.index, freq="h")

    result = {}

    for col in features:
        series = hist_indexed[col].dropna().astype(np.float64)

        forecaster = ForecasterEquivalentDate(
            offset=offset,
            n_offsets=n_offsets,
            agg_func=agg_func,
        )
        forecaster.fit(y=series)
        preds = forecaster.predict(steps=steps)
        result[col] = preds.values

    return result


def get_predictions(
    model,
    dataset: pd.DataFrame,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    forecast_horizon: int = 168,
    fitted_scaler=None,
    persist_offset: int = 168,
    persist_n_offsets: int = 2,
    persist_agg_func=np.mean,
):
    """
    Predicts from val_start to val_end in blocks of forecast_horizon hours.

    For each block:
    - known_future features  : true future calendar values
    - weather features       : true future values (simulated perfect forecast)
    - persist features       : ForecasterEquivalentDate forecast, using the
                               last `persist_n_offsets` equivalent periods of
                               `persist_offset` hours (default: mean of the
                               last 2 weeks at the same hour)
    - recursive lag features : updated hour-by-hour using simulated price /
                               GrossCon values from previous hours in the block

    Parameters
    ----------
    model               : fitted sklearn-compatible model
    dataset             : full DataFrame for one zone (DKPrice first, Time present)
    val_start           : start of validation period
    val_end             : end of validation period
    forecast_horizon    : hours per prediction block (default 168)
    fitted_scaler       : StandardScaler already fitted on training features, or None
    persist_offset      : seasonal period for ForecasterEquivalentDate (default 168)
    persist_n_offsets   : number of past periods to average (default 2)
    persist_agg_func    : aggregation function across periods (default np.mean)

    Returns
    -------
    dict  {block_no: pd.DataFrame with columns ["Time", "Prediction"]}
    """

    data = dataset.copy().reset_index(drop=True)

    target_col = data.columns[0]
    feature_columns = [col for col in data.columns[1:] if col != "Time"]

    known_future_features = [
        "Year", "Month", "Day", "WeekDay", "Hour",
        "OffshoreWindCapacity", "OnshoreWindCapacity", "SolarPowerCapacity",
    ]
    weather_features = ["WindSpeed", "Radiation"]
    persist_features = [
        'OffshoreWindPower', 'OnshoreWindPower', 'HydroPower', 'SolarPower',
        'Biomass', 'Biogas', 'Waste', 'FossilGas', 'FossilOil',
        'FossilHardCoal', 'ExchangeGreatBelt', 'ExchangeGermany',
        'ExchangeSweden', 'ExchangeNorway', 'ExchangeNetherlands',
        'DEPrice', 'NO2Price', 'SE3Price', 'SE4Price', 'NLPrice',
        'GrossCon', 'TotalProduction',
    ]
    recursive_lag_features = [
        "Price_lag1", "Price_lag24", "Price_lag168",
        "GrossCon_lag1", "GrossCon_lag24", "GrossCon_lag168",
    ]
    dropped_features = ["Time"]

    # --- sanity checks -------------------------------------------------------
    categorized = (
        known_future_features + weather_features + persist_features
        + recursive_lag_features + dropped_features
    )
    uncategorized = [c for c in feature_columns if c not in categorized]
    if uncategorized:
        raise ValueError(
            "The following feature columns are not categorized and would leak "
            "future information:\n" + "\n".join(uncategorized)
        )

    missing = [
        c for c in categorized
        if c not in feature_columns and c not in dropped_features
    ]
    if missing:
        warnings.warn(
            "Some categorized features are not present in the dataset:\n"
            + "\n".join(missing) + "\nThey will be ignored."
        )

    # Only forecast persist features that actually exist in this dataset
    active_persist = [f for f in persist_features if f in data.columns]

    # Which recursive lag features exist?
    has_price_lag1   = "Price_lag1"      in data.columns
    has_price_lag24  = "Price_lag24"     in data.columns
    has_price_lag168 = "Price_lag168"    in data.columns
    has_gc_lag1      = "GrossCon_lag1"   in data.columns
    has_gc_lag24     = "GrossCon_lag24"  in data.columns
    has_gc_lag168    = "GrossCon_lag168" in data.columns
    has_grosscon     = "GrossCon"        in data.columns

    # O(1) Time → row index lookup
    time_to_idx = {t: i for i, t in enumerate(data["Time"])}

    # --- validation horizon --------------------------------------------------
    horizon_mask  = (data["Time"] >= val_start) & (data["Time"] <= val_end)
    horizon_hours = data.loc[horizon_mask, "Time"].tolist()
    if not horizon_hours:
        raise ValueError("No timestamps found between val_start and val_end.")

    hist_mask = data["Time"] < val_start
    simulated_history = data.loc[hist_mask].copy().reset_index(drop=True)
    if simulated_history.empty:
        raise ValueError("Need historical data before val_start to build lag features.")

    # Rolling price and GrossCon histories as plain lists (O(1) append/index)
    LOOKBACK = 168
    price_history    = list(simulated_history[target_col].values[-LOOKBACK:])
    grosscon_history = (
        list(simulated_history["GrossCon"].values[-LOOKBACK:])
        if has_grosscon else []
    )

    # persist_history needs enough past weeks to cover n_offsets * offset hours
    PERSIST_LOOKBACK = (persist_n_offsets + 1) * persist_offset
    persist_history = simulated_history.tail(PERSIST_LOOKBACK).copy()

    # -------------------------------------------------------------------------
    block_predictions = {}
    block_no = 1
    block_start_idx = 0

    col_to_feat_idx = {col: i for i, col in enumerate(feature_columns)}
    future_col_names = known_future_features + weather_features
    future_col_idx   = {col: i for i, col in enumerate(future_col_names)}

    while block_start_idx < len(horizon_hours):
        block_hours = horizon_hours[block_start_idx:block_start_idx + forecast_horizon]
        n_block = len(block_hours)

        # --- ForecasterEquivalentDate: forecast all persist features at once -
        persist_fc = _build_persist_forecast(
            history=persist_history,
            future_times=block_hours,
            features=active_persist,
            offset=persist_offset,
            n_offsets=persist_n_offsets,
            agg_func=persist_agg_func,
        )

        # Pre-extract known future + weather values for the whole block
        future_rows = data.iloc[[time_to_idx[h] for h in block_hours]][
            future_col_names
        ].values  # shape (n_block, n_future_cols)

        block_price_preds   = []
        block_grosscon_vals = []

        for step in range(n_block):
            feat_vec = np.empty(len(feature_columns))

            # 1. Known future features
            for col in known_future_features:
                if col in col_to_feat_idx:
                    feat_vec[col_to_feat_idx[col]] = future_rows[step, future_col_idx[col]]

            # 2. Weather features
            for col in weather_features:
                if col in col_to_feat_idx:
                    feat_vec[col_to_feat_idx[col]] = future_rows[step, future_col_idx[col]]

            # 3. Persist features — from ForecasterEquivalentDate
            for col, arr in persist_fc.items():
                if col in col_to_feat_idx:
                    feat_vec[col_to_feat_idx[col]] = arr[step]

            # 4. Recursive price lags
            # If history is shorter than the lag (only possible at the very start),
            # fall back to the earliest available value rather than NaN
            ph = len(price_history)
            if has_price_lag1 and "Price_lag1" in col_to_feat_idx:
                feat_vec[col_to_feat_idx["Price_lag1"]]   = price_history[max(-ph, -1)]
            if has_price_lag24 and "Price_lag24" in col_to_feat_idx:
                feat_vec[col_to_feat_idx["Price_lag24"]]  = price_history[max(-ph, -24)]
            if has_price_lag168 and "Price_lag168" in col_to_feat_idx:
                feat_vec[col_to_feat_idx["Price_lag168"]] = price_history[max(-ph, -168)]

            # 5. Recursive GrossCon lags
            gh = len(grosscon_history)
            if has_gc_lag1 and "GrossCon_lag1" in col_to_feat_idx:
                feat_vec[col_to_feat_idx["GrossCon_lag1"]]   = grosscon_history[max(-gh, -1)]
            if has_gc_lag24 and "GrossCon_lag24" in col_to_feat_idx:
                feat_vec[col_to_feat_idx["GrossCon_lag24"]]  = grosscon_history[max(-gh, -24)]
            if has_gc_lag168 and "GrossCon_lag168" in col_to_feat_idx:
                feat_vec[col_to_feat_idx["GrossCon_lag168"]] = grosscon_history[max(-gh, -168)]

            # --- predict -----------------------------------------------------
            # Replace any remaining NaNs (e.g. from persist forecast gaps) with
            # column means from the scaler to avoid breaking NaN-intolerant models
            nan_mask = np.isnan(feat_vec)
            if nan_mask.any():
                if fitted_scaler is not None:
                    feat_vec[nan_mask] = fitted_scaler.mean_[nan_mask]
                else:
                    feat_vec[nan_mask] = 0.0

            X_row = feat_vec.reshape(1, -1)
            if fitted_scaler is not None:
                X_row = fitted_scaler.transform(X_row)
            y_pred = model.predict(X_row)[0]

            # Update rolling histories for next step's lag lookups
            price_history.append(y_pred)
            if len(price_history) > LOOKBACK:
                price_history.pop(0)

            gc_val = persist_fc["GrossCon"][step] if "GrossCon" in persist_fc else np.nan
            block_grosscon_vals.append(gc_val)
            if has_grosscon:
                grosscon_history.append(gc_val)
                if len(grosscon_history) > LOOKBACK:
                    grosscon_history.pop(0)

            block_price_preds.append(y_pred)

        # Store block result
        block_predictions[block_no] = pd.DataFrame({
            "Time": block_hours,
            "Prediction": block_price_preds,
        })

        # Extend persist_history with this block's simulated rows so that the
        # next block's ForecasterEquivalentDate fits can reach back into it
        simulated_block = data.iloc[[time_to_idx[h] for h in block_hours]].copy()
        simulated_block[target_col] = block_price_preds
        if has_grosscon:
            simulated_block["GrossCon"] = block_grosscon_vals
        persist_history = pd.concat(
            [persist_history, simulated_block], ignore_index=True
        ).tail(PERSIST_LOOKBACK)

        block_no += 1
        block_start_idx += forecast_horizon

    return block_predictions