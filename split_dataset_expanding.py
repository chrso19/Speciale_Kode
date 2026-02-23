def split_dataset(data_series, fold_num, training_window, forecast_horizon):
    """
    Splits the full dataset based on the fold number, the training window, and the forecast horizon.

    Args:
        data_series: The full dataset used for the model
        fold_num: The fold that is initialized
        training_window: The number of hours the model is trained on
        forecast_horizon: The number of hours the model will forecast
    """
    start = 0

    train_data = data_series[start:(fold_num + 1) * training_window]
    val_data = data_series[(fold_num + 1) * training_window:(fold_num + 1) * training_window + forecast_horizon]

    return train_data, val_data