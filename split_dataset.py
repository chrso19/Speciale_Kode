def num_folds(data_series, training_window, forecast_horizon):
    """
    Calculates the number of folds that the training and validation should be split into.

    Args:
    data_series: The full dataset used for the model
    training_window: The number of hours the model is trained on
    forecast_horizon: The number of hours the model will forecast
    """
    num_folds = (len(data_series) - training_window - forecast_horizon) // forecast_horizon
    
    return num_folds

def split_dataset(data_series, fold_num, training_window, forecast_horizon):
    """
    Splits the full dataset based on the fold number, the training window, and the forecast horizon.

    Args:
        data_series: The full dataset used for the model
        fold_num: The fold that is initialized
        training_window: The number of hours the model is trained on
        forecast_horizon: The number of hours the model will forecast
    """
    start = fold_num * forecast_horizon

    train_data = data_series[start:start + training_window]
    val_data = data_series[start + training_window:start + training_window + forecast_horizon]

    return train_data, val_data