from turtle import pd
import pandas as pd
import numpy as np


def num_folds(data_series, training_window, forecast_horizon):
    """
    Calculates the number of folds that the training and validation should be split into.

    Args:
    data_series: The full dataset used for training the model
    training_window: The number of hours the model is trained on
    forecast_horizon: The number of hours the model will forecast
    """
    num_folds = (len(data_series) - training_window - forecast_horizon) // forecast_horizon
    
    return num_folds

def split_dataset(data_series, fold_num, training_window, forecast_horizon):
    """
    Splits the full train dataset based on the fold number, the training window, 
    and the forecast horizon. Uses rolling window, with strides equal to the forecast horizon.
    Returns train window data and forecast_horizon data for the given fold number.

    Args:
        data_series: The full dataset used for training the model
        fold_num: The fold that is initialized
        training_window: The number of hours the model is trained on
        forecast_horizon: The number of hours the model will forecast
    """
    start = fold_num * forecast_horizon

    train_data = data_series[start:start + training_window]
    val_data = data_series[start + training_window:start + training_window + forecast_horizon]

    return train_data, val_data

def hyper_param_split(dataset:pd.DataFrame,
                      split_setup:int=1,
                      train_window:int=3*8760,
                      predict_horizon:int=17*168,
                      stride:int=17*168):
    """
    Splits the dataset into training and validation folds according
    to the chosen setup.
    
    Arguments:
    - dataset: the dataset to be split.
    - split_setup: choose the desired split setup number.
    - train_window: the number of hours the model is trained on.
    - predict_horizon: the number of hours the model will forecast.
    - stride: the number of hours to walk forward between folds.
    """
    data = dataset.copy()
    last_time = data["Time"].max()

    folds = []
    fold_num = 1

    if split_setup == 1:
        first_val_start = pd.Timestamp("2024-01-01 00:00:00")
        val_start = first_val_start

        while True:
            train_end = val_start - pd.Timedelta(hours=1)
            train_start = train_end - pd.Timedelta(hours=train_window - 1)

            val_end = val_start + pd.Timedelta(hours=predict_horizon - 1)

            # Stop if fold falls outside dataset
            if train_start < data["Time"].min():
                break
            if val_end > last_time:
                break

            folds.append({
                "fold": fold_num,
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end
            })

            fold_num += 1
            val_start = val_start + pd.Timedelta(hours=stride)

    elif split_setup == 2:
        val_end = last_time
        val_start = val_end - pd.Timedelta(hours=predict_horizon - 1)
        train_end = val_start - pd.Timedelta(hours=1)
        train_start = train_end - pd.Timedelta(hours=train_window - 1)

        folds.append({
                "fold": fold_num,
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end
            })

    print(f"Number of folds: {len(folds)}")
    for fold in folds:
        print(
            f"Fold {fold['fold']}: "
            f"Train from {fold['train_start']} to {fold['train_end']},\n"
            f"        Validation from {fold['val_start']} to {fold['val_end']}"
        )

    return folds