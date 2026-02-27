import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def walk_forward_validation(
        data_series,
        predict_fn,
        training_window=8760,
        forecast_horizon=168,
        expanding=False,
):
    """
    General walk-forward validation.
    
    data_series: pandas DataFrame with target as first column
    predict_fn: callable with signature predict_fn(train_data) -> np.array of length forecast_horizon
    expanding: if True, uses expanding window; if False, uses sliding window
    """
    all_predictions = []
    all_actuals = []
    all_fold_rmse = []
    all_fold_mae = []

    if expanding:
        num_folds = (len(data_series) - training_window) // forecast_horizon
    else:
        num_folds = (len(data_series) - training_window - forecast_horizon) // forecast_horizon

    print(f"Total folds: {num_folds}")
    print(f"Training window: {training_window} hours")
    print(f"Forecast horizon: {forecast_horizon} hours")
    print(f"Mode: {'Expanding' if expanding else 'Sliding'} window\n")

    for fold in range(num_folds):
        if expanding:
            train_start = 0
            train_end = training_window + fold * forecast_horizon
        else:
            train_start = fold * forecast_horizon
            train_end = train_start + training_window

        val_start = train_end
        val_end = val_start + forecast_horizon

        if val_end > len(data_series):
            break

        print(f"Fold {fold + 1}/{num_folds}")

        train_data = data_series.iloc[train_start:train_end]
        val_data = data_series.iloc[val_start:val_end]

        predictions = predict_fn(train_data)
        actuals = val_data.iloc[:, 0].values  # target column only

        assert len(predictions) == forecast_horizon, \
            f"predict_fn must return {forecast_horizon} predictions, got {len(predictions)}"

        fold_rmse = np.sqrt(mean_squared_error(actuals, predictions))
        fold_mae = mean_absolute_error(actuals, predictions)

        all_fold_rmse.append(fold_rmse)
        all_fold_mae.append(fold_mae)
        all_predictions.extend(predictions)
        all_actuals.extend(actuals)

        print(f"  Fold RMSE: {fold_rmse:.2f}, MAE: {fold_mae:.2f}\n")

    overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
    overall_mae = mean_absolute_error(all_actuals, all_predictions)

    print(f"Overall RMSE: {overall_rmse:.2f}")
    print(f"Overall MAE: {overall_mae:.2f}")
    print(f"Average Fold RMSE: {np.mean(all_fold_rmse):.2f} (+/- {np.std(all_fold_rmse):.2f})")
    print(f"Average Fold MAE: {np.mean(all_fold_mae):.2f} (+/- {np.std(all_fold_mae):.2f})")

    return all_predictions, all_actuals, all_fold_rmse, all_fold_mae

def plot_walk_forward_results(
        predictions, 
        actuals, 
        title,
        training_window = 8760, # 365 days of data - hours of data trained on
        forecast_horizon = 168, # 1 week - hours of data predicted
        ):
    """
    Plot walk-forward validation results.
    """
    plt.figure(figsize=(15, 6))
    
    # Create x-axis (fold numbers repeated for each prediction in that fold)
    x = []
    fold_num = 0
    for i in range(len(predictions)):
        if i % (forecast_horizon) == 0 and i > 0:
            fold_num += 1
        x.append(fold_num)
    
    plt.plot(actuals, label='Actual', alpha=0.7, linewidth=2)
    plt.plot(predictions, label='Predicted', alpha=0.7, linewidth=2)
    
    # Add vertical lines to separate folds
    for i in range(0, len(predictions), forecast_horizon):
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3)
    
    plt.xlabel('Time Step')
    plt.ylabel('Electricity Price (DKK)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()