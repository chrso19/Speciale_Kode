import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def smape(actuals, predictions):
    # Symmetric MAPE
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    denominator = (np.abs(actuals) + np.abs(predictions)) / 2
    mask = denominator > 0
    return np.mean(np.abs(actuals[mask] - predictions[mask]) / denominator[mask]) * 100

def walk_forward_validation(
        data_series,
        weather_forecast_series,
        predict_fn,
        known_prices_series=None,
        training_window=17520,
        forecast_horizon=192,
        known_hours=24, # number of hours that future electricity prices are known for
        stride = 4, # number of weeks
        expanding=False,
):
    """
    General walk-forward validation.
    
    data_series: pandas DataFrame with target as first column
    weather_forecast_series: forecasted weather data for the forecast horizon
    predict_fn: callable with signature predict_fn(train_data) -> np.array of length forecast_horizon
    known_prices_series: already forecasted electricity prices
    training_window: number of hours each fold is trained on
    forecast_horizon: number of hours that the price should be forecasted
    known hours: number of hours in the future with an already forecasted price
    stride: number of weeks that should be skipped between each fold
    expanding: if True, uses expanding window; if False, uses sliding window
    """
    all_predictions = []
    all_actuals = []
    all_fold_rmse = []
    all_fold_mae = []
    all_fold_smape = []

    all_daily_rmse = []
    all_daily_mae = []
    all_daily_smape = []

    predict_horizon = forecast_horizon - known_hours

    if expanding:
        num_folds = (len(data_series) - training_window) // (stride * predict_horizon)
    else:
        num_folds = (len(data_series) - training_window - forecast_horizon) // (stride * predict_horizon)

    print(f"Total folds: {num_folds}")
    print(f"Training window: {training_window} hours")
    print(f"Forecast horizon: {forecast_horizon} hours")
    print(f"Electricity price is known for: {known_hours} hours")
    print(f"Predict horizon is from hour {known_hours + 1} to hour { forecast_horizon}")
    print(f"Stride: {stride} weeks or {stride * predict_horizon} hours")
    print(f"Mode: {'Expanding' if expanding else 'Sliding'} window\n")

    for fold in range(num_folds + 1):
        if expanding:
            train_start = 0
            train_end = training_window + fold * predict_horizon * stride
        else:
            train_start = fold * predict_horizon* stride
            train_end = train_start + training_window

        val_start = train_end
        val_end = val_start + forecast_horizon

        if val_end > len(data_series):
            print(f"Fold {fold + 1}/{num_folds + 1}: Skipped (not enough data)")
            break

        print(f"Fold {fold + 1}/{num_folds + 1}")

        train_data = data_series.iloc[train_start:train_end]
        
        if known_prices_series is not None:
            known_data = known_prices_series.iloc[val_start:val_start + known_hours]
        else:
            known_data = data_series.iloc[val_start:val_start + known_hours]

        val_data = data_series.iloc[val_start + known_hours: val_end]

        weather_forecast = weather_forecast_series.iloc[val_start:val_start + forecast_horizon]

        predictions = predict_fn(
            train_data=train_data,
            known_data = known_data,
            weather_forecast = weather_forecast,
            predict_horizon = predict_horizon
            )
        
        actuals = val_data.iloc[:, 0].values  # target column only

        assert len(predictions) == predict_horizon, \
            f"predict_fn must return {predict_horizon} predictions, got {len(predictions)}"

        fold_rmse = np.sqrt(mean_squared_error(actuals, predictions))
        fold_mae = mean_absolute_error(actuals, predictions)
        fold_mape = smape(actuals, predictions)

        all_fold_rmse.append(fold_rmse)
        all_fold_mae.append(fold_mae)
        all_fold_smape.append(fold_mape)
        all_predictions.extend(predictions)
        all_actuals.extend(actuals)

        n_days = predict_horizon // 24  # 7
        fold_daily_rmse = []
        fold_daily_mae = []
        fold_daily_smape = []

        for day in range(n_days):
            day_start = day * 24
            day_end = day_start + 24

            day_actuals = actuals[day_start:day_end]
            day_predictions = predictions[day_start:day_end]

            day_rmse = np.sqrt(mean_squared_error(day_actuals, day_predictions))
            day_mae = mean_absolute_error(day_actuals, day_predictions)
            day_smape = smape(day_actuals, day_predictions)

            fold_daily_rmse.append(day_rmse)
            fold_daily_mae.append(day_mae)
            fold_daily_smape.append(day_smape)

        all_daily_rmse.append(fold_daily_rmse)
        all_daily_mae.append(fold_daily_mae)
        all_daily_smape.append(fold_daily_smape)

        print(f"  Fold RMSE: {fold_rmse:.2f}, MAE: {fold_mae:.2f}, SMAPE: {fold_mape:.2f}\n")
        print(f"  Daily SMAPE: {[f'{x:.2f}' for x in fold_daily_smape]}\n")

    overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
    overall_mae = mean_absolute_error(all_actuals, all_predictions)
    overall_smape = smape(all_actuals, all_predictions)

    print(f"Overall RMSE: {overall_rmse:.2f}")
    print(f"Overall MAE: {overall_mae:.2f}")
    print(f"Overall MAPE: {overall_smape:.2f}")
    print(f"Average Fold RMSE: {np.mean(all_fold_rmse):.2f} (+/- {np.std(all_fold_rmse):.2f})")
    print(f"Average Fold MAE: {np.mean(all_fold_mae):.2f} (+/- {np.std(all_fold_mae):.2f})")
    print(f"Average Fold SMAPE: {np.mean(all_fold_smape):.2f} (+/- {np.std(all_fold_smape):.2f})")

    # Average daily errors across all folds - shape (7,)
    avg_daily_rmse = np.mean(all_daily_rmse, axis=0)
    avg_daily_mae = np.mean(all_daily_mae, axis=0)
    avg_daily_smape = np.mean(all_daily_smape, axis=0)

    print(f"\nAverage RMSE by day: {[f'{x:.2f}' for x in avg_daily_rmse]}")
    print(f"Average MAE by day:  {[f'{x:.2f}' for x in avg_daily_mae]}")
    print(f"Average SMAPE by day: {[f'{x:.2f}%' for x in avg_daily_smape]}")

    return (all_predictions, all_actuals, 
            all_fold_rmse, all_fold_mae, all_fold_smape,
            all_daily_rmse, all_daily_mae, all_daily_smape)

def plot_walk_forward_results(
        predictions, 
        actuals, 
        title,
        training_window = 17520, # 365 days of data - hours of data trained on
        predict_horizon = 192, # 1 week - hours of data predicted
        ):
    """
    Plot walk-forward validation results.
    """
    plt.figure(figsize=(15, 6))
    
    plt.plot(actuals, label='Actual', alpha=0.7, linewidth=2)
    plt.plot(predictions, label='Predicted', alpha=0.7, linewidth=2)
    
    # Vertical lines to separate folds
    for i in range(0, len(predictions), predict_horizon):
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3)
    
    plt.xlabel('Time Step')
    plt.ylabel('Electricity Price (DKK)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_walk_forward_results2(
    predictions, 
    actuals, 
    title,
    data_series,
    training_window=17520,
    predict_horizon=168,
    stride=4,             # number of weeks between folds
    forecast_horizon=192, # total hours per fold including known hours
    ):

    plt.figure(figsize=(15, 6))

    # Only plot up to training window for gray background
    plt.plot(data_series.iloc[:training_window, 0].values,  # stop at training window
             label='Training Data', 
             alpha=0.5, 
             linewidth=1, 
             color='gray')

    # Offset predictions and actuals to start after training window
    x_axis = range(training_window, training_window + len(predictions))

    plt.plot(x_axis, actuals, label='Actual', alpha=0.7, linewidth=2, color='blue')
    plt.plot(x_axis, predictions, label='Predicted', alpha=0.7, linewidth=2, color='orange')

    # Vertical lines every stride * forecast_horizon steps to separate folds
    fold_step = stride * forecast_horizon  # 4 * 192 = 768 hours between folds
    for i in range(training_window, training_window + len(predictions), fold_step):
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3)

    # Vertical line to separate training from validation
    plt.axvline(x=training_window, color='black', linestyle='-', alpha=0.5, label='Train/Val split')

    plt.xlabel('Time Step')
    plt.ylabel('Electricity Price (DKK)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()