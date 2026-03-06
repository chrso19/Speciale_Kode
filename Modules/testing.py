import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from validation import smape
import torch

def predict_with_deep_model(model, known_data, weather_forecast, predict_horizon, 
                             train_mean, train_std, sequence_length=24,
                             reshape_fn=None):  # optional reshape function
    """
    Autoregressive prediction using a pre-trained model.
    Seed from known_data, update weather features from forecast.
    
    reshape_fn: optional callable to transform input_seq before passing to model
                defaults to None which works for LSTM and Transformer
    """
    device = next(model.parameters()).device

    known_scaled = (known_data.values - train_mean.values) / train_std.values
    weather_scaled = (weather_forecast.values - train_mean.values) / train_std.values

    input_seq = torch.FloatTensor(known_scaled[-sequence_length:]).unsqueeze(0).to(device)

    model.eval()
    predictions = []

    with torch.no_grad():
        for step in range(predict_horizon):

            # Apply reshape if needed for different architectures
            model_input = reshape_fn(input_seq) if reshape_fn else input_seq

            output = model(model_input)
            next_val = output.squeeze().item()
            predictions.append(next_val)

            next_step = input_seq[:, -1, :].clone()
            next_step[:, 0] = next_val

            weather_step = len(known_data) + step
            next_step[:, 1:] = torch.FloatTensor(
                weather_scaled[weather_step, 1:]
            ).to(device)

            input_seq = torch.cat([input_seq[:, 1:, :], next_step.unsqueeze(1)], dim=1)

    predictions = np.array(predictions) * train_std.iloc[0] + train_mean.iloc[0]
    return predictions

def predict_with_shallow_model(model, known_data, weather_forecast, predict_horizon, train_mean=None, train_std=None, needs_scaling=False):
    """
    Autoregressive prediction using a pre-trained shallow model (XGBoost, SVM etc.)
    """
    # Start from last known row
    last_row = known_data.iloc[-1:].copy()
    last_row = last_row.iloc[:, 1:]  # drop target column, keep features

    predictions = []

    for step in range(predict_horizon):

        # Scale if needed (SVM yes, XGBoost no)
        if needs_scaling and train_mean is not None:
            input_row = (last_row.values - train_mean.values[1:]) / train_std.values[1:]
        else:
            input_row = last_row.values

        next_val = model.predict(input_row)[0]
        predictions.append(next_val)

        # Update lagged price features
        if 'Price_lag1' in last_row.columns:
            last_row['Price_lag1'] = next_val
        if 'Price_lag24' in last_row.columns:
            last_row['Price_lag24'] = next_val

        # Update weather features from forecast
        weather_step = len(known_data) + step
        for col in last_row.columns:
            if col in weather_forecast.columns and 'Price' not in col:
                last_row[col] = weather_forecast.iloc[weather_step][col]

    return np.array(predictions)

def final_model_evaluation(
        trained_model,
        train_data,
        test_data,
        weather_forecast_series,
        inference_fn,  # pass predict_with_model or predict_with_shallow_model
        known_hours=24,
        forecast_horizon=192,
        known_prices_series=None,
):
    """
    Final evaluation of a pre-trained model on the test set.

    trained_model: already fitted model object
    train_data: full training DataFrame used to fit the model
    test_data: test DataFrame (8760 rows) with target as first column
    weather_forecast_series: forecasted weather data covering the test period
    known_hours: number of hours of actual prices known in advance (24)
    forecast_horizon: total hours per evaluation window (192)
    known_prices_series: optional separate series for known prices
    """

    all_predictions = []
    all_actuals = []
    all_fold_rmse = []
    all_fold_mae = []
    all_fold_smape = []
    all_daily_rmse = []
    all_daily_mae = []
    all_daily_smape = []

    predict_horizon = forecast_horizon - known_hours  # 168 hours
    n_days = predict_horizon // 24                    # 7 days
    num_folds = len(test_data) // forecast_horizon    # 45 windows

    print(f"Training data size: {len(train_data)} hours")
    print(f"Test data size: {len(test_data)} hours")
    print(f"Forecast horizon: {forecast_horizon} hours")
    print(f"Known hours: {known_hours} hours")
    print(f"Predict horizon: hour {known_hours + 1} to hour {forecast_horizon}")
    print(f"Number of evaluation windows: {num_folds}\n")

    # Compute training statistics once for scaling
    train_mean = train_data.mean()
    train_std = train_data.std()

    for fold in range(num_folds):
        fold_start = fold * forecast_horizon
        fold_end = fold_start + forecast_horizon

        if fold_end > len(test_data):
            break

        print(f"Evaluation window {fold + 1}/{num_folds}")

        # Known prices - first 24 hours of this window
        if known_prices_series is not None:
            known_data = known_prices_series.iloc[fold_start:fold_start + known_hours]
        else:
            known_data = test_data.iloc[fold_start:fold_start + known_hours]

        # Target - hours 25 to 192
        val_data = test_data.iloc[fold_start + known_hours:fold_end]

        # Weather forecast for full 192 hours
        weather_forecast = weather_forecast_series.iloc[fold_start:fold_end]

        # Generate predictions using the pre-trained model
        predictions = inference_fn(
            model=trained_model,
            known_data=known_data,
            weather_forecast=weather_forecast,
            predict_horizon=predict_horizon,
            train_mean=train_mean,
            train_std=train_std,
        )
        actuals = val_data.iloc[:, 0].values

        assert len(predictions) == predict_horizon, \
            f"Expected {predict_horizon} predictions, got {len(predictions)}"

        # Fold level metrics
        fold_rmse = np.sqrt(mean_squared_error(actuals, predictions))
        fold_mae = mean_absolute_error(actuals, predictions)
        fold_smape = smape(actuals, predictions)

        all_fold_rmse.append(fold_rmse)
        all_fold_mae.append(fold_mae)
        all_fold_smape.append(fold_smape)
        all_predictions.extend(predictions)
        all_actuals.extend(actuals)

        # Daily level metrics
        fold_daily_rmse = []
        fold_daily_mae = []
        fold_daily_smape = []

        for day in range(n_days):
            day_start = day * 24
            day_end = day_start + 24

            day_actuals = actuals[day_start:day_end]
            day_predictions = predictions[day_start:day_end]

            fold_daily_rmse.append(np.sqrt(mean_squared_error(day_actuals, day_predictions)))
            fold_daily_mae.append(mean_absolute_error(day_actuals, day_predictions))
            fold_daily_smape.append(smape(day_actuals, day_predictions))

        all_daily_rmse.append(fold_daily_rmse)
        all_daily_mae.append(fold_daily_mae)
        all_daily_smape.append(fold_daily_smape)

        print(f"  Fold RMSE: {fold_rmse:.2f}, MAE: {fold_mae:.2f}, SMAPE: {fold_smape:.2f}%")
        print(f"  Daily RMSE:  {[f'{x:.2f}' for x in fold_daily_rmse]}")
        print(f"  Daily MAE:   {[f'{x:.2f}' for x in fold_daily_mae]}")
        print(f"  Daily SMAPE: {[f'{x:.2f}%' for x in fold_daily_smape]}\n")

    # Overall metrics
    overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
    overall_mae = mean_absolute_error(all_actuals, all_predictions)
    overall_smape = smape(all_actuals, all_predictions)

    avg_daily_rmse = np.mean(all_daily_rmse, axis=0)
    avg_daily_mae = np.mean(all_daily_mae, axis=0)
    avg_daily_smape = np.mean(all_daily_smape, axis=0)

    print(f"Overall RMSE: {overall_rmse:.2f}")
    print(f"Overall MAE: {overall_mae:.2f}")
    print(f"Overall SMAPE: {overall_smape:.2f}%")
    print(f"Average Fold RMSE: {np.mean(all_fold_rmse):.2f} (+/- {np.std(all_fold_rmse):.2f})")
    print(f"Average Fold MAE: {np.mean(all_fold_mae):.2f} (+/- {np.std(all_fold_mae):.2f})")
    print(f"Average Fold SMAPE: {np.mean(all_fold_smape):.2f}% (+/- {np.std(all_fold_smape):.2f}%)")
    print(f"\nAverage RMSE by day:  {[f'{x:.2f}' for x in avg_daily_rmse]}")
    print(f"Average MAE by day:   {[f'{x:.2f}' for x in avg_daily_mae]}")
    print(f"Average SMAPE by day: {[f'{x:.2f}%' for x in avg_daily_smape]}")

    return (all_predictions, all_actuals,
            all_fold_rmse, all_fold_mae, all_fold_smape,
            all_daily_rmse, all_daily_mae, all_daily_smape)

