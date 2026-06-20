import sys
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List
from collections import OrderedDict

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0

    values = np.zeros_like(y_true, dtype=float)
    values[mask] = 2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]

    return 100 * np.mean(values)

def find_paths():
    # Find the project root (Speciale_Kode)
    current_dir = os.getcwd()
    project_root = current_dir

    # Looks for "Speciale_Kode" folder:
    while os.path.basename(project_root) != "Speciale_Kode":
        project_root = os.path.dirname(project_root)

    # Add to Python path
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    deep_folder = os.path.join(project_root, "Deep learners")

    final_eval_folder = os.path.join(deep_folder, "Final_eval")

    return final_eval_folder

def xpoints(folder: str, filename: str):
    filepath = os.path.join(folder, filename)

    df = pd.read_csv(filepath, sep = ";")

    column = "Date"

    xpoints = df[column].values.tolist()

    xpoints = list(OrderedDict.fromkeys(xpoints))

    return xpoints

def generate_metrics_graph(file: str, model: str, 
                           price_zone: str, xpoints: List[str],
                           folder: str):
    
    filepath = os.path.join(folder, file)
    df_model = pd.read_csv(filepath, sep = ";")

    df_actuals = df_model["DKPrice"].values.tolist()
    df_preds = df_model["Prediction"].values.tolist()

    smape_list = []
    rmse_list = []
    mae_list = []

    for i in range(0, len(df_preds), 24):
        actuals_segment = df_actuals[i:i+24]
        preds_segment = df_preds[i:i+24]
        smape_val = smape(actuals_segment, preds_segment)
        rmse_val = np.sqrt(mean_squared_error(actuals_segment, preds_segment))
        mae_val = mean_absolute_error(actuals_segment, preds_segment)
        smape_list.append(smape_val)
        rmse_list.append(rmse_val)
        mae_list.append(mae_val)
    
    if model == "LSTM-AE":
        smape_list.pop()
        rmse_list.pop()
        mae_list.pop()

    fig, ax1 = plt.subplots(figsize = (30, 5))

    color_smape = 'tab:red'
    color_yaxis1 = 'firebrick'
    color_rmse = 'navy'
    color_mae = 'tab:green'
    color_yaxis2 = 'tab:blue'

    ax1.set_xlabel('Prediction Day')
    ax1.set_ylabel('SMAPE (%)', color = color_yaxis1)
    line1, = ax1.plot(xpoints, smape_list, label = "SMAPE (%)", color = color_smape)
    ax1.tick_params(axis = 'y', labelcolor = color_yaxis1)

    ax2 = ax1.twinx()

    ax2.set_ylabel('DKK/MWh', color = color_yaxis2)
    line2, = ax2.plot(xpoints, rmse_list, label = "RMSE", color = color_rmse)
    line3, = ax2.plot(xpoints, mae_list, label = "MAE", color = color_mae)
    ax2.tick_params(axis = 'y', labelcolor = color_yaxis2)

    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc = 'upper right')

    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.title(f"Evaluation Metric Development for 2025 for {model} for {price_zone}")
    fig.tight_layout()
    plt.savefig(f"metrics_graph_{model}_{price_zone}")
    plt.show()


def generate_smape_graph(price_zone: str, xpoints: List[str],
                         folder: str):
    filename_rnn = "DK1_RNN_2y_price_lag1excl_2024incl_4val_test_predictions.csv"
    filepath_rnn = os.path.join(folder, filename_rnn)
    df_rnn = pd.read_csv(filepath_rnn, sep = ";")

    filename_gru = "DK1_GRU_2y_price_lag1excl_2024incl_4val_test_predictions.csv"
    filepath_gru = os.path.join(folder, filename_gru)
    df_gru = pd.read_csv(filepath_gru, sep = ";")

    filename_lstm = "DK1_LSTM_2y_MaskExcl_Lag1Incl_2024Incl_4val_hidden64_layers3_bs64_seq24_drop0.2_test_predictions.csv"
    filepath_lstm = os.path.join(folder, filename_lstm)
    df_lstm = pd.read_csv(filepath_lstm, sep = ";")

    filename_lstmae = "DK1_LSTM_AE_2y_2024incl_Lag1_incl_lags_excl_4valFE_lays1_FE_LatDim33_EnHid28_DeHid48_EnDeLays2_test_predictions.csv"
    filepath_lstmae = os.path.join(folder, filename_lstmae)
    df_lstmae = pd.read_csv(filepath_lstmae, sep = ";")
    
    actuals = df_rnn["DKPrice"].values.tolist()

    rnn_preds = df_rnn["Prediction"].values.tolist()
    gru_preds = df_gru["Prediction"].values.tolist()
    lstm_preds = df_lstm["Prediction"].values.tolist()
    lstmae_preds = df_lstmae["Prediction"].values.tolist()

    rnn_smape = []
    gru_smape = []
    lstm_smape = []
    lstmae_smape = []
    
    for i in range(0, len(actuals), 24):
        actuals_segment = actuals[i:i+24]
        rnn_segment = rnn_preds[i:i+24]
        gru_segment = gru_preds[i:i+24]
        lstm_segment = lstm_preds[i:i+24]
        lstmae_segment = lstmae_preds[i:i+24]
        rnn_val = smape(actuals_segment, rnn_segment)
        gru_val = smape(actuals_segment, gru_segment)
        lstm_val = smape(actuals_segment, lstm_segment)
        lstmae_val = smape(actuals_segment, lstmae_segment)
        rnn_smape.append(rnn_val)
        gru_smape.append(gru_val)
        lstm_smape.append(lstm_val)
        lstmae_smape.append(lstmae_val)

    plt.figure(figsize=(30, 5))
    plt.plot(xpoints, rnn_smape, label = "RNN")
    plt.plot(xpoints, gru_smape, label = "GRU")
    plt.plot(xpoints, lstm_smape, label = "LSTM")
    plt.plot(xpoints, lstmae_smape, label = "LSTM-AE")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xlabel('Prediction Day')
    plt.ylabel('SMAPE (%)')
    plt.title(f'Daily SMAPE (%) for 2025 for Deep Learners for {price_zone}')
    plt.legend(loc = 'center right')
    plt.tight_layout()
    plt.savefig(f"SMAPE_graph_deep_{price_zone}")
    plt.show()

def generate_rmse_graph(price_zone: str, xpoints: List[str],
                         folder: str):
    filename_rnn = "DK1_RNN_2y_price_lag1excl_2024incl_4val_test_predictions.csv"
    filepath_rnn = os.path.join(folder, filename_rnn)
    df_rnn = pd.read_csv(filepath_rnn, sep = ";")

    filename_gru = "DK1_GRU_2y_price_lag1excl_2024incl_4val_test_predictions.csv"
    filepath_gru = os.path.join(folder, filename_gru)
    df_gru = pd.read_csv(filepath_gru, sep = ";")

    filename_lstm = "DK1_LSTM_2y_MaskExcl_Lag1Incl_2024Incl_4val_hidden64_layers3_bs64_seq24_drop0.2_test_predictions.csv"
    filepath_lstm = os.path.join(folder, filename_lstm)
    df_lstm = pd.read_csv(filepath_lstm, sep = ";")

    filename_lstmae = "DK1_LSTM_AE_2y_2024incl_Lag1_incl_lags_excl_4valFE_lays1_FE_LatDim33_EnHid28_DeHid48_EnDeLays2_test_predictions.csv"
    filepath_lstmae = os.path.join(folder, filename_lstmae)
    df_lstmae = pd.read_csv(filepath_lstmae, sep = ";")
    
    actuals = df_rnn["DKPrice"].values.tolist()

    rnn_preds = df_rnn["Prediction"].values.tolist()
    gru_preds = df_gru["Prediction"].values.tolist()
    lstm_preds = df_lstm["Prediction"].values.tolist()
    lstmae_preds = df_lstmae["Prediction"].values.tolist()

    rnn_rmse = []
    gru_rmse = []
    lstm_rmse = []
    lstmae_rmse = []
    
    for i in range(0, len(actuals), 24):
        actuals_segment = actuals[i:i+24]
        rnn_segment = rnn_preds[i:i+24]
        gru_segment = gru_preds[i:i+24]
        lstm_segment = lstm_preds[i:i+24]
        lstmae_segment = lstmae_preds[i:i+24]
        rnn_val = np.sqrt(mean_squared_error(actuals_segment, rnn_segment))
        gru_val = np.sqrt(mean_squared_error(actuals_segment, gru_segment))
        lstm_val = np.sqrt(mean_squared_error(actuals_segment, lstm_segment))
        lstmae_val = np.sqrt(mean_squared_error(actuals_segment, lstmae_segment))
        rnn_rmse.append(rnn_val)
        gru_rmse.append(gru_val)
        lstm_rmse.append(lstm_val)
        lstmae_rmse.append(lstmae_val)

    plt.figure(figsize=(30, 5))
    plt.plot(xpoints, rnn_rmse, label = "RNN")
    plt.plot(xpoints, gru_rmse, label = "GRU")
    plt.plot(xpoints, lstm_rmse, label = "LSTM")
    plt.plot(xpoints, lstmae_rmse, label = "LSTM-AE")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xlabel('Prediction Day')
    plt.ylabel('RMSE (DKK/MWh)')
    plt.title(f'RMSE (DKK/MWh) for 2025 for Deep Learners for {price_zone}')
    plt.legend(loc = 'center right')
    plt.tight_layout()
    plt.savefig(f"RMSE_graph_deep_{price_zone}")
    plt.show()

def generate_mae_graph(price_zone: str, xpoints: List[str],
                         folder: str):
    filename_rnn = "DK1_RNN_2y_price_lag1excl_2024incl_4val_test_predictions.csv"
    filepath_rnn = os.path.join(folder, filename_rnn)
    df_rnn = pd.read_csv(filepath_rnn, sep = ";")

    filename_gru = "DK1_GRU_2y_price_lag1excl_2024incl_4val_test_predictions.csv"
    filepath_gru = os.path.join(folder, filename_gru)
    df_gru = pd.read_csv(filepath_gru, sep = ";")

    filename_lstm = "DK1_LSTM_2y_MaskExcl_Lag1Incl_2024Incl_4val_hidden64_layers3_bs64_seq24_drop0.2_test_predictions.csv"
    filepath_lstm = os.path.join(folder, filename_lstm)
    df_lstm = pd.read_csv(filepath_lstm, sep = ";")

    filename_lstmae = "DK1_LSTM_AE_2y_2024incl_Lag1_incl_lags_excl_4valFE_lays1_FE_LatDim33_EnHid28_DeHid48_EnDeLays2_test_predictions.csv"
    filepath_lstmae = os.path.join(folder, filename_lstmae)
    df_lstmae = pd.read_csv(filepath_lstmae, sep = ";")
    
    actuals = df_rnn["DKPrice"].values.tolist()

    rnn_preds = df_rnn["Prediction"].values.tolist()
    gru_preds = df_gru["Prediction"].values.tolist()
    lstm_preds = df_lstm["Prediction"].values.tolist()
    lstmae_preds = df_lstmae["Prediction"].values.tolist()

    rnn_mae = []
    gru_mae = []
    lstm_mae = []
    lstmae_mae = []
    
    for i in range(0, len(actuals), 24):
        actuals_segment = actuals[i:i+24]
        rnn_segment = rnn_preds[i:i+24]
        gru_segment = gru_preds[i:i+24]
        lstm_segment = lstm_preds[i:i+24]
        lstmae_segment = lstmae_preds[i:i+24]
        rnn_val = mean_absolute_error(actuals_segment, rnn_segment)
        gru_val = mean_absolute_error(actuals_segment, gru_segment)
        lstm_val = mean_absolute_error(actuals_segment, lstm_segment)
        lstmae_val = mean_absolute_error(actuals_segment, lstmae_segment)
        rnn_mae.append(rnn_val)
        gru_mae.append(gru_val)
        lstm_mae.append(lstm_val)
        lstmae_mae.append(lstmae_val)

    plt.figure(figsize=(30, 5))
    plt.plot(xpoints, rnn_mae, label = "RNN")
    plt.plot(xpoints, gru_mae, label = "GRU")
    plt.plot(xpoints, lstm_mae, label = "LSTM")
    plt.plot(xpoints, lstmae_mae, label = "LSTM-AE")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xlabel('Prediction Day')
    plt.ylabel('MAE (DKK/MWh)')
    plt.title(f'MAE (DKK/MWh) for 2025 for Deep Learners for {price_zone}')
    plt.legend(loc = 'center right')
    plt.tight_layout()
    plt.savefig(f"MAE_graph_deep_{price_zone}")
    plt.show()

def generate_quarterly_smape_graph(price_zone: str, xpoints: List[str],
                                   folder: str):
    filename_rnn = "DK1_RNN_2y_price_lag1excl_2024incl_4val_test_predictions.csv"
    filepath_rnn = os.path.join(folder, filename_rnn)
    df_rnn = pd.read_csv(filepath_rnn, sep = ";")

    filename_gru = "DK1_GRU_2y_price_lag1excl_2024incl_4val_test_predictions.csv"
    filepath_gru = os.path.join(folder, filename_gru)
    df_gru = pd.read_csv(filepath_gru, sep = ";")

    filename_lstm = "DK1_LSTM_2y_MaskExcl_Lag1Incl_2024Incl_4val_hidden64_layers3_bs64_seq24_drop0.2_test_predictions.csv"
    filepath_lstm = os.path.join(folder, filename_lstm)
    df_lstm = pd.read_csv(filepath_lstm, sep = ";")

    filename_lstmae = "DK1_LSTM_AE_2y_2024incl_Lag1_incl_lags_excl_4valFE_lays1_FE_LatDim33_EnHid28_DeHid48_EnDeLays2_test_predictions.csv"
    filepath_lstmae = os.path.join(folder, filename_lstmae)
    df_lstmae = pd.read_csv(filepath_lstmae, sep = ";")
    
    actuals = df_rnn["DKPrice"].values.tolist()

    rnn_preds = df_rnn["Prediction"].values.tolist()
    gru_preds = df_gru["Prediction"].values.tolist()
    lstm_preds = df_lstm["Prediction"].values.tolist()
    lstmae_preds = df_lstmae["Prediction"].values.tolist()

    rnn_smape = []
    gru_smape = []
    lstm_smape = []
    lstmae_smape = []

    for i in range(0, len(actuals), 24):
        actuals_segment = actuals[i:i+24]
        rnn_smape.append(smape(actuals_segment, rnn_preds[i:i+24]))
        gru_smape.append(smape(actuals_segment, gru_preds[i:i+24]))
        lstm_smape.append(smape(actuals_segment, lstm_preds[i:i+24]))
        lstmae_smape.append(smape(actuals_segment, lstmae_preds[i:i+24]))

    xpoints = pd.to_datetime(xpoints)

    # Define quarter masks
    quarters = [
        "Q1 (Jan-Mar) 2025", "Q2 (Apr-Jun) 2025", "Q3 (Jul-Sep) 2025", "Q4 (Oct-Dec) 2025"
    ]
    masks = [
        (xpoints.month >= 1) & (xpoints.month <= 3),
        (xpoints.month >= 4) & (xpoints.month <= 6),
        (xpoints.month >= 7) & (xpoints.month <= 9),
        (xpoints.month >= 10) & (xpoints.month <= 12),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(30, 20))
    axes = axes.flatten()

    for idx, (quarter, mask) in enumerate(zip(quarters, masks)):
        ax = axes[idx]
        xq = xpoints[mask]
        ax.plot(xq, [v for v, m in zip(rnn_smape, mask) if m], label="RNN")
        ax.plot(xq, [v for v, m in zip(gru_smape, mask) if m], label="GRU")
        ax.plot(xq, [v for v, m in zip(lstm_smape, mask) if m], label="LSTM")
        ax.plot(xq, [v for v, m in zip(lstmae_smape, mask) if m], label="LSTM-AE")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_xlabel('Prediction Day')
        ax.set_ylabel('SMAPE (%)')
        ax.set_title(quarter)
        ax.legend(loc='upper center')

    fig.suptitle(f'Quarterly Daily SMAPE (%) for 2025 for Deep Learners for {price_zone}', fontsize=14)
    plt.tight_layout(pad = 2.0)
    plt.savefig(f"SMAPE_graph_deep_quarterly_{price_zone}")
    plt.show()

def make_deep_7_day_metrics(folder, file, price_zone):
    filepath = os.path.join(folder, file)

    df = pd.read_csv(filepath, sep = ";")

    actuals = df["DKPrice"].values.tolist()
    preds = df["Prediction"].values.tolist()

    smape_list = []
    rmse_list = []
    mae_list = []

    for i in range(0, len(actuals), 24):
        actuals_segment = actuals[i:i+24]
        preds_segment = preds[i:i+24]
        smape_val = smape(actuals_segment, preds_segment)
        rmse_val = np.sqrt(mean_squared_error(actuals_segment, preds_segment))
        mae_val = mean_absolute_error(actuals_segment, preds_segment)
        smape_list.append(smape_val)
        rmse_list.append(rmse_val)
        mae_list.append(mae_val)

    smape_averages = {}
    rmse_averages = {}
    mae_averages = {}
    for i in range(7):
        smape_averages[f'avg_smape_day{i+1}'] = sum(smape_list[i::7]) / len(smape_list[i::7])
        rmse_averages[f'avg_rmse_day_{i+1}'] = sum(rmse_list[i::7]) / len(rmse_list[i::7])
        mae_averages[f'avg_mae_day_{i+1}'] = sum(mae_list[i::7]) / len(mae_list[i::7])

    smape_final = list(smape_averages.values())
    rmse_final = list(rmse_averages.values())
    mae_final = list(mae_averages.values())

    return smape_final, rmse_final, mae_final

def deep_7_day_smape(rnn_list, gru_list, lstm_list,
                       lstmae_list, price_zone):
    xpoints = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]

    plt.plot(xpoints, rnn_list, label = "RNN")
    plt.plot(xpoints, gru_list, label = "GRU")
    plt.plot(xpoints, lstm_list, label = "LSTM")
    plt.plot(xpoints, lstmae_list, label = "LSTM-AE")
    plt.xlabel('Prediction Day')
    plt.ylabel('SMAPE (%)')
    plt.title(f'7-Day SMAPE (%) Development for Deep Learners with Noise for {price_zone}')
    plt.legend()
    plt.savefig(f"deep_7day_smape_{price_zone}")
    plt.show()

def deep_7_day_rmse(rnn_list, gru_list, lstm_list,
                       lstmae_list, price_zone):
    xpoints = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]

    plt.plot(xpoints, rnn_list, label = "RNN")
    plt.plot(xpoints, gru_list, label = "GRU")
    plt.plot(xpoints, lstm_list, label = "LSTM")
    plt.plot(xpoints, lstmae_list, label = "LSTM-AE")
    plt.xlabel('Prediction Day')
    plt.ylabel('RMSE (DKK/MWh)')
    plt.title(f'7-Day RMSE (DKK/MWh) Development for Deep Learners with Noise for {price_zone}')
    plt.legend()
    plt.savefig(f"deep_7day_rmse_{price_zone}")
    plt.show()

def deep_7_day_mae(rnn_list, gru_list, lstm_list,
                       lstmae_list, price_zone):
    xpoints = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]

    plt.plot(xpoints, rnn_list, label = "RNN")
    plt.plot(xpoints, gru_list, label = "GRU")
    plt.plot(xpoints, lstm_list, label = "LSTM")
    plt.plot(xpoints, lstmae_list, label = "LSTM-AE")
    plt.xlabel('Prediction Day')
    plt.ylabel('MAE (DKK/MWh)')
    plt.title(f'7-Day MAE (DKK/MWh) Development for Deep Learners with Noise for {price_zone}')
    plt.legend()
    plt.savefig(f"deep_7day_mae_{price_zone}")
    plt.show()