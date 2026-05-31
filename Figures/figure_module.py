import sys
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List

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
    
    baseline_folder = os.path.join(project_root, "Baseline")

    naive_folder = os.path.join(baseline_folder, "Naive")
    MA_folder = os.path.join(baseline_folder, "MovingAverage")
    seasonal_folder = os.path.join(baseline_folder, "Seasonal")

    shallow_folder = os.path.join(project_root, "Shallow learners")

    final_eval_folder = os.path.join(shallow_folder, "Final_eval")

    final_eval_folder = os.path.join(final_eval_folder, "Shap")

    return naive_folder, MA_folder, seasonal_folder, final_eval_folder

def xpoints(folder: str, filename: str):
    filename_naive_DK1 = filename
    filepath_naive_DK1 = os.path.join(folder, filename_naive_DK1)

    df_naive_DK1 = pd.read_csv(filepath_naive_DK1)

    column = "Date"

    xpoints = df_naive_DK1[column].values.tolist()

    return xpoints

def generate_metrics_graph_baseline(file: str, model: str, 
                                    price_zone: str, xpoints: List[str],
                                    folder: str):
    
    if model == "Naive Persistence":
        filepath = os.path.join(folder, file)
    elif model == "Moving Average":
        filepath = os.path.join(folder, file)
    elif model == "Seasonal Naive":
        filepath = os.path.join(folder, file)
    else:
        print("The correct model type was not given.")
        print("Please try again.")
        sys.exit()
    df_baseline = pd.read_csv(filepath)

    smape_column = "daily_smape"
    rmse_column = "daily_rmse"
    mae_column = "daily_mae"

    smape_list = df_baseline[smape_column].values.tolist()
    rmse_list = df_baseline[rmse_column].values.tolist()
    mae_list = df_baseline[mae_column].values.tolist()

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

def generate_metrics_graph(file: str, model: str, 
                           price_zone: str, xpoints: List[str],
                           folder: str):
    
    filepath = os.path.join(folder, file)
    df_model = pd.read_csv(filepath, decimal = ",")

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

def generate_smape_graph_baseline(price_zone: str, xpoints: List[str],
                                  naive_folder: str, MA_folder: str,
                                  seasonal_folder: str):
    if price_zone == "DK1":
        filename_naive_DK1 = "DK1_daily_results_naive.csv"
        filepath_naive_DK1 = os.path.join(naive_folder, filename_naive_DK1)
        df_naive = pd.read_csv(filepath_naive_DK1)

        filename_MA_DK1 = "DK1_daily_results_MA.csv"
        filepath_MA_DK1 = os.path.join(MA_folder, filename_MA_DK1)
        df_MA = pd.read_csv(filepath_MA_DK1)

        filename_seasonal_DK1 = "DK1_daily_results_seasonal.csv"
        filepath_seasonal_DK1 = os.path.join(seasonal_folder, filename_seasonal_DK1)
        df_seasonal = pd.read_csv(filepath_seasonal_DK1)
    elif price_zone == "DK2":
        filename_naive_DK2 = "DK2_daily_results_naive.csv"
        filepath_naive_DK2 = os.path.join(naive_folder, filename_naive_DK2)
        df_naive = pd.read_csv(filepath_naive_DK2)

        filename_MA_DK2 = "DK2_daily_results_MA.csv"
        filepath_MA_DK2 = os.path.join(MA_folder, filename_MA_DK2)
        df_MA = pd.read_csv(filepath_MA_DK2)

        filename_seasonal_DK2 = "DK2_daily_results_seasonal.csv"
        filepath_seasonal_DK2 = os.path.join(seasonal_folder, filename_seasonal_DK2)
        df_seasonal = pd.read_csv(filepath_seasonal_DK2)
    else:
        print("The correct price zone was not given.")
        print("Please try again.")
        sys.exit()
    
    column = "daily_smape"
    
    naive_list = df_naive[column].values.tolist()
    MA_list = df_MA[column].values.tolist()
    seasonal_list = df_seasonal[column].values.tolist()

    plt.figure(figsize=(30, 5))
    plt.plot(xpoints, naive_list, label = "Naive Persistence")
    plt.plot(xpoints, MA_list, label = "Moving Average")
    plt.plot(xpoints, seasonal_list, label = "Seasonal Naive")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xlabel('Prediction Day')
    plt.ylabel('SMAPE (%)')
    plt.title(f'Daily SMAPE (%) for 2025 for Baseline Models for {price_zone}')
    plt.legend(loc = 'center right')
    plt.tight_layout()
    plt.savefig(f"SMAPE_graph_baseline_{price_zone}")
    plt.show()

def generate_rmse_graph_baseline(price_zone: str, xpoints: List[str],
                                  naive_folder: str, MA_folder: str,
                                  seasonal_folder: str):
    if price_zone == "DK1":
        filename_naive_DK1 = "DK1_daily_results_naive.csv"
        filepath_naive_DK1 = os.path.join(naive_folder, filename_naive_DK1)
        df_naive = pd.read_csv(filepath_naive_DK1)

        filename_MA_DK1 = "DK1_daily_results_MA.csv"
        filepath_MA_DK1 = os.path.join(MA_folder, filename_MA_DK1)
        df_MA = pd.read_csv(filepath_MA_DK1)

        filename_seasonal_DK1 = "DK1_daily_results_seasonal.csv"
        filepath_seasonal_DK1 = os.path.join(seasonal_folder, filename_seasonal_DK1)
        df_seasonal = pd.read_csv(filepath_seasonal_DK1)
    elif price_zone == "DK2":
        filename_naive_DK2 = "DK2_daily_results_naive.csv"
        filepath_naive_DK2 = os.path.join(naive_folder, filename_naive_DK2)
        df_naive = pd.read_csv(filepath_naive_DK2)

        filename_MA_DK2 = "DK2_daily_results_MA.csv"
        filepath_MA_DK2 = os.path.join(MA_folder, filename_MA_DK2)
        df_MA = pd.read_csv(filepath_MA_DK2)

        filename_seasonal_DK2 = "DK2_daily_results_seasonal.csv"
        filepath_seasonal_DK2 = os.path.join(seasonal_folder, filename_seasonal_DK2)
        df_seasonal = pd.read_csv(filepath_seasonal_DK2)
    else:
        print("The correct price zone was not given.")
        print("Please try again.")
        sys.exit()
    
    column = "daily_rmse"
    
    naive_list = df_naive[column].values.tolist()
    MA_list = df_MA[column].values.tolist()
    seasonal_list = df_seasonal[column].values.tolist()

    plt.figure(figsize=(30, 5))
    plt.plot(xpoints, naive_list, label = "Naive Persistence")
    plt.plot(xpoints, MA_list, label = "Moving Average")
    plt.plot(xpoints, seasonal_list, label = "Seasonal Naive")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xlabel('Prediction Day')
    plt.ylabel('RMSE (DKK/MWh)')
    plt.title(f'Daily RMSE (DKK/MWh) for 2025 for Baseline Models for {price_zone}')
    plt.legend(loc = 'center right')
    plt.tight_layout()
    plt.savefig(f"RMSE_graph_baseline_{price_zone}")
    plt.show()

def generate_mae_graph_baseline(price_zone: str, xpoints: List[str],
                                  naive_folder: str, MA_folder: str,
                                  seasonal_folder: str):
    if price_zone == "DK1":
        filename_naive_DK1 = "DK1_daily_results_naive.csv"
        filepath_naive_DK1 = os.path.join(naive_folder, filename_naive_DK1)
        df_naive = pd.read_csv(filepath_naive_DK1)

        filename_MA_DK1 = "DK1_daily_results_MA.csv"
        filepath_MA_DK1 = os.path.join(MA_folder, filename_MA_DK1)
        df_MA = pd.read_csv(filepath_MA_DK1)

        filename_seasonal_DK1 = "DK1_daily_results_seasonal.csv"
        filepath_seasonal_DK1 = os.path.join(seasonal_folder, filename_seasonal_DK1)
        df_seasonal = pd.read_csv(filepath_seasonal_DK1)
    elif price_zone == "DK2":
        filename_naive_DK2 = "DK2_daily_results_naive.csv"
        filepath_naive_DK2 = os.path.join(naive_folder, filename_naive_DK2)
        df_naive = pd.read_csv(filepath_naive_DK2)

        filename_MA_DK2 = "DK2_daily_results_MA.csv"
        filepath_MA_DK2 = os.path.join(MA_folder, filename_MA_DK2)
        df_MA = pd.read_csv(filepath_MA_DK2)

        filename_seasonal_DK2 = "DK2_daily_results_seasonal.csv"
        filepath_seasonal_DK2 = os.path.join(seasonal_folder, filename_seasonal_DK2)
        df_seasonal = pd.read_csv(filepath_seasonal_DK2)
    else:
        print("The correct price zone was not given.")
        print("Please try again.")
        sys.exit()
    
    column = "daily_mae"
    
    naive_list = df_naive[column].values.tolist()
    MA_list = df_MA[column].values.tolist()
    seasonal_list = df_seasonal[column].values.tolist()

    plt.figure(figsize=(30, 5))
    plt.plot(xpoints, naive_list, label = "Naive Persistence")
    plt.plot(xpoints, MA_list, label = "Moving Average")
    plt.plot(xpoints, seasonal_list, label = "Seasonal Naive")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xlabel('Prediction Day')
    plt.ylabel('MAE (DKK/MWh)')
    plt.title(f'Daily MAE (DKK/MWh) for 2025 for Baseline Models for {price_zone}')
    plt.legend(loc = 'center right')
    plt.tight_layout()
    plt.savefig(f"MAE_graph_baseline_{price_zone}")
    plt.show()

def generate_smape_graph(price_zone: str, xpoints: List[str],
                         folder: str):
    if price_zone == "DK1":
        filename_lasso_DK1 = "DK1_Lasso_predictions2.csv"
        filepath_lasso_DK1 = os.path.join(folder, filename_lasso_DK1)
        df_lasso = pd.read_csv(filepath_lasso_DK1, decimal = ",")

        filename_SVR_DK1 = "DK1_SVR_predictions.csv"
        filepath_SVR_DK1 = os.path.join(folder, filename_SVR_DK1)
        df_SVR = pd.read_csv(filepath_SVR_DK1, decimal = ",")

        filename_XGB_DK1 = "DK1_XGBoost_predictions2.csv"
        filepath_XGB_DK1 = os.path.join(folder, filename_XGB_DK1)
        df_XGB = pd.read_csv(filepath_XGB_DK1, decimal = ",")

        filename_lightgbm_DK1 = "DK1_LightGBM_predictions2.csv"
        filepath_lightgbm_DK1 = os.path.join(folder, filename_lightgbm_DK1)
        df_lightgbm = pd.read_csv(filepath_lightgbm_DK1, decimal = ",")

        filename_RF_DK1 = "DK1_RF_predictions2.csv"
        filepath_RF_DK1 = os.path.join(folder, filename_RF_DK1)
        df_RF = pd.read_csv(filepath_RF_DK1, decimal = ",")
    elif price_zone == "DK2":
        filename_lasso_DK2 = "DK2_Lasso_predictions2.csv"
        filepath_lasso_DK2 = os.path.join(folder, filename_lasso_DK2)
        df_lasso = pd.read_csv(filepath_lasso_DK2, decimal = ",")

        filename_SVR_DK2 = "DK2_SVR_predictions.csv"
        filepath_SVR_DK2 = os.path.join(folder, filename_SVR_DK2)
        df_SVR = pd.read_csv(filepath_SVR_DK2, decimal = ",")

        filename_XGB_DK2 = "DK2_XGBoost_predictions2.csv"
        filepath_XGB_DK2 = os.path.join(folder, filename_XGB_DK2)
        df_XGB = pd.read_csv(filepath_XGB_DK2, decimal = ",")

        filename_lightgbm_DK2 = "DK2_LightGBM_predictions2.csv"
        filepath_lightgbm_DK2 = os.path.join(folder, filename_lightgbm_DK2)
        df_lightgbm = pd.read_csv(filepath_lightgbm_DK2, decimal = ",")

        filename_RF_DK2 = "DK2_RF_predictions2.csv"
        filepath_RF_DK2 = os.path.join(folder, filename_RF_DK2)
        df_RF = pd.read_csv(filepath_RF_DK2, decimal = ",")
    else:
        print("The correct price zone was not given.")
        print("Please try again.")
        sys.exit()
    
    actuals = df_lasso["DKPrice"].values.tolist()

    lasso_preds = df_lasso["Prediction"].values.tolist()
    svr_preds = df_SVR["Prediction"].values.tolist()
    xgb_preds = df_XGB["Prediction"].values.tolist()
    lightgbm_preds = df_lightgbm["Prediction"].values.tolist()
    rf_preds = df_RF["Prediction"].values.tolist()

    lasso_smape = []
    svr_smape = []
    xgb_smape = []
    lightgbm_smape = []
    rf_smape = []
    
    for i in range(0, len(actuals), 24):
        actuals_segment = actuals[i:i+24]
        lasso_segment = lasso_preds[i:i+24]
        svr_segment = svr_preds[i:i+24]
        xgb_segment = xgb_preds[i:i+24]
        lightgbm_segment = lightgbm_preds[i:i+24]
        rf_segment = rf_preds[i:i+24]
        lasso_val = smape(actuals_segment, lasso_segment)
        svr_val = smape(actuals_segment, svr_segment)
        xgb_val = smape(actuals_segment, xgb_segment)
        lightgbm_val = smape(actuals_segment, lightgbm_segment)
        rf_val = smape(actuals_segment, rf_segment)
        lasso_smape.append(lasso_val)
        svr_smape.append(svr_val)
        xgb_smape.append(xgb_val)
        lightgbm_smape.append(lightgbm_val)
        rf_smape.append(rf_val)

    plt.figure(figsize=(30, 5))
    plt.plot(xpoints, lasso_smape, label = "Lasso Regression")
    plt.plot(xpoints, svr_smape, label = "Support Vector Regression")
    plt.plot(xpoints, xgb_smape, label = "XGBoost")
    plt.plot(xpoints, lightgbm_smape, label = "LightGBM")
    plt.plot(xpoints, rf_smape, label = "Random Forest")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xlabel('Prediction Day')
    plt.ylabel('SMAPE (%)')
    plt.title(f'Daily SMAPE (%) for 2025 for Shallow Learners for {price_zone}')
    plt.legend(loc = 'center right')
    plt.tight_layout()
    plt.savefig(f"SMAPE_graph_shallow_{price_zone}")
    plt.show()

def generate_rmse_graph(price_zone: str, xpoints: List[str],
                         folder: str):
    if price_zone == "DK1":
        filename_lasso_DK1 = "DK1_Lasso_predictions2.csv"
        filepath_lasso_DK1 = os.path.join(folder, filename_lasso_DK1)
        df_lasso = pd.read_csv(filepath_lasso_DK1, decimal = ",")

        filename_SVR_DK1 = "DK1_SVR_predictions.csv"
        filepath_SVR_DK1 = os.path.join(folder, filename_SVR_DK1)
        df_SVR = pd.read_csv(filepath_SVR_DK1, decimal = ",")

        filename_XGB_DK1 = "DK1_XGBoost_predictions2.csv"
        filepath_XGB_DK1 = os.path.join(folder, filename_XGB_DK1)
        df_XGB = pd.read_csv(filepath_XGB_DK1, decimal = ",")

        filename_lightgbm_DK1 = "DK1_LightGBM_predictions2.csv"
        filepath_lightgbm_DK1 = os.path.join(folder, filename_lightgbm_DK1)
        df_lightgbm = pd.read_csv(filepath_lightgbm_DK1, decimal = ",")

        filename_RF_DK1 = "DK1_RF_predictions2.csv"
        filepath_RF_DK1 = os.path.join(folder, filename_RF_DK1)
        df_RF = pd.read_csv(filepath_RF_DK1, decimal = ",")
    elif price_zone == "DK2":
        filename_lasso_DK2 = "DK2_Lasso_predictions2.csv"
        filepath_lasso_DK2 = os.path.join(folder, filename_lasso_DK2)
        df_lasso = pd.read_csv(filepath_lasso_DK2, decimal = ",")

        filename_SVR_DK2 = "DK2_SVR_predictions.csv"
        filepath_SVR_DK2 = os.path.join(folder, filename_SVR_DK2)
        df_SVR = pd.read_csv(filepath_SVR_DK2, decimal = ",")

        filename_XGB_DK2 = "DK2_XGBoost_predictions2.csv"
        filepath_XGB_DK2 = os.path.join(folder, filename_XGB_DK2)
        df_XGB = pd.read_csv(filepath_XGB_DK2, decimal = ",")

        filename_lightgbm_DK2 = "DK2_LightGBM_predictions2.csv"
        filepath_lightgbm_DK2 = os.path.join(folder, filename_lightgbm_DK2)
        df_lightgbm = pd.read_csv(filepath_lightgbm_DK2, decimal = ",")

        filename_RF_DK2 = "DK2_RF_predictions2.csv"
        filepath_RF_DK2 = os.path.join(folder, filename_RF_DK2)
        df_RF = pd.read_csv(filepath_RF_DK2, decimal = ",")
    else:
        print("The correct price zone was not given.")
        print("Please try again.")
        sys.exit()
    
    actuals = df_lasso["DKPrice"].values.tolist()

    lasso_preds = df_lasso["Prediction"].values.tolist()
    svr_preds = df_SVR["Prediction"].values.tolist()
    xgb_preds = df_XGB["Prediction"].values.tolist()
    lightgbm_preds = df_lightgbm["Prediction"].values.tolist()
    rf_preds = df_RF["Prediction"].values.tolist()

    lasso_rmse = []
    svr_rmse = []
    xgb_rmse = []
    lightgbm_rmse = []
    rf_rmse = []
    
    for i in range(0, len(actuals), 24):
        actuals_segment = actuals[i:i+24]
        lasso_segment = lasso_preds[i:i+24]
        svr_segment = svr_preds[i:i+24]
        xgb_segment = xgb_preds[i:i+24]
        lightgbm_segment = lightgbm_preds[i:i+24]
        rf_segment = rf_preds[i:i+24]
        lasso_val = np.sqrt(mean_squared_error(actuals_segment, lasso_segment))
        svr_val = np.sqrt(mean_squared_error(actuals_segment, svr_segment))
        xgb_val = np.sqrt(mean_squared_error(actuals_segment, xgb_segment))
        lightgbm_val = np.sqrt(mean_squared_error(actuals_segment, lightgbm_segment))
        rf_val = np.sqrt(mean_squared_error(actuals_segment, rf_segment))
        lasso_rmse.append(lasso_val)
        svr_rmse.append(svr_val)
        xgb_rmse.append(xgb_val)
        lightgbm_rmse.append(lightgbm_val)
        rf_rmse.append(rf_val)

    plt.figure(figsize=(30, 5))
    plt.plot(xpoints, lasso_rmse, label = "Lasso Regression")
    plt.plot(xpoints, svr_rmse, label = "Support Vector Regression")
    plt.plot(xpoints, xgb_rmse, label = "XGBoost")
    plt.plot(xpoints, lightgbm_rmse, label = "LightGBM")
    plt.plot(xpoints, rf_rmse, label = "Random Forest")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xlabel('Prediction Day')
    plt.ylabel('RMSE (DKK/MWh)')
    plt.title(f'Daily RMSE (DKK/MWh) for 2025 for Shallow Learners for {price_zone}')
    plt.legend(loc = 'center right')
    plt.tight_layout()
    plt.savefig(f"RMSE_graph_shallow_{price_zone}")
    plt.show()

def generate_mae_graph(price_zone: str, xpoints: List[str],
                         folder: str):
    if price_zone == "DK1":
        filename_lasso_DK1 = "DK1_Lasso_predictions2.csv"
        filepath_lasso_DK1 = os.path.join(folder, filename_lasso_DK1)
        df_lasso = pd.read_csv(filepath_lasso_DK1, decimal = ",")

        filename_SVR_DK1 = "DK1_SVR_predictions.csv"
        filepath_SVR_DK1 = os.path.join(folder, filename_SVR_DK1)
        df_SVR = pd.read_csv(filepath_SVR_DK1, decimal = ",")

        filename_XGB_DK1 = "DK1_XGBoost_predictions2.csv"
        filepath_XGB_DK1 = os.path.join(folder, filename_XGB_DK1)
        df_XGB = pd.read_csv(filepath_XGB_DK1, decimal = ",")

        filename_lightgbm_DK1 = "DK1_LightGBM_predictions2.csv"
        filepath_lightgbm_DK1 = os.path.join(folder, filename_lightgbm_DK1)
        df_lightgbm = pd.read_csv(filepath_lightgbm_DK1, decimal = ",")

        filename_RF_DK1 = "DK1_RF_predictions2.csv"
        filepath_RF_DK1 = os.path.join(folder, filename_RF_DK1)
        df_RF = pd.read_csv(filepath_RF_DK1, decimal = ",")
    elif price_zone == "DK2":
        filename_lasso_DK2 = "DK2_Lasso_predictions2.csv"
        filepath_lasso_DK2 = os.path.join(folder, filename_lasso_DK2)
        df_lasso = pd.read_csv(filepath_lasso_DK2, decimal = ",")

        filename_SVR_DK2 = "DK2_SVR_predictions.csv"
        filepath_SVR_DK2 = os.path.join(folder, filename_SVR_DK2)
        df_SVR = pd.read_csv(filepath_SVR_DK2, decimal = ",")

        filename_XGB_DK2 = "DK2_XGBoost_predictions2.csv"
        filepath_XGB_DK2 = os.path.join(folder, filename_XGB_DK2)
        df_XGB = pd.read_csv(filepath_XGB_DK2, decimal = ",")

        filename_lightgbm_DK2 = "DK2_LightGBM_predictions2.csv"
        filepath_lightgbm_DK2 = os.path.join(folder, filename_lightgbm_DK2)
        df_lightgbm = pd.read_csv(filepath_lightgbm_DK2, decimal = ",")

        filename_RF_DK2 = "DK2_RF_predictions2.csv"
        filepath_RF_DK2 = os.path.join(folder, filename_RF_DK2)
        df_RF = pd.read_csv(filepath_RF_DK2, decimal = ",")
    else:
        print("The correct price zone was not given.")
        print("Please try again.")
        sys.exit()
    
    actuals = df_lasso["DKPrice"].values.tolist()

    lasso_preds = df_lasso["Prediction"].values.tolist()
    svr_preds = df_SVR["Prediction"].values.tolist()
    xgb_preds = df_XGB["Prediction"].values.tolist()
    lightgbm_preds = df_lightgbm["Prediction"].values.tolist()
    rf_preds = df_RF["Prediction"].values.tolist()

    lasso_mae = []
    svr_mae = []
    xgb_mae = []
    lightgbm_mae = []
    rf_mae = []
    
    for i in range(0, len(actuals), 24):
        actuals_segment = actuals[i:i+24]
        lasso_segment = lasso_preds[i:i+24]
        svr_segment = svr_preds[i:i+24]
        xgb_segment = xgb_preds[i:i+24]
        lightgbm_segment = lightgbm_preds[i:i+24]
        rf_segment = rf_preds[i:i+24]
        lasso_val = mean_absolute_error(actuals_segment, lasso_segment)
        svr_val = mean_absolute_error(actuals_segment, svr_segment)
        xgb_val = mean_absolute_error(actuals_segment, xgb_segment)
        lightgbm_val = mean_absolute_error(actuals_segment, lightgbm_segment)
        rf_val = mean_absolute_error(actuals_segment, rf_segment)
        lasso_mae.append(lasso_val)
        svr_mae.append(svr_val)
        xgb_mae.append(xgb_val)
        lightgbm_mae.append(lightgbm_val)
        rf_mae.append(rf_val)

    plt.figure(figsize=(30, 5))
    plt.plot(xpoints, lasso_mae, label = "Lasso Regression")
    plt.plot(xpoints, svr_mae, label = "Support Vector Regression")
    plt.plot(xpoints, xgb_mae, label = "XGBoost")
    plt.plot(xpoints, lightgbm_mae, label = "LightGBM")
    plt.plot(xpoints, rf_mae, label = "Random Forest")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xlabel('Prediction Day')
    plt.ylabel('MAE (DKK/MWh)')
    plt.title(f'Daily MAE (DKK/MWh) for 2025 for Shallow Learners for {price_zone}')
    plt.legend(loc = 'center right')
    plt.tight_layout()
    plt.savefig(f"MAE_graph_shallow_{price_zone}")
    plt.show()

def generate_quarterly_smape_graph(price_zone: str, xpoints: List[str],
                                   folder: str):
    if price_zone == "DK1":
        filename_lasso = "DK1_Lasso_predictions2.csv"
        filename_SVR = "DK1_SVR_predictions.csv"
        filename_XGB = "DK1_XGBoost_predictions2.csv"
        filename_lightgbm = "DK1_LightGBM_predictions2.csv"
        filename_RF = "DK1_RF_predictions2.csv"
    elif price_zone == "DK2":
        filename_lasso = "DK2_Lasso_predictions2.csv"
        filename_SVR = "DK2_SVR_predictions.csv"
        filename_XGB = "DK2_XGBoost_predictions2.csv"
        filename_lightgbm = "DK2_LightGBM_predictions2.csv"
        filename_RF = "DK2_RF_predictions2.csv"
    else:
        print("The correct price zone was not given.")
        print("Please try again.")
        sys.exit()

    df_lasso = pd.read_csv(os.path.join(folder, filename_lasso), decimal=",")
    df_SVR = pd.read_csv(os.path.join(folder, filename_SVR), decimal=",")
    df_XGB = pd.read_csv(os.path.join(folder, filename_XGB), decimal=",")
    df_lightgbm = pd.read_csv(os.path.join(folder, filename_lightgbm), decimal=",")
    df_RF = pd.read_csv(os.path.join(folder, filename_RF), decimal=",")

    actuals = df_lasso["DKPrice"].values.tolist()
    lasso_preds = df_lasso["Prediction"].values.tolist()
    svr_preds = df_SVR["Prediction"].values.tolist()
    xgb_preds = df_XGB["Prediction"].values.tolist()
    lightgbm_preds = df_lightgbm["Prediction"].values.tolist()
    rf_preds = df_RF["Prediction"].values.tolist()

    lasso_smape, svr_smape, xgb_smape, lightgbm_smape, rf_smape = [], [], [], [], []

    for i in range(0, len(actuals), 24):
        actuals_segment = actuals[i:i+24]
        lasso_smape.append(smape(actuals_segment, lasso_preds[i:i+24]))
        svr_smape.append(smape(actuals_segment, svr_preds[i:i+24]))
        xgb_smape.append(smape(actuals_segment, xgb_preds[i:i+24]))
        lightgbm_smape.append(smape(actuals_segment, lightgbm_preds[i:i+24]))
        rf_smape.append(smape(actuals_segment, rf_preds[i:i+24]))

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
        ax.plot(xq, [v for v, m in zip(lasso_smape, mask) if m], label="Lasso Regression")
        ax.plot(xq, [v for v, m in zip(svr_smape, mask) if m], label="Support Vector Regression")
        ax.plot(xq, [v for v, m in zip(xgb_smape, mask) if m], label="XGBoost")
        ax.plot(xq, [v for v, m in zip(lightgbm_smape, mask) if m], label="LightGBM")
        ax.plot(xq, [v for v, m in zip(rf_smape, mask) if m], label="Random Forest")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_xlabel('Prediction Day')
        ax.set_ylabel('SMAPE (%)')
        ax.set_title(quarter)
        ax.legend(loc='upper center')

    fig.suptitle(f'Quarterly Daily SMAPE (%) for 2025 for Shallow Learners for {price_zone}', fontsize=14)
    plt.tight_layout(pad = 2.0)
    plt.savefig(f"SMAPE_graph_shallow_quarterly_{price_zone}")
    plt.show()

def generate_quarterly_smape_graph_baseline(price_zone: str, xpoints: List[str],
                                            naive_folder: str, MA_folder: str,
                                            seasonal_folder: str):
    if price_zone == "DK1":
        filename_naive = "DK1_daily_results_naive.csv"
        filename_MA = "DK1_daily_results_MA.csv"
        filename_seasonal = "DK1_daily_results_seasonal.csv"
        df_naive = pd.read_csv(os.path.join(naive_folder, filename_naive))
        df_MA = pd.read_csv(os.path.join(MA_folder, filename_MA))
        df_seasonal = pd.read_csv(os.path.join(seasonal_folder, filename_seasonal))
    elif price_zone == "DK2":
        filename_naive = "DK2_daily_results_naive.csv"
        filename_MA = "DK2_daily_results_MA.csv"
        filename_seasonal = "DK2_daily_results_seasonal.csv"
        df_naive = pd.read_csv(os.path.join(naive_folder, filename_naive))
        df_MA = pd.read_csv(os.path.join(MA_folder, filename_MA))
        df_seasonal = pd.read_csv(os.path.join(seasonal_folder, filename_seasonal))
    else:
        print("The correct price zone was not given.")
        print("Please try again.")
        sys.exit()

    column = "daily_smape"
    naive_list = df_naive[column].values.tolist()
    MA_list = df_MA[column].values.tolist()
    seasonal_list = df_seasonal[column].values.tolist()

    xpoints = pd.to_datetime(xpoints)

    quarters = [
        "Q1 (Jan-Mar) 2025", "Q2 (Apr-Jun) 2025", "Q3 (Jul-Sep) 2025", "Q4 (Oct-Dec) 2025"
    ]
    masks = [
        (xpoints.month >= 1) & (xpoints.month <= 3),
        (xpoints.month >= 4) & (xpoints.month <= 6),
        (xpoints.month >= 7) & (xpoints.month <= 9),
        (xpoints.month >= 10) & (xpoints.month <= 12),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(20, 20))
    axes = axes.flatten()

    for idx, (quarter, mask) in enumerate(zip(quarters, masks)):
        ax = axes[idx]
        xq = xpoints[mask]
        ax.plot(xq, [v for v, m in zip(naive_list, mask) if m], label="Naive Persistence")
        ax.plot(xq, [v for v, m in zip(MA_list, mask) if m], label="Moving Average")
        ax.plot(xq, [v for v, m in zip(seasonal_list, mask) if m], label="Seasonal Naive")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_xlabel('Prediction Day')
        ax.set_ylabel('SMAPE (%)')
        ax.set_title(quarter)
        ax.legend(loc='center right')

    fig.suptitle(f'Quarterly Daily SMAPE (%) for 2025 for Baseline Models for {price_zone}', fontsize=14)
    plt.tight_layout(pad=2.0)
    plt.savefig(f"SMAPE_graph_baseline_quarterly_{price_zone}")
    plt.show()

def baseline_data(file, folder):
    filepath = os.path.join(folder, file)
    df = pd.read_csv(filepath)

    smape_averages = {}
    column = "daily_smape"
    for i in range(7):
        smape_averages[f'avg_smape_day_{i+1}'] = df[column].iloc[i::7].mean()
    smape_list = list(smape_averages.values())

    rmse_averages = {}
    column = "daily_rmse"
    for i in range(7):
        rmse_averages[f'avg_rmse_day_{i+1}'] = df[column].iloc[i::7].mean()
    rmse_list = list(rmse_averages.values())

    mae_averages = {}
    column = "daily_mae"
    for i in range(7):
        mae_averages[f'avg_mae_day_{i+1}'] = df[column].iloc[i::7].mean()
    mae_list = list(mae_averages.values())

    return smape_list, rmse_list, mae_list

def baseline_7_day_smape(smape_naive: List[float], smape_MA: List[float],
                         smape_seasonal: List[float], price_zone):
    xpoints = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]

    plt.plot(xpoints, smape_naive, label = "Naive Persistence")
    plt.plot(xpoints, smape_MA, label = "Moving Average")
    plt.plot(xpoints, smape_seasonal, label = "Seasonal Naive")
    plt.xlabel('Prediction Day')
    plt.ylabel('SMAPE (%)')
    plt.title(f'7-Day SMAPE (%) Development for Baseline Models for {price_zone}')
    plt.legend()
    plt.savefig(f"baseline_7day_SMAPE_{price_zone}")
    plt.show()

def baseline_7_day_rmse(rmse_naive: List[float], rmse_MA: List[float],
                         rmse_seasonal: List[float], price_zone):
    xpoints = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]

    plt.plot(xpoints, rmse_naive, label = "Naive Persistence")
    plt.plot(xpoints, rmse_MA, label = "Moving Average")
    plt.plot(xpoints, rmse_seasonal, label = "Seasonal Naive")
    plt.xlabel('Prediction Day')
    plt.ylabel('RMSE (DKK/MWh)')
    plt.title(f'7-Day RMSE (DKK/MWh) Development for Baseline Models for {price_zone}')
    plt.legend()
    plt.savefig(f"baseline_7day_RMSE_{price_zone}")
    plt.show()

def baseline_7_day_mae(mae_naive: List[float], mae_MA: List[float],
                         mae_seasonal: List[float], price_zone):
    xpoints = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]

    plt.plot(xpoints, mae_naive, label = "Naive Persistence")
    plt.plot(xpoints, mae_MA, label = "Moving Average")
    plt.plot(xpoints, mae_seasonal, label = "Seasonal Naive")
    plt.xlabel('Prediction Day')
    plt.ylabel('MAE (DKK/MWh)')
    plt.title(f'7-Day MAE (DKK/MWh) Development for Baseline Models for {price_zone}')
    plt.legend()
    plt.savefig(f"baseline_7day_MAE_{price_zone}")
    plt.show()

def read_shallow_results(folder: str, price_zone: str):
    if price_zone == "DK1":
        lasso_name = "DK1_final_Lasso_results.csv_2.csv"
        svr_name = "DK1_final_SVR_results.csv_2.csv"
        xgb_name = "DK1_final_XGB_results.csv_2.csv"
        lightgbm_name = "DK1_LightGBM_final_results.csv_1.csv"
        rf_name = "DK1_RF_final_results.csv_1.csv"
    elif price_zone == "DK2":
        lasso_name = "DK2_final_Lasso_results.csv_2.csv"
        svr_name = "DK2_final_SVR_results.csv_2.csv"
        xgb_name = "DK2_final_XGB_results.csv_2.csv"
        lightgbm_name = "DK2_LightGBM_final_results.csv_1.csv"
        rf_name = "DK2_RF_final_results.csv_1.csv"
    else:
        print("The correct price zone was not given.")
        print("Please try again.")
        sys.exit()
    
    lasso_name = os.path.join(folder, lasso_name)
    svr_name = os.path.join(folder, svr_name)
    xgb_name = os.path.join(folder, xgb_name)
    lightgbm_name = os.path.join(folder, lightgbm_name)
    rf_name = os.path.join(folder, rf_name)

    df_lasso = pd.read_csv(lasso_name, decimal = ",")
    df_svr = pd.read_csv(svr_name, decimal = ",")
    df_xgb = pd.read_csv(xgb_name, decimal = ",")
    df_lightgbm = pd.read_csv(lightgbm_name, decimal = ",")
    df_rf = pd.read_csv(rf_name, decimal = ",")

    df_lasso = df_lasso.iloc[:, 8:]
    lasso_list = df_lasso.values.tolist()[0]

    if price_zone == "DK1":
        df_svr = df_svr.iloc[:, 12:]
        svr_list = df_svr.values.tolist()[0]
    elif price_zone == "DK2":
        df_svr = df_svr.iloc[:, 11:]
        svr_list = df_svr.values.tolist()[0]

    df_xgb = df_xgb.iloc[:, 12:]
    xgb_list = df_xgb.values.tolist()[0]

    df_lightgbm = df_lightgbm.iloc[:, 19:]
    lightgbm_list = df_lightgbm.values.tolist()[0]

    df_rf = df_rf.iloc[:, 13:]
    rf_list = df_rf.values.tolist()[0]

    return lasso_list, svr_list, xgb_list, lightgbm_list, rf_list

def shallow_7_day_smape(lasso_list, svr_list, xgb_list, 
                        lightgbm_list, rf_list, price_zone):
    xpoints = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]

    plt.plot(xpoints, lasso_list, label = "Lasso Regression")
    plt.plot(xpoints, svr_list, label = "Support Vector Regression")
    plt.plot(xpoints, xgb_list, label = "XGBoost")
    plt.plot(xpoints, lightgbm_list, label = "LightGBM")
    plt.plot(xpoints, rf_list, label = "Random Forest")
    plt.xlabel('Prediction Day')
    plt.ylabel('SMAPE (%)')
    plt.title(f'7-Day SMAPE (%) Development for Shallow Learners for {price_zone}')
    plt.legend()
    plt.savefig(f"shallow_7day_smape_{price_zone}")
    plt.show()

def make_shallow_7_day_metrics(folder, file, price_zone):
    filepath = os.path.join(folder, file)

    df = pd.read_csv(filepath, decimal = ",")

    actuals = df["DKPrice"].values.tolist()
    preds = df["Prediction"].values.tolist()

    rmse_list = []
    mae_list = []

    for i in range(0, len(actuals), 24):
        actuals_segment = actuals[i:i+24]
        preds_segment = preds[i:i+24]
        rmse_val = np.sqrt(mean_squared_error(actuals_segment, preds_segment))
        mae_val = mean_absolute_error(actuals_segment, preds_segment)
        rmse_list.append(rmse_val)
        mae_list.append(mae_val)

    rmse_averages = {}
    mae_averages = {}
    for i in range(7):
        rmse_averages[f'avg_rmse_day_{i+1}'] = sum(rmse_list[i::7]) / len(rmse_list[i::7])
        mae_averages[f'avg_mae_day_{i+1}'] = sum(mae_list[i::7]) / len(mae_list[i::7])

    rmse_final = list(rmse_averages.values())
    mae_final = list(mae_averages.values())

    return rmse_final, mae_final

def shallow_7_day_rmse(lasso_list, svr_list, xgb_list, 
                        lightgbm_list, rf_list, price_zone):
    xpoints = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]

    plt.plot(xpoints, lasso_list, label = "Lasso Regression")
    plt.plot(xpoints, svr_list, label = "Support Vector Regression")
    plt.plot(xpoints, xgb_list, label = "XGBoost")
    plt.plot(xpoints, lightgbm_list, label = "LightGBM")
    plt.plot(xpoints, rf_list, label = "Random Forest")
    plt.xlabel('Prediction Day')
    plt.ylabel('RMSE (DKK/MWh)')
    plt.title(f'7-Day RMSE (DKK/MWh) Development for Shallow Learners for {price_zone}')
    plt.legend()
    plt.savefig(f"shallow_7day_rmse_{price_zone}")
    plt.show()

def shallow_7_day_mae(lasso_list, svr_list, xgb_list, 
                        lightgbm_list, rf_list, price_zone):
    xpoints = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]

    plt.plot(xpoints, lasso_list, label = "Lasso Regression")
    plt.plot(xpoints, svr_list, label = "Support Vector Regression")
    plt.plot(xpoints, xgb_list, label = "XGBoost")
    plt.plot(xpoints, lightgbm_list, label = "LightGBM")
    plt.plot(xpoints, rf_list, label = "Random Forest")
    plt.xlabel('Prediction Day')
    plt.ylabel('MAE (DKK/MWh)')
    plt.title(f'7-Day MAE (DKK/MWh) Development for Shallow Learners for {price_zone}')
    plt.legend()
    plt.savefig(f"shallow_7day_mae_{price_zone}")
    plt.show()