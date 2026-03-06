# Importing packages
import os
import pandas as pd
from datetime import datetime

def read_data(file_name: str):

    # Getting the file path
    notebook_dir = os.getcwd()
    python_dir = os.path.dirname(notebook_dir)
    data_folder = os.path.join(python_dir, 'Data')
    data_file = os.path.join(data_folder, file_name)

    # Reading the data as a DataFrame
    df = pd.read_csv(data_file, decimal = ",")

    # Columns that need to change datatype
    cols = ['OffshoreWindPower',
        'OnshoreWindPower',
        'HydroPower',
        'SolarPower',
        'Biomass',
        'Biogas',
        'Waste',
        'FossilGas',
        'FossilOil',
        'FossilHardCoal',
        'ExchangeGreatBelt',
        'ExchangeGermany',
        'ExchangeSweden',
        'ExchangeNorway',
        'ExchangeNetherlands',
        'WindSpeed',
        'Radiation',
        'DKPrice',
        'DEPrice',
        'NO2Price',
        'SE3Price',
        'SE4Price',
        'OffshoreWindCapacity',
        'OnshoreWindCapacity',
        'SolarPowerCapacity',
        'GrossCon',
        'TotalProduction',
        'Year',
        'Month',
        'Day',
        'WeekDay',
        'Hour']
    
    df[cols] = df[cols].astype(float)

    # Saving datetime format 
    date_format = '%Y-%m-%d %H:%M:%S'

    temp_list = []
    for i in list(df["TimeUTC"]):
        n = 19
        j = i[:n]
        k = datetime.strptime(j, date_format)
        temp_list.append(k)
    df.insert(0, "Time", temp_list, True)
    df = df.drop("TimeUTC", axis = 1)

    df['TotalProduction_lag1'] = df['TotalProduction'].shift(1)  # value from 1 hour ago
    df['TotalProduction_lag24'] = df['TotalProduction'].shift(24)  # value from 24 hours ago
    df['Price_lag1'] = df['DKPrice'].shift(1)
    df['Price_lag24'] = df['DKPrice'].shift(24)

    # Dropping NaN rows
    df = df.dropna()

    # Moving price to first column
    col = df.pop('DKPrice')
    df.insert(0, 'DKPrice', col)

    # Filtering for price zone
    df_DK1 = df[df['DKZone'] == 'DK1']
    df_DK1 = df_DK1.drop('DKZone', axis = 1)
    df_DK1.reset_index(drop=True, inplace=True)
    df_DK2 = df[df['DKZone'] == 'DK2']
    df_DK2 = df_DK2.drop('DKZone', axis = 1)
    df_DK2.reset_index(drop=True, inplace=True)

    # Splitting between initial train and test set for DK1
    DK1_train_set = df_DK1.loc[df_DK1['Time'] < pd.Timestamp('2025-01-01')]
    DK1_train_set = DK1_train_set.drop('Time', axis = 1)
    DK1_train_weather = DK1_train_set[['WindSpeed','Radiation']].copy()
    DK1_test_set = df_DK1.loc[df_DK1['Time'] >= pd.Timestamp('2025-01-01')]
    DK1_test_set = DK1_test_set.drop('Time', axis = 1)
    DK1_test_weather = DK1_test_set[['WindSpeed','Radiation']].copy()

    # Printing the shape for training and test date for DK1
    print(f"Training data shape (DK1): {DK1_train_set.shape}")
    print(f"Test data shape (DK1): {DK1_test_set.shape}")
    print(f"Test set fraction (DK1): {len(DK1_test_set)/len(DK1_train_set):.2%}")

    # Splitting between initial train and test set for DK2
    DK2_train_set = df_DK2.loc[df_DK1['Time'] < pd.Timestamp('2025-01-01')]
    DK2_train_set = DK2_train_set.drop('Time', axis = 1)
    DK2_train_weather = DK2_train_set[['WindSpeed','Radiation']].copy()
    DK2_test_set = df_DK2.loc[df_DK1['Time'] >= pd.Timestamp('2025-01-01')]
    DK2_test_set = DK2_test_set.drop('Time', axis = 1)
    DK2_test_weather = DK2_test_set[['WindSpeed','Radiation']].copy()

    # Printing the shape for training and test date for DK2
    print(f"Training data shape (DK2): {DK2_train_set.shape}")
    print(f"Test data shape (DK2): {DK2_test_set.shape}")
    print(f"Test set fraction (DK2): {len(DK2_test_set)/len(DK2_train_set):.2%}")

    return (DK1_train_set, DK1_test_set, DK2_train_set, DK2_test_set, 
            DK1_train_weather, DK1_test_weather, DK2_train_weather, DK2_test_weather)