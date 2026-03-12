import os
import pandas as pd
from datetime import datetime

def load_data(file_name: str):
    """This function does the following:
    - Gets the working directory of this script, goes to the parent directory,
    then adds the "Data" folder to the end and then adds the file_name from the 
    input.
    - Reads the csv file.
    - Changes data types all number columns to float and removes timezone from 
    TimeUTC and renames it "Time".
    - Creates lag features for TotalProduction and DKPrice.
    - Splits the data into DK1 and DK2.
    - Splits the data into training and test set for DK1 and DK2.
    - Creates separate weather datasets for training and test.
    - Returns all datasets: train1, test1, train2, test2, train_weather1, 
    test_weather1, train_weather2, test_weather2.

    Arguments:
    file_name: The name of the csv file as a string.
    """
    # Getting the file path from the repository library structure
    notebook_dir = os.path.dirname(os.path.abspath(__file__))       # Data_loader.py directory
    print(f"Notebook_dir: {notebook_dir}")
    python_dir = os.path.dirname(notebook_dir)                      # parent directory of Data_loader.py
    print(f"Python_dir: {python_dir}")
    data_folder = os.path.join(python_dir, 'Data')
    print(f"Data_folder: {data_folder}")
    data_file = os.path.join(data_folder, file_name)

    # When running from my own library
    # data_file = r"C:\Users\n_and\OneDrive\Delt skrivebord\Data Science\Speciale\Energinet\Data\combined_data_cleaned_v4.csv"

    df = pd.read_csv(data_file, decimal = ",")

    # Columns that need to be floats
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
        'NLPrice',
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

    # Converting TimeUTC to datetime format 
    date_format = '%Y-%m-%d %H:%M:%S'

    temp_list = []
    for i in list(df["TimeUTC"]):
        n = 19
        j = i[:n]
        k = datetime.strptime(j, date_format)
        temp_list.append(k)
    df.insert(0, "Time", temp_list, True)
    df = df.drop("TimeUTC", axis = 1)

    # df['TotalProduction_lag1'] = df['TotalProduction'].shift(1)  # value from 1 hour ago
    # df['TotalProduction_lag24'] = df['TotalProduction'].shift(24)  # value from 24 hours ago
    df['GrossCon_lag1'] = df['GrossCon'].shift(1)  # value from 1 hour ago
    df['GrossCon_lag24'] = df['GrossCon'].shift(24)  # value from 24 hours ago
    df['GrossCon_lag168'] = df['GrossCon'].shift(168)  # value from 1 week ago
    df['Price_lag1'] = df['DKPrice'].shift(1)
    df['Price_lag24'] = df['DKPrice'].shift(24)
    df['Price_lag168'] = df['DKPrice'].shift(168)

    # Dropping NaN rows created by lag features - first 168 rows (hours) of 2016
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
    # DK1_train_set = DK1_train_set.drop('Time', axis = 1)
    DK1_train_weather = DK1_train_set[['WindSpeed','Radiation']].copy()
    DK1_test_set = df_DK1.loc[df_DK1['Time'] >= pd.Timestamp('2025-01-01')]
    # DK1_test_set = DK1_test_set.drop('Time', axis = 1)
    DK1_test_weather = DK1_test_set[['WindSpeed','Radiation']].copy()

    # Printing the shape for training and test date for DK1
    print(f"Training data shape (DK1): {DK1_train_set.shape}")
    print(f"Test data shape (DK1): {DK1_test_set.shape}")
    print(f"Test set fraction (DK1): {len(DK1_test_set)/len(df_DK1):.2%}")

    # Splitting between initial train and test set for DK2
    DK2_train_set = df_DK2.loc[df_DK2['Time'] < pd.Timestamp('2025-01-01')]
    # DK2_train_set = DK2_train_set.drop('Time', axis = 1)
    DK2_train_weather = DK2_train_set[['WindSpeed','Radiation']].copy()
    DK2_test_set = df_DK2.loc[df_DK2['Time'] >= pd.Timestamp('2025-01-01')]
    # DK2_test_set = DK2_test_set.drop('Time', axis = 1)
    DK2_test_weather = DK2_test_set[['WindSpeed','Radiation']].copy()

    # Printing the shape for training and test date for DK2
    print(f"Training data shape (DK2): {DK2_train_set.shape}")
    print(f"Test data shape (DK2): {DK2_test_set.shape}")
    print(f"Test set fraction (DK2): {len(DK2_test_set)/len(df_DK2):.2%}")

    return (DK1_train_set, DK1_test_set, DK2_train_set, DK2_test_set,
            DK1_train_weather, DK1_test_weather, DK2_train_weather, DK2_test_weather)

if __name__ == "__main__":
    load_data("combined_data_cleaned.csv")