import pandas as pd
import os

# Path to data
data_folder = os.path.dirname(os.path.realpath(__file__))
data_file = os.path.join(data_folder, 'combined_data_cleaned_v2.csv')

# Load data
df = pd.read_csv(data_file, decimal=',')

# Add TotalProduction column
power_cols = [
    'OffshoreWindPower',
    'OnshoreWindPower',
    'HydroPower',
    'SolarPower',
    'Biomass',
    'Biogas',
    'Waste',
    'FossilGas',
    'FossilOil',
    'FossilHardCoal'
]

df[power_cols] = df[power_cols].astype(float)
df['TotalProduction'] = df[power_cols].sum(axis=1)

# Parse datetime and extract time features
df['TimeUTC'] = pd.to_datetime(df['TimeUTC'])
df['Year'] = df['TimeUTC'].dt.year
df['Month'] = df['TimeUTC'].dt.month
df['Day'] = df['TimeUTC'].dt.day
df['WeekDay'] = df['TimeUTC'].dt.weekday  # Monday=0, Sunday=6
df['Hour'] = df['TimeUTC'].dt.hour

# Save
output_file = os.path.join(data_folder, 'combined_data_cleaned_v3.csv')
df.to_csv(output_file, index=False)
print(f"Saved to {output_file}")
print(f"Shape: {df.shape}")
print(df[['TimeUTC', 'TotalProduction', 'Year',
      'Month', 'Day', 'WeekDay', 'Hour']].head())
