"""
Energy Data Visualization Dashboard
Interactive Dash application for visualizing Danish energy production, exchange, prices, and weather data
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
import os

# Try to load data from parquet first (faster), fall back to CSV
DATA_PATH = os.path.join(os.path.dirname(__file__), 'combined_data_cleaned_v4')

# Global variable for cached data
CACHED_DATA = {}


def load_data():
    """Load data from parquet or CSV file and merge with CO2 data"""
    try:
        if os.path.exists(DATA_PATH + '.parquet'):
            print("Loading parquet file...")
            df = pd.read_parquet(DATA_PATH + '.parquet')
            print(f"Loaded {len(df)} rows from parquet")
        else:
            print("Loading CSV file...")
            df = pd.read_csv(DATA_PATH + '.csv')
            print(f"Loaded {len(df)} rows from CSV")

        # Ensure TimeUTC is datetime
        df['TimeUTC'] = pd.to_datetime(df['TimeUTC'])

        # Load CO2 and Great Britain exchange data from GenerationProdTypeExchange.csv
        print("Loading CO2 and Great Britain exchange data...")
        co2_path = os.path.join(os.path.dirname(
            __file__), 'Energinet data', 'GenerationProdTypeExchange.csv')
        if os.path.exists(co2_path):
            co2_df = pd.read_csv(co2_path, sep=';', decimal=',')
            co2_df['TimeUTC'] = pd.to_datetime(co2_df['TimeUTC'], utc=True)

            # Select only the columns we need
            co2_df = co2_df[['TimeUTC', 'PriceArea',
                             'CO2PerkWh', 'ExchangeGreatBritain']].copy()
            co2_df.rename(columns={'PriceArea': 'DKZone',
                          'CO2PerkWh': 'CO2perkWh'}, inplace=True)

            # Ensure TimeUTC columns have compatible dtypes
            df['TimeUTC'] = pd.to_datetime(df['TimeUTC'], utc=True)

            # Merge with main dataframe
            df = df.merge(co2_df, on=['TimeUTC', 'DKZone'], how='left')
            print(
                f"Merged CO2 data: {df['CO2perkWh'].notna().sum()} rows with CO2 values")
            print(
                f"Merged Great Britain exchange: {df['ExchangeGreatBritain'].notna().sum()} rows with data")
        else:
            print(f"Warning: CO2 data file not found at {co2_path}")
            df['CO2perkWh'] = None
            df['ExchangeGreatBritain'] = None

        # Sort by time
        df = df.sort_values('TimeUTC')

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def preprocess_resolutions(df):
    """Precalculate and cache data at different resolutions"""
    print("Precalculating resolutions...")

    # Production columns
    production_cols = ['OffshoreWindPower', 'OnshoreWindPower', 'HydroPower',
                       'SolarPower', 'Biomass', 'Biogas', 'Waste',
                       'FossilGas', 'FossilOil', 'FossilHardCoal']

    # Exchange columns
    exchange_cols = ['ExchangeGreatBelt', 'ExchangeGermany', 'ExchangeSweden',
                     'ExchangeNorway', 'ExchangeNetherlands', 'ExchangeGreatBritain']

    # Price columns
    price_cols = ['DKPrice', 'DEPrice', 'NO2Price', 'SE3Price', 'SE4Price', 'NLPrice']

    # Weather columns
    weather_cols = ['WindSpeed', 'Radiation']

    # CO2 column
    co2_cols = ['CO2perkWh']

    # Capacity and GrossCon columns
    capacity_cols = ['OffshoreWindCapacity',
                     'OnshoreWindCapacity', 'SolarPowerCapacity']
    grosscon_cols = ['GrossCon']

    # Daily resolution
    print("  Calculating daily averages...")
    daily_df = df.copy()
    daily_df['Date'] = daily_df['TimeUTC'].dt.date

    daily_agg_funcs = {}
    for col in production_cols + exchange_cols + weather_cols + co2_cols + grosscon_cols:
        if col in df.columns:
            daily_agg_funcs[col] = 'mean'
    for col in price_cols:
        if col in df.columns:
            daily_agg_funcs[col] = 'mean'
    for col in capacity_cols:
        if col in df.columns:
            # capacity is constant within a month
            daily_agg_funcs[col] = 'first'

    daily_data = daily_df.groupby(['Date', 'DKZone']).agg(
        daily_agg_funcs).reset_index()
    daily_data['TimeUTC'] = pd.to_datetime(daily_data['Date'])

    # Weekly resolution
    print("  Calculating weekly averages...")
    weekly_df = df.copy()
    weekly_df['Week'] = weekly_df['TimeUTC'].dt.to_period(
        'W').apply(lambda r: r.start_time)

    weekly_agg_funcs = {}
    for col in production_cols + exchange_cols + weather_cols + co2_cols + grosscon_cols:
        if col in df.columns:
            weekly_agg_funcs[col] = 'mean'
    for col in price_cols:
        if col in df.columns:
            weekly_agg_funcs[col] = 'mean'
    for col in capacity_cols:
        if col in df.columns:
            weekly_agg_funcs[col] = 'first'

    weekly_data = weekly_df.groupby(['Week', 'DKZone']).agg(
        weekly_agg_funcs).reset_index()
    weekly_data['TimeUTC'] = weekly_data['Week']

    # Monthly resolution
    print("  Calculating monthly averages...")
    monthly_df = df.copy()
    monthly_df['Month'] = monthly_df['TimeUTC'].dt.to_period(
        'M').apply(lambda r: r.start_time)

    monthly_agg_funcs = {}
    for col in production_cols + exchange_cols + weather_cols + co2_cols + grosscon_cols:
        if col in df.columns:
            monthly_agg_funcs[col] = 'mean'
    for col in price_cols:
        if col in df.columns:
            monthly_agg_funcs[col] = 'mean'
    for col in capacity_cols:
        if col in df.columns:
            monthly_agg_funcs[col] = 'first'

    monthly_data = monthly_df.groupby(['Month', 'DKZone']).agg(
        monthly_agg_funcs).reset_index()
    monthly_data['TimeUTC'] = monthly_data['Month']

    print("Precalculation complete!")

    return {
        'hourly': df,
        'daily': daily_data,
        'weekly': weekly_data,
        'monthly': monthly_data
    }


def filter_data(df, start_date, end_date):
    """Filter dataframe by date range, handling both tz-aware and tz-naive TimeUTC"""
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    # Match timezone awareness of the DataFrame
    if df['TimeUTC'].dt.tz is not None:
        start_date = start_date.tz_localize(
            'UTC') if start_date.tzinfo is None else start_date
        end_date = end_date.tz_localize(
            'UTC') if end_date.tzinfo is None else end_date
    else:
        start_date = start_date.tz_localize(
            None) if start_date.tzinfo is not None else start_date
        end_date = end_date.tz_localize(
            None) if end_date.tzinfo is not None else end_date
    mask = (df['TimeUTC'] >= start_date) & (df['TimeUTC'] <= end_date)
    return df[mask]


def create_production_chart(df, combine_zones):
    """Create production line chart"""
    production_cols = {
        'OffshoreWindPower': 'Offshore Wind',
        'OnshoreWindPower': 'Onshore Wind',
        'HydroPower': 'Hydro',
        'SolarPower': 'Solar',
        'Biomass': 'Biomass',
        'Biogas': 'Biogas',
        'Waste': 'Waste',
        'FossilGas': 'Fossil Gas',
        'FossilOil': 'Fossil Oil',
        'FossilHardCoal': 'Fossil Coal'
    }

    fig = go.Figure()

    if combine_zones:
        # Combine DK1 and DK2
        combined_df = df.groupby('TimeUTC')[list(
            production_cols.keys())].sum().reset_index()

        for col, label in production_cols.items():
            if col in combined_df.columns:
                fig.add_trace(go.Scatter(
                    x=combined_df['TimeUTC'],
                    y=combined_df[col],
                    mode='lines',
                    name=label,
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Power: %{y:.2f} MW<br>' +
                                  '<extra></extra>'
                ))
    else:
        # Separate DK1 and DK2
        for zone in ['DK1', 'DK2']:
            zone_df = df[df['DKZone'] == zone].copy()

            for col, label in production_cols.items():
                if col in zone_df.columns:
                    fig.add_trace(go.Scatter(
                        x=zone_df['TimeUTC'],
                        y=zone_df[col],
                        mode='lines',
                        name=f'{label} ({zone})',
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                      'Time: %{x}<br>' +
                                      'Power: %{y:.2f} MW<br>' +
                                      '<extra></extra>'
                    ))

    fig.update_layout(
        title='Energy Production by Source',
        xaxis_title='Time',
        yaxis_title='Power (MW)',
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top",
                    y=1, xanchor="left", x=1.02),
        height=500
    )

    return fig


def create_overview_chart(df, combine_zones):
    """Create overview chart with Total Production, Gross Consumption, and Capacities"""
    production_cols = ['OffshoreWindPower', 'OnshoreWindPower', 'HydroPower',
                       'SolarPower', 'Biomass', 'Biogas', 'Waste',
                       'FossilGas', 'FossilOil', 'FossilHardCoal']

    overview_lines = {
        'TotalProduction': ('Total Production', 'black', 'solid'),
        'GrossCon': ('Gross Consumption', 'red', 'solid'),
        'OffshoreWindCapacity': ('Offshore Wind Capacity', 'steelblue', 'dash'),
        'OnshoreWindCapacity': ('Onshore Wind Capacity', 'green', 'dash'),
        'SolarPowerCapacity': ('Solar Power Capacity', 'orange', 'dash'),
    }

    fig = go.Figure()

    if combine_zones:
        # Combine DK1 and DK2
        sum_cols = [c for c in production_cols if c in df.columns]
        agg_dict = {c: 'sum' for c in sum_cols}
        if 'GrossCon' in df.columns:
            agg_dict['GrossCon'] = 'sum'
        for cap in ['OffshoreWindCapacity', 'OnshoreWindCapacity', 'SolarPowerCapacity']:
            if cap in df.columns:
                agg_dict[cap] = 'sum'

        combined_df = df.groupby('TimeUTC').agg(agg_dict).reset_index()
        combined_df['TotalProduction'] = combined_df[sum_cols].sum(axis=1)

        for col, (label, color, dash) in overview_lines.items():
            if col in combined_df.columns:
                fig.add_trace(go.Scatter(
                    x=combined_df['TimeUTC'],
                    y=combined_df[col],
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2, dash=dash),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Value: %{y:.2f} MW<br>' +
                                  '<extra></extra>'
                ))
    else:
        zone_styles = {
            'DK1': {'width': 2},
            'DK2': {'width': 2}
        }
        for zone in ['DK1', 'DK2']:
            zone_df = df[df['DKZone'] == zone].copy()
            sum_cols = [c for c in production_cols if c in zone_df.columns]
            zone_df['TotalProduction'] = zone_df[sum_cols].sum(axis=1)

            for col, (label, color, dash) in overview_lines.items():
                if col in zone_df.columns:
                    fig.add_trace(go.Scatter(
                        x=zone_df['TimeUTC'],
                        y=zone_df[col],
                        mode='lines',
                        name=f'{label} ({zone})',
                        line=dict(color=color, width=zone_styles[zone]['width'],
                                  dash=dash if zone == 'DK1' else 'dot'),
                        hovertemplate=f'<b>{label} ({zone})</b><br>' +
                        'Time: %{x}<br>' +
                        'Value: %{y:.2f} MW<br>' +
                        '<extra></extra>'
                    ))

    fig.update_layout(
        title='Production, Consumption & Installed Capacity Overview',
        xaxis_title='Time',
        yaxis_title='Power (MW)',
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top",
                    y=1, xanchor="left", x=1.02),
        height=500
    )

    return fig


def create_exchange_chart(df, combine_zones):
    """Create exchange line chart"""
    exchange_cols = {
        'ExchangeGreatBelt': 'Great Belt (DK1-DK2)',
        'ExchangeGermany': 'Germany',
        'ExchangeSweden': 'Sweden',
        'ExchangeNorway': 'Norway',
        'ExchangeNetherlands': 'Netherlands',
        'ExchangeGreatBritain': 'Great Britain'
    }

    fig = go.Figure()

    if combine_zones:
        # For Great Belt, we just take the value (it's the same for both zones)
        # For international, we sum DK1 and DK2
        agg_dict = {
            'ExchangeGreatBelt': 'first',  # Same value for both zones
            'ExchangeGermany': 'sum',
            'ExchangeSweden': 'sum',
            'ExchangeNorway': 'sum',
            'ExchangeNetherlands': 'sum'
        }

        # Add Great Britain if it exists
        if 'ExchangeGreatBritain' in df.columns:
            agg_dict['ExchangeGreatBritain'] = 'sum'

        combined_df = df.groupby('TimeUTC').agg(agg_dict).reset_index()

        for col, label in exchange_cols.items():
            if col in combined_df.columns:
                fig.add_trace(go.Scatter(
                    x=combined_df['TimeUTC'],
                    y=combined_df[col],
                    mode='lines',
                    name=label,
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Exchange: %{y:.2f} MW<br>' +
                                  '<extra></extra>'
                ))
    else:
        # Separate DK1 and DK2
        for zone in ['DK1', 'DK2']:
            zone_df = df[df['DKZone'] == zone]
            for col, label in exchange_cols.items():
                if col in zone_df.columns:
                    fig.add_trace(go.Scatter(
                        x=zone_df['TimeUTC'],
                        y=zone_df[col],
                        mode='lines',
                        name=f'{label} ({zone})',
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                      'Time: %{x}<br>' +
                                      'Exchange: %{y:.2f} MW<br>' +
                                      '<extra></extra>'
                    ))

    fig.update_layout(
        title='Energy Exchange',
        xaxis_title='Time',
        yaxis_title='Exchange (MW, positive = import)',
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top",
                    y=1, xanchor="left", x=1.02),
        height=500
    )

    return fig


def create_price_chart(df):
    """Create price line chart"""
    price_cols = {
        'DKPrice': 'Denmark',
        'DEPrice': 'Germany',
        'NO2Price': 'Norway (NO2)',
        'SE3Price': 'Sweden (SE3)',
        'SE4Price': 'Sweden (SE4)',
        'NLPrice': 'Netherlands'
    }

    fig = go.Figure()

    # For DK prices, we need to handle DK1 and DK2 separately
    for zone in ['DK1', 'DK2']:
        zone_df = df[df['DKZone'] == zone]
        if 'DKPrice' in zone_df.columns:
            fig.add_trace(go.Scatter(
                x=zone_df['TimeUTC'],
                y=zone_df['DKPrice'],
                mode='lines',
                name=f'Denmark ({zone})',
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Time: %{x}<br>' +
                              'Price: %{y:.2f} DKK/MWh<br>' +
                              '<extra></extra>'
            ))

    # For foreign prices, use DK1 data (they're the same across zones)
    dk1_df = df[df['DKZone'] == 'DK1']
    for col, label in price_cols.items():
        if col != 'DKPrice' and col in dk1_df.columns:
            fig.add_trace(go.Scatter(
                x=dk1_df['TimeUTC'],
                y=dk1_df[col],
                mode='lines',
                name=label,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Time: %{x}<br>' +
                              'Price: %{y:.2f} DKK/MWh<br>' +
                              '<extra></extra>'
            ))

    fig.update_layout(
        title='Electricity Prices',
        xaxis_title='Time',
        yaxis_title='Price (DKK/MWh)',
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top",
                    y=1, xanchor="left", x=1.02),
        height=500
    )

    return fig


def create_weather_chart(df, combine_zones):
    """Create combined weather line chart for wind speed and radiation with dual y-axes"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if combine_zones:
        # Average across zones
        combined_df = df.groupby('TimeUTC')[
            ['WindSpeed', 'Radiation']].mean().reset_index()

        if 'WindSpeed' in combined_df.columns:
            fig.add_trace(go.Scatter(
                x=combined_df['TimeUTC'],
                y=combined_df['WindSpeed'],
                mode='lines',
                name='Wind Speed (Combined)',
                line=dict(color='blue'),
                hovertemplate='<b>Wind Speed</b><br>' +
                              'Time: %{x}<br>' +
                              'Speed: %{y:.2f} m/s<br>' +
                              '<extra></extra>'
            ), secondary_y=False)

        if 'Radiation' in combined_df.columns:
            fig.add_trace(go.Scatter(
                x=combined_df['TimeUTC'],
                y=combined_df['Radiation'],
                mode='lines',
                name='Radiation (Combined)',
                line=dict(color='orange'),
                hovertemplate='<b>Solar Radiation</b><br>' +
                              'Time: %{x}<br>' +
                              'Radiation: %{y:.2f} W/m²<br>' +
                              '<extra></extra>'
            ), secondary_y=True)
    else:
        # Separate DK1 and DK2
        wind_colors = {'DK1': 'blue', 'DK2': 'lightblue'}
        rad_colors = {'DK1': 'orange', 'DK2': 'yellow'}

        for zone in ['DK1', 'DK2']:
            zone_df = df[df['DKZone'] == zone]

            if 'WindSpeed' in zone_df.columns:
                fig.add_trace(go.Scatter(
                    x=zone_df['TimeUTC'],
                    y=zone_df['WindSpeed'],
                    mode='lines',
                    name=f'Wind Speed ({zone})',
                    line=dict(color=wind_colors[zone]),
                    hovertemplate=f'<b>Wind Speed ({zone})</b><br>' +
                    'Time: %{x}<br>' +
                    'Speed: %{y:.2f} m/s<br>' +
                    '<extra></extra>'
                ), secondary_y=False)

            if 'Radiation' in zone_df.columns:
                fig.add_trace(go.Scatter(
                    x=zone_df['TimeUTC'],
                    y=zone_df['Radiation'],
                    mode='lines',
                    name=f'Radiation ({zone})',
                    line=dict(color=rad_colors[zone]),
                    hovertemplate=f'<b>Solar Radiation ({zone})</b><br>' +
                    'Time: %{x}<br>' +
                    'Radiation: %{y:.2f} W/m²<br>' +
                    '<extra></extra>'
                ), secondary_y=True)

    # Set axis titles
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Wind Speed (m/s)", secondary_y=False)
    fig.update_yaxes(title_text="Solar Radiation (W/m²)", secondary_y=True)

    fig.update_layout(
        title='Weather Data (Wind Speed & Solar Radiation)',
        hovermode='x unified',
        height=500,
        showlegend=True
    )

    return fig


def create_co2_chart(df, combine_zones):
    """Create CO2 per kWh line chart"""
    fig = go.Figure()

    # Note: CO2perkWh column not found in data
    # This is a placeholder - needs to be calculated or added to the dataset
    if 'CO2perkWh' in df.columns:
        if combine_zones:
            combined_df = df.groupby('TimeUTC')[
                'CO2perkWh'].mean().reset_index()
            fig.add_trace(go.Scatter(
                x=combined_df['TimeUTC'],
                y=combined_df['CO2perkWh'],
                mode='lines',
                name='CO2 per kWh (Combined)',
                hovertemplate='<b>CO2 per kWh</b><br>' +
                              'Time: %{x}<br>' +
                              'CO2: %{y:.2f} g/kWh<br>' +
                              '<extra></extra>'
            ))
        else:
            for zone in ['DK1', 'DK2']:
                zone_df = df[df['DKZone'] == zone]
                fig.add_trace(go.Scatter(
                    x=zone_df['TimeUTC'],
                    y=zone_df['CO2perkWh'],
                    mode='lines',
                    name=f'CO2 per kWh ({zone})',
                    hovertemplate=f'<b>CO2 per kWh ({zone})</b><br>' +
                    'Time: %{x}<br>' +
                    'CO2: %{y:.2f} g/kWh<br>' +
                    '<extra></extra>'
                ))
    else:
        # Show placeholder message
        fig.add_annotation(
            text="CO2 per kWh data not available in dataset<br>" +
                 "This metric needs to be calculated or added",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )

    fig.update_layout(
        title='CO2 Emissions per kWh',
        xaxis_title='Time',
        yaxis_title='CO2 (g/kWh)',
        hovermode='x unified',
        height=500
    )

    return fig


# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load and preprocess data
print("Loading data...")
df_raw = load_data()
CACHED_DATA = preprocess_resolutions(df_raw)

# Get date range for the date picker
min_date = df_raw['TimeUTC'].min().date()
max_date = df_raw['TimeUTC'].max().date()

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Energy Data Visualization Dashboard",
                    className="text-center my-4"),
            html.Hr()
        ])
    ]),

    # Control panel
    dbc.Row([
        dbc.Col([
            html.Label("Resolution:", className="fw-bold"),
            dcc.Dropdown(
                id='resolution-dropdown',
                options=[
                    {'label': 'Hourly', 'value': 'hourly'},
                    {'label': 'Daily', 'value': 'daily'},
                    {'label': 'Weekly', 'value': 'weekly'},
                    {'label': 'Monthly', 'value': 'monthly'}
                ],
                value='daily',
                clearable=False
            )
        ], width=3),

        dbc.Col([
            html.Label("Start Date:", className="fw-bold"),
            dcc.DatePickerSingle(
                id='start-date-picker',
                date=min_date,
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                display_format='YYYY-MM-DD'
            )
        ], width=2),

        dbc.Col([
            html.Label("End Date:", className="fw-bold"),
            dcc.DatePickerSingle(
                id='end-date-picker',
                date=max_date,
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                display_format='YYYY-MM-DD'
            )
        ], width=2),

        dbc.Col([
            html.Label("Zone View:", className="fw-bold"),
            dcc.RadioItems(
                id='zone-toggle',
                options=[
                    {'label': ' Combined', 'value': 'combined'},
                    {'label': ' Separate (DK1/DK2)', 'value': 'separate'}
                ],
                value='combined',
                inline=True
            )
        ], width=3),

        dbc.Col([
            html.Div([
                html.Label("", className="fw-bold"),
                dbc.Button("Update Charts", id="update-button",
                           color="primary", className="w-100 mt-4")
            ])
        ], width=2)
    ], className="mb-4 p-3 bg-light rounded"),

    # Charts
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-overview",
                type="default",
                children=dcc.Graph(id='overview-chart')
            )
        ], width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-production",
                type="default",
                children=dcc.Graph(id='production-chart')
            )
        ], width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-exchange",
                type="default",
                children=dcc.Graph(id='exchange-chart')
            )
        ], width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-price",
                type="default",
                children=dcc.Graph(id='price-chart')
            )
        ], width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-weather",
                type="default",
                children=dcc.Graph(id='weather-chart')
            )
        ], width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-co2",
                type="default",
                children=dcc.Graph(id='co2-chart')
            )
        ], width=12)
    ], className="mb-4"),

    html.Footer([
        html.Hr(),
        html.P(f"Data range: {min_date} to {max_date} | Total records: {len(df_raw):,}",
               className="text-center text-muted")
    ])
], fluid=True)


# Callback to update all charts
@app.callback(
    [Output('overview-chart', 'figure'),
     Output('production-chart', 'figure'),
     Output('exchange-chart', 'figure'),
     Output('price-chart', 'figure'),
     Output('weather-chart', 'figure'),
     Output('co2-chart', 'figure')],
    [Input('update-button', 'n_clicks')],
    [State('resolution-dropdown', 'value'),
     State('start-date-picker', 'date'),
     State('end-date-picker', 'date'),
     State('zone-toggle', 'value')]
)
def update_charts(n_clicks, resolution, start_date, end_date, zone_view):
    """Update all charts based on selected filters"""

    # Get the appropriate resolution data from cache
    df = CACHED_DATA[resolution].copy()

    # Convert dates
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter by date range
    df_filtered = filter_data(df, start_date, end_date)

    # Determine if zones should be combined
    combine_zones = (zone_view == 'combined')

    # Create all charts
    overview_fig = create_overview_chart(df_filtered, combine_zones)
    production_fig = create_production_chart(df_filtered, combine_zones)
    exchange_fig = create_exchange_chart(df_filtered, combine_zones)
    price_fig = create_price_chart(df_filtered)
    weather_fig = create_weather_chart(df_filtered, combine_zones)
    co2_fig = create_co2_chart(df_filtered, combine_zones)

    return overview_fig, production_fig, exchange_fig, price_fig, weather_fig, co2_fig


# Run the app
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Energy Data Visualization Dashboard...")
    print("="*60)
    print(f"Data loaded: {len(df_raw):,} records")
    print(f"Date range: {min_date} to {max_date}")
    print("="*60 + "\n")

    app.run(debug=True, port=8050, use_reloader=False)
