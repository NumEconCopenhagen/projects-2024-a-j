import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime

def fetch_and_process_data():
    '''
    Fetching data, creating dataframes and exporting as .csv files
    '''
    # Lists to store data
    hour_dk_elspot = []
    price_dkk = []
    hour_dk_prod_cons = []
    gross_cons = []
    wind_offshore_under_100 = []
    wind_offshore_over_100 = []
    wind_onshore_under_50 = []
    wind_onshore_over_50 = []
    solar_0_10 = []
    solar_10_40 = []
    solar_40_over = []
    total_solar = []
    total_wind = []

    # Fetching and processing Elspot prices data
    response_elspot = requests.get(
        'https://api.energidataservice.dk/dataset/Elspotprices?start=2023-01-01&end=2023-12-31&filter={"PriceArea":["DK2"]}&limit=0')
    result_elspot = response_elspot.json()

    records_elspot = result_elspot.get('records', [])
    for record in records_elspot:
        hour_dk_elspot.append(record['HourDK'])
        price_dkk.append(record['SpotPriceDKK'])

    # Creating dataframe and exporting for Elspot prices
    df_elspot = pd.DataFrame({
        'HourDK': hour_dk_elspot,
        'SpotPriceDKK': price_dkk
    })
    df_elspot.to_csv("elspot_prices.csv", index=False)

    # Fetching and processing Production and Consumption data
    response_prod_cons = requests.get(
        'https://api.energidataservice.dk/dataset/ProductionConsumptionSettlement?start=2023-01-01&end=2023-12-31&filter={"PriceArea":["DK2"]}&limit=0')
    result_prod_cons = response_prod_cons.json()

    records_prod_cons = result_prod_cons.get('records', [])
    for record in records_prod_cons:
        hour_dk_prod_cons.append(record['HourDK'])
        gross_cons.append(record['GrossConsumptionMWh'])
        wind_offshore_under_100.append(record['OffshoreWindLt100MW_MWh'])
        wind_offshore_over_100.append(record['OffshoreWindGe100MW_MWh'])
        wind_onshore_under_50.append(record['OnshoreWindLt50kW_MWh'])
        wind_onshore_over_50.append(record['OnshoreWindGe50kW_MWh'])
        solar_0_10.append(record['SolarPowerLt10kW_MWh'])
        solar_10_40.append(record['SolarPowerGe10Lt40kW_MWh'])
        solar_40_over.append(record['SolarPowerGe40kW_MWh'])

    # Summarizing solar and wind production for each datapoint
        sum_solar = record['SolarPowerLt10kW_MWh'] + record['SolarPowerGe10Lt40kW_MWh'] + record['SolarPowerGe40kW_MWh']
        sum_wind = record['OffshoreWindLt100MW_MWh'] + record['OffshoreWindGe100MW_MWh'] + record['OnshoreWindLt50kW_MWh'] + record['OnshoreWindGe50kW_MWh']

        total_solar.append(sum_solar)
        total_wind.append(sum_wind)

    # Creating dataframe and export for Production and Consumption
    df_prod_cons = pd.DataFrame({
        'HourDK': hour_dk_prod_cons,
        'GrossConsumptionMWh': gross_cons,
        'TotalSolarProductionMWh': total_solar,
        'TotalWindProductionMWh': total_wind
    })
    df_prod_cons.to_csv("production_consumption.csv", index=False)


def print_descriptive_stats(merged_df):
    '''
    Printing output for descriptive statistics table
    '''

    # Calculating descriptive statistics
    descriptive_stats = merged_df.describe()
    
    # Styling the descriptive statistics table
    styled_descriptive_stats = descriptive_stats.style.format("{:.2f}").set_table_styles([{
        'selector': 'th',
        'props': [('font-size', '12pt'), ('text-align', 'center'), ('color', 'black')]
    }, {
        'selector': 'td',
        'props': [('text-align', 'center'), ('color', 'black')]
    }, {
        'selector': 'tr:nth-of-type(odd)',
        'props': [('background', '#f5f5f5'), ('color', 'black')]
    }, {
        'selector': 'tr:nth-of-type(even)',
        'props': [('background', 'white')]
    }, {
        'selector': 'tr:hover',
        'props': [('background-color', '#ffff99')]
    }]).set_properties(**{'border': '1.5px solid black'})
    
    return styled_descriptive_stats

def plot_series(elspot_prices_path, production_consumption_path):
    '''
    Printing figures
    '''
    # Loading the data
    elspot_prices_df = pd.read_csv(elspot_prices_path, parse_dates=['HourDK'])
    production_consumption_df = pd.read_csv(production_consumption_path, parse_dates=['HourDK'])

    plt.style.use('fivethirtyeight')

    line_width=0.8

    # Creating subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plotting SpotPriceDKK
    ax1.set_xlabel('Month')
    ax1.set_ylabel('DKK')
    ax1.plot(elspot_prices_df['HourDK'], elspot_prices_df['SpotPriceDKK'], 'r-', linewidth=line_width)
    ax1.set_title('Spot Price (DKK)')

    # Plotting GrossConsumptionMWh
    ax2.set_xlabel('Month')
    ax2.set_ylabel('MWh')
    ax2.plot(production_consumption_df['HourDK'], production_consumption_df['GrossConsumptionMWh'], 'b-', linewidth=line_width)
    ax2.set_title('Gross Consumption (MWh)')

    fig.tight_layout() 
    plt.show()