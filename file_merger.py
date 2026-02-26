import pandas as pd
import numpy as np

# List of your files
files = {
    "Kitui": "TAMSAT & Open Meteo daily data for 10 yrs for Kitui (soil moisture, rainfall, Av temp& Av humidity).csv",
    "Syokimau": "TAMSAT & Open Meteo daily data for 10 yrs for Syokimau (soil moisture, rainfall, Av temp& Av humidity).csv",
    "Turkana": "TAMSAT & Open Meteo daily data for 10 yrs for Turkana (soil moisture, rainfall, Av temp& Av humidity).csv"
}

def prepare_data(city, file_path):
    df = pd.read_csv(file_path)
    
    # 1. Create Lagged Feature: Previous day's moisture (t-1)
    df['prev_moisture'] = df['sm_c4grass'].shift(1)
    
    # 2. Convert date to millisecond timestamps (1 day = 86,400,000 ms)
    df['timestamp'] = np.arange(len(df)) * 86400000
    
    # 3. Clean: Remove the first row (which has a NaN in prev_moisture)
    df = df.dropna().reset_index(drop=True)
    
    # 4. Select only relevant columns for the ML model
    # Inputs: Temp, Humidity, Rainfall, Previous Moisture. Output: Current Moisture
    final_df = df[['timestamp', 'temperature_2m_mean (°C)', 'relative_humidity_2m_mean (%)', 'rfe_filled', 'prev_moisture', 'sm_c4grass']]
    
    # Rename columns to be Edge Impulse friendly
    final_df.columns = ['timestamp', 'temp', 'hum', 'rain', 'prev_sm', 'target_sm']
    
    # Save as separate files to upload to Edge Impulse
    final_df.to_csv(f"{city}_prepared.csv", index=False)
    print(f"Prepared {city} data with {len(final_df)} samples.")

for city, path in files.items():
    prepare_data(city, path)