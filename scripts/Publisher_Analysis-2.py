# -*- coding: utf-8 -*-

import pandas as pd
import os
import matplotlib.pyplot as plt
import talib as ta

def load_and_extract_data(directory_path):
    """
    Load CSV files from the given directory, extract relevant columns, and add a column for the filename.
   
    Parameters:
    directory_path (str): The path to the directory containing the CSV files.
   
    Returns:
    DataFrame: A pandas DataFrame containing the extracted data with an additional 'filename' column.
    """
    # List to hold DataFrames
    dataframes = []

    # Define the expected dtype for columns if necessary
    dtype_spec = {
        'headline': str,
        'url': str,
        'publisher': str,
        'date': str,
        'stock': str
    }

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
           
            # Load the CSV file into a DataFrame
            try:
                df = pd.read_csv(file_path, dtype=dtype_spec, low_memory=False)
               
                # Check for required columns
                required_columns = ['headline', 'url', 'publisher', 'date', 'stock']
                if all(column in df.columns for column in required_columns):
                    # Add 'filename' column to DataFrame
                    df['filename'] = filename
                    dataframes.append(df)
                else:
                    print(f"File {filename} does not have the required columns.")
           
            except Exception as e:
                print(f"Error reading {filename}: {e}")
   
    # Check if any DataFrames were collected
    if not dataframes:
        raise ValueError("No valid DataFrames found to concatenate.")

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
   
    return combined_df

def calculate_technical_indicators(df):
    """
    Calculate various technical indicators using TA-Lib.
   
    Parameters:
    df (DataFrame): The DataFrame containing stock data.
   
    Returns:
    DataFrame: The DataFrame with additional columns for technical indicators.
    """
    # Convert 'stock' to numeric, forcing errors to NaN
    df['stock'] = pd.to_numeric(df['stock'], errors='coerce')
   
    # Remove rows where stock price could not be converted
    df = df.dropna(subset=['stock']).copy()
   
    # Calculate Moving Averages (SMA)
    df.loc[:, 'SMA_20'] = ta.SMA(df['stock'], timeperiod=20)
    df.loc[:, 'SMA_50'] = ta.SMA(df['stock'], timeperiod=50)
   
    # Calculate RSI
    df.loc[:, 'RSI'] = ta.RSI(df['stock'], timeperiod=14)
   
    # Calculate MACD
    macd, macd_signal, macd_hist = ta.MACD(df['stock'], fastperiod=12, slowperiod=26, signalperiod=9)
    df.loc[:, 'MACD'] = macd
    df.loc[:, 'MACD_signal'] = macd_signal
    df.loc[:, 'MACD_hist'] = macd_hist
   
    return df

def analyze_and_plot(df, output_directory):
    """
    Perform analyses on the dataset and save results as PNG images grouped by filename.
   
    Parameters:
    df (DataFrame): The DataFrame containing the combined dataset.
    output_directory (str): The directory where PNG files will be saved.
   
    Returns:
    None
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
   
    # Add technical indicators
    df = calculate_technical_indicators(df)
   
    # Iterate through each filename group
    for file_name, group in df.groupby('filename'):
        # Convert 'date' to datetime
        group = group.copy()  # Explicitly create a copy
        group['date'] = pd.to_datetime(group['date'], errors='coerce')
       
        # Skip the file if date conversion fails completely
        if group['date'].isna().all():
            print(f"Skipping {file_name} due to date parsing issues.")
            continue
       
        # Remove rows where date parsing failed
        group = group.dropna(subset=['date'])
       
        # Use .loc to ensure we're modifying a copy
        group.loc[:, 'headline_length'] = group['headline'].apply(len)
       
        # Publication Frequency Over Time
        plt.figure(figsize=(12, 8))
        group['date'].dt.to_period('D').value_counts().sort_index().plot(kind='line')
        plt.title(f'Publication Frequency Over Time - {file_name}')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.grid(True)
        plt.savefig(os.path.join(output_directory, f'publication_frequency_{file_name}.png'))
        plt.close()
       
        # Publication Times Analysis
        group.loc[:, 'time_of_day'] = group['date'].dt.time
        plt.figure(figsize=(12, 8))
        group['time_of_day'].value_counts().sort_index().plot(kind='line')
        plt.title(f'Distribution of Publication Times - {file_name}')
        plt.xlabel('Time of Day')
        plt.ylabel('Number of Articles')
        plt.grid(True)
        plt.savefig(os.path.join(output_directory, f'publication_times_{file_name}.png'))
        plt.close()
       
        # Identify Significant Spikes in Publication Frequency
        plt.figure(figsize=(12, 8))
        daily_counts = group['date'].dt.to_period('D').value_counts().sort_index()
        rolling_avg = daily_counts.rolling(window=7).mean()  # 7-day rolling average
        daily_counts.plot(label='Daily Count', color='blue')
        rolling_avg.plot(label='7-Day Rolling Average', color='red', linestyle='--')
        plt.title(f'Publication Frequency with Rolling Average - {file_name}')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_directory, f'publication_frequency_with_trend_{file_name}.png'))
        plt.close()

        # Publisher Contribution Analysis
        plt.figure(figsize=(12, 8))
        publisher_counts = group['publisher'].value_counts()
        publisher_counts.plot(kind='bar')
        plt.title(f'Publisher Contribution - {file_name}')
        plt.xlabel('Publisher')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=90)
        plt.grid(axis='y')
        plt.savefig(os.path.join(output_directory, f'publisher_contribution_{file_name}.png'))
        plt.close()
       
        # Domain Analysis
        group['publisher_domain'] = group['publisher'].apply(lambda x: x.split('@')[-1] if '@' in x else 'No Domain')
       
        domain_counts = group['publisher_domain'].value_counts()
       
        plt.figure(figsize=(12, 8))
        domain_counts.plot(kind='bar')
        plt.title(f'Publisher Domains - {file_name}')
        plt.xlabel('Domain')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=90)
        plt.grid(axis='y')
        plt.savefig(os.path.join(output_directory, f'publisher_domains_{file_name}.png'))
        plt.close()

        # Technical Indicators Analysis
        # Moving Averages
        plt.figure(figsize=(12, 8))
        plt.plot(group['date'], group['stock'], label='Stock Price', color='blue')
        plt.plot(group['date'], group['SMA_20'], label='SMA 20', color='red')
        plt.plot(group['date'], group['SMA_50'], label='SMA 50', color='green')
        plt.title(f'Moving Averages - {file_name}')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_directory, f'moving_averages_{file_name}.png'))
        plt.close()
       
        # RSI
        plt.figure(figsize=(12, 8))
        plt.plot(group['date'], group['RSI'], label='RSI', color='purple')
        plt.axhline(y=70, color='red', linestyle='--', label='Overbought')
        plt.axhline(y=30, color='green', linestyle='--', label='Oversold')
        plt.title(f'Relative Strength Index (RSI) - {file_name}')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_directory, f'RSI_{file_name}.png'))
        plt.close()
       
        # MACD
        plt.figure(figsize=(12, 8))
        plt.plot(group['date'], group['MACD'], label='MACD', color='blue')
        plt.plot(group['date'], group['MACD_signal'], label='MACD Signal', color='red')
        plt.bar(group['date'], group['MACD_hist'], label='MACD Histogram', color='green')
        plt.title(f'Moving Average Convergence Divergence (MACD) - {file_name}')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_directory, f'MACD_{file_name}.png'))
        plt.close()

def save_combined_data(df, output_file):
    """
    Save the combined DataFrame to a CSV file.
   
    Parameters:
    df (DataFrame): The DataFrame containing the combined data.
    output_file (str): The path to the output CSV file.
   
    Returns:
    None
    """
    df.to_csv(output_file, index=False)

def main(input_directory, output_directory, combined_csv_file):
    """
    Main function to execute the data processing pipeline.
   
    Parameters:
    input_directory (str): The directory containing input CSV files.
    output_directory (str): The directory where PNG files will be saved.
    combined_csv_file (str): The path to the output CSV file for combined data.
   
    Returns:
    None
    """
    df = load_and_extract_data(input_directory)
    save_combined_data(df, combined_csv_file)
    analyze_and_plot(df, output_directory)

if __name__ == '__main__':
    input_directory = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\Publisher_Data'  # Change this to your input directory
    output_directory = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\plots'  # Change this to your output directory
    combined_csv_file = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\combined_data.csv'  # Change this to your output CSV file
    
      main(input_directory, output_directory, combined_csv_file)