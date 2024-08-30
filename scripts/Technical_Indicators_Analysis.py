import pandas as pd
import glob
import os
import talib as ta
import matplotlib.pyplot as plt

# Define directories
input_directory = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\yfinance_data'
output_directory = r'C:\Users\User\Desktop\10Acadamy\Week 1\plots\Stock'
combined_csv_file = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\combined_data.csv'

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Get list of all CSV files in the input directory
csv_files = glob.glob(os.path.join(input_directory, '*.csv'))

# Combine all CSV files into one DataFrame and add a column for the source file name
dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    df['Source File'] = os.path.basename(file).replace('.csv', '')  # Extract file name without extension
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# Ensure the Date column is in datetime format
combined_df['Date'] = pd.to_datetime(combined_df['Date'])

# Sort by Date
combined_df.sort_values('Date', inplace=True)

# Reset index
combined_df.reset_index(drop=True, inplace=True)

# Save the combined DataFrame to CSV
combined_df.to_csv(combined_csv_file, index=False)

print(f"Combined CSV saved to {combined_csv_file}")

# Calculate technical indicators using TA-Lib
def calculate_technical_indicators(df):
    df['SMA_20'] = ta.SMA(df['Close'], timeperiod=20)  # 20-period Simple Moving Average
    df['SMA_50'] = ta.SMA(df['Close'], timeperiod=50)  # 50-period Simple Moving Average
    df['RSI'] = ta.RSI(df['Close'], timeperiod=14)     # 14-period Relative Strength Index
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)  # MACD

    return df

# Apply the technical indicators to the combined DataFrame
combined_df = calculate_technical_indicators(combined_df)

# Save the DataFrame with technical indicators to CSV
combined_with_indicators_file = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\combined_data_with_indicators.csv'
combined_df.to_csv(combined_with_indicators_file, index=False)

print(f"Combined CSV with indicators saved to {combined_with_indicators_file}")

# Function to plot technical indicators
def plot_technical_indicators(df, output_dir):
    # Plot for Moving Averages
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    plt.plot(df['Date'], df['SMA_20'], label='SMA 20', color='orange')
    plt.plot(df['Date'], df['SMA_50'], label='SMA 50', color='green')
    plt.title('Simple Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'sma_plot.png'))
    plt.close()

    # Plot for RSI
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['RSI'], label='RSI', color='red')
    plt.axhline(70, color='gray', linestyle='--')
    plt.axhline(30, color='gray', linestyle='--')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'rsi_plot.png'))
    plt.close()

    # Plot for MACD
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['MACD'], label='MACD', color='blue')
    plt.plot(df['Date'], df['MACD_signal'], label='MACD Signal', color='red')
    plt.bar(df['Date'], df['MACD_hist'], label='MACD Histogram', color='gray', alpha=0.3)
    plt.title('MACD')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'macd_plot.png'))
    plt.close()

# Plot technical indicators and save to PNG files
plot_technical_indicators(combined_df, output_directory)

print(f"Technical indicators plots saved to {output_directory}")