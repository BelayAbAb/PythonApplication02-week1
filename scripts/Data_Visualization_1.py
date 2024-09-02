import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import yfinance as yf

# Define directories
input_directory = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\yfinance_data'
fns_pid_directory = r'C:\Users\User\Desktop\10Acadamy\Week 1\FNSPID_data'
output_directory = r'C:\Users\User\Desktop\10Acadamy\Week 1\plots\Stock'
combined_csv_file = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\combined_data.csv'
metrics_csv_file = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\financial_metrics.csv'

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Function to process CSV files
def process_file(file):
    df = pd.read_csv(file)
   
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
   
    # Convert all columns except 'Date' to numeric
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
   
    # Add a column for the source file name
    df['Source File'] = os.path.basename(file).replace('.csv', '')
   
    return df

# Combine all CSV files from the input and FNSPID directories
dfs = []
csv_files = glob.glob(os.path.join(input_directory, '*.csv'))
for file in csv_files:
    dfs.append(process_file(file))

# Process FNSPID data
fns_pid_files = glob.glob(os.path.join(fns_pid_directory, '*.csv'))
for file in fns_pid_files:
    dfs.append(process_file(file))

# Combine all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Sort by Date
combined_df.sort_values('Date', inplace=True)

# Reset index
combined_df.reset_index(drop=True, inplace=True)

# Save the combined DataFrame to CSV
combined_df.to_csv(combined_csv_file, index=False)
print(f"Combined CSV saved to {combined_csv_file}")

# Function to calculate financial metrics using yfinance
def calculate_yf_metrics(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
   
    # Extract financial metrics
    try:
        pe_ratio = info.get('forwardEps', None) / info.get('currentPrice', 1)
        pb_ratio = info.get('priceToBook', None)
        market_cap = info.get('marketCap', None)
       
        return {
            'PE Ratio': pe_ratio,
            'PB Ratio': pb_ratio,
            'Market Cap': market_cap
        }
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return {
            'PE Ratio': 'Data Required',
            'PB Ratio': 'Data Required',
            'Market Cap': 'Data Required'
        }

# Function to calculate basic financial metrics
def calculate_basic_metrics(df):
    metrics = {}
   
    if 'Dividends' in df.columns and 'Close' in df.columns:
        total_dividends = df['Dividends'].sum()
        average_close_price = df['Close'].mean()
        if average_close_price > 0:
            metrics['Dividend Yield'] = (total_dividends / average_close_price) * 100
        else:
            metrics['Dividend Yield'] = 0
    else:
        metrics['Dividend Yield'] = 'Data Required'
   
    return metrics

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    df = df.copy()  # Avoid modifying the original DataFrame
   
    # Calculate Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
   
    # Calculate Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
   
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
   
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
   
    # Calculate Bollinger Bands
    df['BB20_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB20_Std'] = df['Close'].rolling(window=20).std()
    df['BB20_Upper'] = df['BB20_Mid'] + (df['BB20_Std'] * 2)
    df['BB20_Lower'] = df['BB20_Mid'] - (df['BB20_Std'] * 2)
   
    return df

# Function to plot technical indicators
def plot_technical_indicators(df, output_dir, symbol):
    plt.figure(figsize=(14, 10))
   
    # Plot closing price and moving averages
    plt.subplot(3, 1, 1)
    plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    plt.plot(df['Date'], df['MA20'], label='20-Day MA', color='orange')
    plt.plot(df['Date'], df['MA50'], label='50-Day MA', color='red')
    plt.title(f'{symbol} - Closing Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
   
    # Plot RSI
    plt.subplot(3, 1, 2)
    plt.plot(df['Date'], df['RSI'], label='RSI', color='green')
    plt.axhline(70, linestyle='--', color='red', alpha=0.5)
    plt.axhline(30, linestyle='--', color='blue', alpha=0.5)
    plt.title(f'{symbol} - Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
   
    # Plot Bollinger Bands
    plt.subplot(3, 1, 3)
    plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    plt.plot(df['Date'], df['BB20_Upper'], label='Upper Bollinger Band', color='red')
    plt.plot(df['Date'], df['BB20_Lower'], label='Lower Bollinger Band', color='green')
    plt.title(f'{symbol} - Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
   
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{symbol}_technical_indicators.png'))
    plt.close()

# Calculate metrics and plot indicators for each unique stock symbol or file
metrics_list = []

for file in csv_files + fns_pid_files:
    symbol = os.path.basename(file).replace('.csv', '')
    print(f"Processing stock symbol: {symbol}")
    symbol_df = pd.read_csv(file)
   
    # Convert numeric columns to appropriate types
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']
    symbol_df[numeric_columns] = symbol_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
   
    # Calculate basic metrics for the current file
    basic_metrics = calculate_basic_metrics(symbol_df)
   
    # Fetch additional metrics using yfinance
    yf_metrics = calculate_yf_metrics(symbol)
   
    # Append metrics to the list
    metrics_list.append({
        'Stock Symbol': symbol,
        'PE Ratio': yf_metrics['PE Ratio'],
        'PB Ratio': yf_metrics['PB Ratio'],
        'Dividend Yield': basic_metrics['Dividend Yield'],
        'Market Cap': yf_metrics['Market Cap']
    })
   
    # Calculate technical indicators
    symbol_df = calculate_technical_indicators(symbol_df)
   
    # Plot technical indicators
    plot_technical_indicators(symbol_df, output_directory, symbol)

# Create a DataFrame from the metrics list
metrics_df = pd.DataFrame(metrics_list)

# Save the financial metrics DataFrame to CSV with error handling
try:
    metrics_df.to_csv(metrics_csv_file, index=False)
    print(f"Financial metrics CSV saved to {metrics_csv_file}")
except PermissionError as e:
    print(f"PermissionError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

print(f"Technical indicators plots saved to {output_directory}")
