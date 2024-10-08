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
   
    try:
        # Fetch the latest financial data
        info = stock.info
       
        # Example calculations (ensure you have the required data)
        pe_ratio = info.get('forwardEps', 'Data Required') / info.get('currentPrice', 1)
        pb_ratio = info.get('priceToBook', 'Data Required')
        market_cap = info.get('marketCap', 'Data Required')
       
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

# Calculate metrics for each unique stock symbol or file
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

# Function to plot financial metrics
def plot_financial_metrics(metrics_df, output_dir):
    metrics = ['PE Ratio', 'PB Ratio', 'Dividend Yield', 'Market Cap']
   
    for metric in metrics:
        plt.figure(figsize=(14, 7))
       
        # Replace 'Data Required' with NaN for better plotting
        metric_values = pd.to_numeric(metrics_df[metric].replace('Data Required', pd.NA))
        plt.bar(metrics_df['Stock Symbol'], metric_values.fillna(0), color='cyan')
        plt.title(f'{metric} for Stocks')
        plt.xlabel('Stock Symbol')
        plt.ylabel(metric)
        plt.xticks(rotation=90)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_plot.png'))
        plt.close()

# Generate and save plots for financial metrics
plot_financial_metrics(metrics_df, output_directory)

print(f"Financial metrics plots saved to {output_directory}")

def main():
    """
    Main function to run the stock analysis application.

    Presents a menu-driven interface for users to select various stock analysis
    functionalities, including data fetching, plotting, and performance
    visualization. The script runs in a loop until the user chooses to exit.
    """
    while True:
        print("\nStock Data Analysis Menu")
        print("1. Process and Combine CSV Files")
        print("2. Calculate and Save Financial Metrics")
        print("3. Plot Financial Metrics")
        print("0. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            # Already processed in this script; this choice can be used for additional CSV processing if needed
            print("CSV files have been processed and combined.")
        elif choice == "2":
            # This option is integrated in the script above
            print("Financial metrics have been calculated and saved.")
        elif choice == "3":
            # This option is integrated in the script above
            print("Financial metrics plots have been generated and saved.")
        elif choice == "0":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()