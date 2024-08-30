import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

# Define directories
input_directory = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\yfinance_data'
output_directory = r'C:\Users\User\Desktop\10Acadamy\Week 1\plots\Stock'
combined_csv_file = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\combined_data.csv'
metrics_csv_file = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\financial_metrics.csv'

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Combine all CSV files into one DataFrame and add a column for the source file name
dfs = []
csv_files = glob.glob(os.path.join(input_directory, '*.csv'))
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

# Function to calculate basic financial metrics
def calculate_financial_metrics(df):
    # Assuming Dividends and Close Price are available
    metrics = {}
    metrics['Dividend Yield'] = (df['Dividends'].sum() / df['Close'].mean()) * 100  # Simple Dividend Yield
    # P/E Ratio and Market Cap typically require additional data which might not be available in CSV files
    metrics['PE Ratio'] = 'N/A'
    metrics['PB Ratio'] = 'N/A'
    metrics['Market Cap'] = 'N/A'
   
    return metrics

# Calculate metrics for each unique stock symbol or file
metrics_list = []

for file in csv_files:
    symbol = os.path.basename(file).replace('.csv', '')
    print(f"Processing stock symbol: {symbol}")
    symbol_df = pd.read_csv(file)
   
    # Calculate metrics for the current file
    symbol_metrics = calculate_financial_metrics(symbol_df)
   
    # Append metrics to the list
    metrics_list.append({
        'Stock Symbol': symbol,
        'PE Ratio': symbol_metrics['PE Ratio'],
        'PB Ratio': symbol_metrics['PB Ratio'],
        'Dividend Yield': symbol_metrics['Dividend Yield'],
        'Market Cap': symbol_metrics['Market Cap']
    })

# Create a DataFrame from the metrics list
metrics_df = pd.DataFrame(metrics_list)

# Save the financial metrics DataFrame to CSV
metrics_df.to_csv(metrics_csv_file, index=False)

print(f"Financial metrics CSV saved to {metrics_csv_file}")

# Plot financial metrics
def plot_financial_metrics(metrics_df, output_dir):
    # Set up the plot
    plt.figure(figsize=(14, 8))
   
    # Plot each metric
    for metric in ['PE Ratio', 'PB Ratio', 'Dividend Yield', 'Market Cap']:
        plt.figure(figsize=(14, 7))
        # Use 'N/A' as 0 for plotting purposes
        metric_values = metrics_df[metric].replace('N/A', 0)
        plt.bar(metrics_df['Stock Symbol'], metric_values, color='cyan')
        plt.title(f'{metric} for Stocks')
        plt.xlabel('Stock Symbol')
        plt.ylabel(metric)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_plot.png'))
        plt.close()

# Generate and save plots for financial metrics
plot_financial_metrics(metrics_df, output_directory)

print(f"Financial metrics plots saved to {output_directory}")