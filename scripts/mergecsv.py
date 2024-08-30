import pandas as pd
import os

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

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):0
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
           
            # Load the CSV file into a DataFrame
            try:
                df = pd.read_csv(file_path)
               
                # Check for required columns
                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']
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

# Specify the directory path
directory_path = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\yfinance_data'

# Extract and combine data
try:
    data = load_and_extract_data(directory_path)
    # Display the first few rows of the combined DataFrame
    print(data.head())
   
    # Optionally, save the combined DataFrame to a new CSV file
    output_file = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\yfinance_data\combined_data.csv'
    data.to_csv(output_file, index=False)

except ValueError as ve:
    print(ve)
except Exception as e:
    print(f"An error occurred: {e}")
