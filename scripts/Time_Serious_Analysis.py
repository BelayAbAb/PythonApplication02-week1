import pandas as pd
import os
import matplotlib.pyplot as plt

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
   
    # Iterate through each filename group
    for file_name, group in df.groupby('filename'):
        # Convert 'date' to datetime
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
        # This is a more complex analysis and can be customized based on specific needs
        # For simplicity, we will plot a rolling average to identify trends
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

        # Additional insights (if needed) can be added here

def save_combined_data(df, output_file):
    """
    Save the combined DataFrame to a CSV file in chunks to handle large files.
   
    Parameters:
    df (DataFrame): The DataFrame to save.
    output_file (str): The path to the output CSV file.
   
    Returns:
    None
    """
    try:
        # Define chunk size
        chunk_size = 10**6  # Number of rows per chunk
        num_chunks = (len(df) // chunk_size) + 1
       
        # Save DataFrame in chunks
        for i in range(num_chunks):
            start_row = i * chunk_size
            end_row = min((i + 1) * chunk_size, len(df))
            df.iloc[start_row:end_row].to_csv(output_file, mode='a', header=(i==0), index=False)
            print(f"Saved chunk {i+1} of {num_chunks} to {output_file}")
           
    except Exception as e:
        print(f"Error saving combined data: {e}")

# Specify the directory path
directory_path = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1'
output_directory = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\plots'
output_file = os.path.join(directory_path, 'combined_data.csv')

# Extract and combine data
try:
    combined_data = load_and_extract_data(directory_path)
   
    # Perform analyses and plot results grouped by filename
    analyze_and_plot(combined_data, output_directory)
   
    # Save the combined DataFrame to a new CSV file in chunks
    save_combined_data(combined_data, output_file)

except ValueError as ve:
    print(ve)
except Exception as e:
    print(f"An error occurred: {e}")