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

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
           
            # Load the CSV file into a DataFrame
            try:
                df = pd.read_csv(file_path)
               
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
    Perform various analyses on the dataset and save results as PNG images.
   
    Parameters:
    df (DataFrame): The DataFrame containing the combined dataset.
    output_directory (str): The directory where PNG files will be saved.
   
    Returns:
    None
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
   
    # Descriptive Statistics for Headline Lengths
    df['headline_length'] = df['headline'].apply(len)
   
    # Plot headline length distribution
    plt.figure(figsize=(10, 6))
    df['headline_length'].hist(bins=50, edgecolor='black')
    plt.title('Distribution of Headline Lengths')
    plt.xlabel('Headline Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_directory, 'headline_lengths.png'))
    plt.close()
   
    # Publisher Activity
    publisher_counts = df['publisher'].value_counts()
   
    # Plot publisher activity
    plt.figure(figsize=(12, 8))
    publisher_counts.plot(kind='bar')
    plt.title('Number of Articles by Publisher')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_directory, 'publisher_activity.png'))
    plt.close()
   
    # Publication Dates Analysis
    try:
        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
       
        # Check for and handle any parsing issues
        if df['date'].isna().any():
            print("Warning: Some dates could not be parsed and were set to NaT.")
            # Optional: Save unparsed dates to a separate file for further inspection
            problematic_dates = df[df['date'].isna()]['date']
            if not problematic_dates.empty:
                problematic_dates.to_csv(os.path.join(output_directory, 'problematic_dates.csv'), index=False)
       
        df['day_of_week'] = df['date'].dt.day_name()
       
        # Plot articles per date
        plt.figure(figsize=(12, 8))
        df['date'].dropna().value_counts().sort_index().plot(kind='line')
        plt.title('Number of Articles per Date')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.grid(True)
        plt.savefig(os.path.join(output_directory, 'articles_per_date.png'))
        plt.close()
       
        # Plot articles per day of the week
        plt.figure(figsize=(10, 6))
        df['day_of_week'].value_counts().sort_index().plot(kind='bar')
        plt.title('Number of Articles per Day of the Week')
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Articles')
        plt.grid(axis='y')
        plt.savefig(os.path.join(output_directory, 'articles_per_day_of_week.png'))
        plt.close()

    except Exception as e:
        print(f"Error processing dates: {e}")

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
    data = load_and_extract_data(directory_path)
   
    # Perform analyses and plot results
    analyze_and_plot(data, output_directory)
   
    # Save the combined DataFrame to a new CSV file in chunks
    save_combined_data(data, output_file)

except ValueError as ve:
    print(ve)
except Exception as e:
    print(f"An error occurred: {e}")