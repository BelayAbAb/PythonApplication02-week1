import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For additional visualizations
import os
import argparse
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.data import find



# Initialize the lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to get all CSV files in a directory
def get_csv_files_from_directory(directory):
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze sentiment and stock returns correlation.")
    parser.add_argument('--stock_dir', default="C:\\Users\\User\\Desktop\\10Acadamy\\Week 1\\Week-1\\yfinance_data", help="Directory containing stock returns CSV files")
    parser.add_argument('--headlines_dir', default="C:\\Users\\User\\Desktop\\10Acadamy\\Week 1\\Week-1\\Publisher_Data", help="Directory containing headlines CSV file")
    return parser.parse_args()

# Parse command-line arguments
args = parse_args()

# Get all stock returns CSV files from the specified directory
stock_returns_files = get_csv_files_from_directory(args.stock_dir)
if not stock_returns_files:
    raise ValueError(f"No stock returns files found in directory: {args.stock_dir}")

# Get the headlines CSV file from the specified directory
headlines_files = get_csv_files_from_directory(args.headlines_dir)
if len(headlines_files) != 1:
    raise ValueError(f"Expected exactly one headlines file in directory: {args.headlines_dir}, found {len(headlines_files)}")
headlines_file = headlines_files[0]

# Load and concatenate all stock returns data
stock_returns_list = [pd.read_csv(file, parse_dates=['Date']) for file in stock_returns_files]
stock_returns = pd.concat(stock_returns_list, ignore_index=True)

# Ensure 'Date' column in stock_returns is in datetime format
stock_returns['Date'] = pd.to_datetime(stock_returns['Date'], errors='coerce')

# Load headlines data and convert 'date' column to datetime
headlines = pd.read_csv(headlines_file, parse_dates=['date'])
headlines['date'] = pd.to_datetime(headlines['date'], errors='coerce')

# Tokenize, remove stop words, and lemmatize text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

# Apply preprocessing to the 'headline' column
headlines['processed_headline'] = headlines['headline'].apply(preprocess_text)

# Analyze sentiment for the processed headlines
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

# Initialize a list to store sentiment data
sentiment_data = []

# Iterate over the 'processed_headline' column and analyze sentiment
for _, row in headlines.iterrows():
    text = row['processed_headline']
    polarity, subjectivity = analyze_sentiment(text)
    sentiment_data.append({
        "date": row['date'].date() if pd.notna(row['date']) else None,  # Extract date part if not NaT
        "polarity": polarity,
        "subjectivity": subjectivity
    })

# Convert sentiment data to a DataFrame
sentiment_df = pd.DataFrame(sentiment_data)

# Ensure 'date' column in sentiment_df is in datetime format
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], errors='coerce')

# Aggregate sentiment scores by date to compute average daily sentiment scores
daily_sentiment = sentiment_df.groupby('date').agg({
    'polarity': 'mean',
    'subjectivity': 'mean'
}).reset_index()

# Ensure stock returns data is aggregated by date and include necessary columns
daily_stock_returns = stock_returns.groupby('Date').agg({
    'Close': 'mean'  # or any other stock return metric you want to use
}).reset_index()
daily_stock_returns.rename(columns={'Date': 'date'}, inplace=True)

# Merge sentiment data with stock returns data on the date
merged_df = pd.merge(daily_stock_returns, daily_sentiment, on='date')

# Compute and plot technical indicators

# Simple Moving Average (SMA)
merged_df['SMA_20'] = merged_df['Close'].rolling(window=20).mean()
merged_df['SMA_50'] = merged_df['Close'].rolling(window=50).mean()

# Relative Strength Index (RSI)
def compute_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

merged_df['RSI'] = compute_rsi(merged_df)

# Bollinger Bands
def compute_bollinger_bands(df, window=20):
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    df['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
    df['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)

compute_bollinger_bands(merged_df)

# Plot stock price with technical indicators
def plot_technical_indicators(df, output_directory):
    plt.figure(figsize=(14, 10))
   
    # Plot stock price and moving averages
    plt.subplot(3, 1, 1)
    plt.plot(df['date'], df['Close'], label='Stock Price', color='black')
    plt.plot(df['date'], df['SMA_20'], label='SMA 20', color='blue')
    plt.plot(df['date'], df['SMA_50'], label='SMA 50', color='red')
    plt.title('Stock Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Plot RSI
    plt.subplot(3, 1, 2)
    plt.plot(df['date'], df['RSI'], label='RSI', color='purple')
    plt.axhline(70, linestyle='--', color='red')
    plt.axhline(30, linestyle='--', color='green')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)

    # Plot Bollinger Bands
    plt.subplot(3, 1, 3)
    plt.plot(df['date'], df['Close'], label='Stock Price', color='black')
    plt.plot(df['date'], df['Bollinger_Upper'], label='Bollinger Upper Band', color='orange')
    plt.plot(df['date'], df['Bollinger_Lower'], label='Bollinger Lower Band', color='orange')
    plt.title('Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Save the plot as a JPG file
    output_file = os.path.join(output_directory, 'technical_indicators.jpg')
    plt.tight_layout()
    plt.savefig(output_file, format='jpg')
    plt.close()

# Specify the output directory for saving the plot
output_directory = 'C:\\Users\\User\\Desktop\\10Acadamy\\Week 1\\plots'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Plot and save technical indicators
plot_technical_indicators(merged_df, output_directory)

