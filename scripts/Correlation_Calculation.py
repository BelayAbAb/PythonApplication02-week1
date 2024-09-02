import pandas as pd
from textblob import TextBlob
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns  # For heatmap visualization
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.data import find

# Download required NLTK resources (if not already installed)
try:
    find('corpora/stopwords.zip')
except:
    nltk.download('stopwords')

try:
    find('tokenizers/punkt.zip')
except:
    nltk.download('punkt')

try:
    find('corpora/wordnet.zip')
except:
    nltk.download('wordnet')

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

# Merge sentiment data with stock returns data on the date
merged_df = pd.merge(stock_returns, sentiment_df, left_on='Date', right_on='date')

# Compute correlation matrix
correlation_matrix = merged_df[['polarity', 'subjectivity', 'Close']].corr()

# Plot and save the correlation matrix
def plot_correlation_matrix(correlation_matrix, output_directory):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, linewidths=0.5)
    plt.title('Correlation Matrix between Sentiment Scores and Stock Returns')
   
    # Save the plot as a JPG file
    output_file = os.path.join(output_directory, 'correlation_matrix.jpg')
    plt.savefig(output_file, format='jpg')
    plt.close()

# Specify the output directory for saving the plot
output_directory = 'C:\\Users\\User\\Desktop\\10Acadamy\\Week 1\\plots'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Plot and save the correlation matrix
plot_correlation_matrix(correlation_matrix, output_directory)

# Output the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)
