import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import os
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

# Specify the path to the dataset
input_file = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\Publisher_Data\raw_analyst_ratings.csv'

# Check if the file exists before attempting to read it
if not os.path.isfile(input_file):
    print("File not found. Listing files in the directory:")
    directory = os.path.dirname(input_file)
    print("Files in directory:", directory)
    for filename in os.listdir(directory):
        print(filename)
    raise FileNotFoundError(f"The file {input_file} does not exist. Please check the path.")

# Read the dataset into a DataFrame
df = pd.read_csv(input_file)

# Check the first few rows to understand the structure
print("Dataset Overview:")
print(df.head())

# Check if the 'headline' column exists
if 'headline' not in df.columns:
    raise ValueError("The expected column 'headline' is not found in the dataset.")

# Tokenize, remove stop words, and lemmatize text
def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

# Apply preprocessing to the 'headline' column
df['processed_headline'] = df['headline'].apply(preprocess_text)

# Analyze sentiment for the processed headlines
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

# Initialize a list to store sentiment data
sentiment_data = []

# Iterate over the 'processed_headline' column and analyze sentiment
for _, row in df.iterrows():
    text = row['processed_headline']
    polarity, subjectivity = analyze_sentiment(text)
    sentiment_data.append({
        "headline": row['headline'],
        "processed_headline": text,
        "polarity": polarity,
        "subjectivity": subjectivity
    })

# Convert sentiment data to a DataFrame
sentiment_df = pd.DataFrame(sentiment_data)

# Debugging: Print the first few rows of the sentiment DataFrame
print("Sentiment DataFrame:")
print(sentiment_df.head())

# Check if the DataFrame contains the required columns
if 'polarity' not in sentiment_df.columns or 'subjectivity' not in sentiment_df.columns:
    raise ValueError("Sentiment DataFrame does not contain required columns.")

# Specify the path for saving plots
output_directory = r'C:\Users\User\Desktop\10Acadamy\Merged_data'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Plot sentiment analysis results
def plot_sentiment(df, output_directory):
    # Plot Polarity
    plt.figure(figsize=(10, 6))
    plt.hist(df['polarity'], bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Polarity')
    plt.xlabel('Polarity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_directory, 'polarity_distribution.jpg'), format='jpg')
    plt.close()

    # Plot Subjectivity
    plt.figure(figsize=(10, 6))
    plt.hist(df['subjectivity'], bins=30, color='lightcoral', edgecolor='black')
    plt.title('Distribution of Subjectivity')
    plt.xlabel('Subjectivity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_directory, 'subjectivity_distribution.jpg'), format='jpg')
    plt.close()

# Plot and save results
plot_sentiment(sentiment_df, output_directory)

# Calculate summary statistics for sentiment scores
summary_stats = {
    "mean_polarity": sentiment_df["polarity"].mean(),
    "mean_subjectivity": sentiment_df["subjectivity"].mean(),
    "min_polarity": sentiment_df["polarity"].min(),
    "max_polarity": sentiment_df["polarity"].max(),
    "min_subjectivity": sentiment_df["subjectivity"].min(),
    "max_subjectivity": sentiment_df["subjectivity"].max()
}

# Print sentiment summary statistics
print("Sentiment Summary Statistics:")
print(f"Mean Polarity: {summary_stats['mean_polarity']:.2f}")
print(f"Mean Subjectivity: {summary_stats['mean_subjectivity']:.2f}")
print(f"Min Polarity: {summary_stats['min_polarity']:.2f}")
print(f"Max Polarity: {summary_stats['max_polarity']:.2f}")
print(f"Min Subjectivity: {summary_stats['min_subjectivity']:.2f}")
print(f"Max Subjectivity: {summary_stats['max_subjectivity']:.2f}")
