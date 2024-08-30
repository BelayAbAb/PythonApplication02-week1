import pandas as pd
import matplotlib.pyplot as plt
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download required NLTK resources (if not already installed)
try:
    nltk.data.find('corpora/stopwords.zip')
except:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt.zip')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet.zip')
except:
    nltk.download('wordnet')

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except:
    nltk.download('vader_lexicon')

# Initialize the lemmatizer, stop words, and VADER sentiment analyzer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
vader_analyzer = SentimentIntensityAnalyzer()

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
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

# Apply preprocessing to the 'headline' column
df['processed_headline'] = df['headline'].apply(preprocess_text)

# Analyze sentiment using TextBlob
def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Analyze sentiment using VADER
def analyze_sentiment_vader(text):
    scores = vader_analyzer.polarity_scores(text)
    return scores['compound'], scores['compound']

# Initialize a list to store sentiment data
sentiment_data = []

# Iterate over the 'processed_headline' column and analyze sentiment
for _, row in df.iterrows():
    text = row['processed_headline']
    polarity_textblob, subjectivity_textblob = analyze_sentiment_textblob(text)
    polarity_vader, _ = analyze_sentiment_vader(text)
    
    sentiment_data.append({
        "headline": row['headline'],
        "processed_headline": text,
        "polarity_textblob": polarity_textblob,
        "subjectivity_textblob": subjectivity_textblob,
        "polarity_vader": polarity_vader
    })

# Convert sentiment data to a DataFrame
sentiment_df = pd.DataFrame(sentiment_data)

# Debugging: Print the first few rows of the sentiment DataFrame
print("Sentiment DataFrame:")
print(sentiment_df.head())

# Check if the DataFrame contains the required columns
required_columns = {'polarity_textblob', 'subjectivity_textblob', 'polarity_vader'}
if not required_columns.issubset(sentiment_df.columns):
    raise ValueError("Sentiment DataFrame does not contain required columns.")

# Specify the path for saving plots
output_directory = r'C:\Users\User\Desktop\10Acadamy\Merged_data'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Plot sentiment analysis results
def plot_sentiment(df, output_directory):
    # Plot TextBlob Polarity
    plt.figure(figsize=(10, 6))
    plt.hist(df['polarity_textblob'], bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of TextBlob Polarity')
    plt.xlabel('Polarity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_directory, 'textblob_polarity_distribution.jpg'), format='jpg')
    plt.close()

    # Plot TextBlob Subjectivity
    plt.figure(figsize=(10, 6))
    plt.hist(df['subjectivity_textblob'], bins=30, color='lightcoral', edgecolor='black')
    plt.title('Distribution of TextBlob Subjectivity')
    plt.xlabel('Subjectivity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_directory, 'textblob_subjectivity_distribution.jpg'), format='jpg')
    plt.close()

    # Plot VADER Polarity
    plt.figure(figsize=(10, 6))
    plt.hist(df['polarity_vader'], bins=30, color='lightgreen', edgecolor='black')
    plt.title('Distribution of VADER Polarity')
    plt.xlabel('Polarity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_directory, 'vader_polarity_distribution.jpg'), format='jpg')
    plt.close()

# Call the function to plot the results
plot_sentiment(sentiment_df, output_directory)
