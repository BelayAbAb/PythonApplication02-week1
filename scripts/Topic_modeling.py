import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import matplotlib.pyplot as plt

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Path to the directory containing CSV files
directory_path = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\Publisher_Data'

# Initialize an empty DataFrame
all_data = pd.DataFrame()

# Iterate over all CSV files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)
        if 'headline' in df.columns:
            all_data = pd.concat([all_data, df[['headline']]], ignore_index=True)

# Check if we have loaded any data
if all_data.empty:
    print("No CSV files with 'headline' column found.")
else:
    # Preprocess text
    all_data['processed_headline'] = all_data['headline'].apply(preprocess_text)

    # Vectorize text
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(all_data['processed_headline'])

    # Perform topic modeling with LDA
    num_topics = 5
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(X)

    # Create a directory to save plots if it doesn't exist
    output_dir = r'C:\Users\User\Desktop\10Acadamy\Week 1\Week-1\Publisher_Data\Topic_Visualizations'
    os.makedirs(output_dir, exist_ok=True)

    # Display topics and save plots
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        plt.figure(figsize=(10, 6))
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        top_values = [topic[i] for i in topic.argsort()[:-11:-1]]

        plt.barh(top_words, top_values, color='skyblue')
        plt.xlabel('Importance')
        plt.title(f'Topic #{topic_idx + 1}')
        plt.gca().invert_yaxis()  # To have the highest values at the top

        # Save the plot as a JPG file
        plt.savefig(os.path.join(output_dir, f'topic_{topic_idx + 1}.jpg'))
        plt.close()

    print(f"Topic visualizations saved to {output_dir}")
