# extract_topic_phrases.py
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import numpy as np

# Load the analyzed review data
with open("data/analyzed_reviews.pkl", "rb") as f:
    data = pickle.load(f)
    df = data['df']

# Remove noise topics if present
df = df[df['topic'] != -1]

# Use CountVectorizer to extract top n-grams per topic
vectorizer = CountVectorizer(stop_words='english', max_features=10000, ngram_range=(2, 3), analyzer='word')
top_phrases = {}

for topic_num in sorted(df['topic'].unique()):
    top_n_topics = list(range(10))  # Adjust N here
    if topic_num not in top_n_topics:
        continue
    topic_docs = df[df['topic'] == topic_num]['clean_text']
    if topic_docs.empty:
        continue
    try:
        topic_matrix = vectorizer.fit_transform(topic_docs)
        sums = topic_matrix.sum(axis=0)
        data = []
        for col, term in enumerate(vectorizer.get_feature_names_out()):
            data.append((term, sums[0, col]))
        ranking = pd.DataFrame(data, columns=['term', 'frequency'])
        top_terms = ranking.sort_values('frequency', ascending=False).head(10)
        top_phrases[topic_num] = list(top_terms['term'])
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        word_freq = dict(zip(top_terms['term'], top_terms['frequency']))
        wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        plt.figure(figsize=(8, 4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Topic {topic_num}")
        plt.tight_layout()
        plt.savefig(f"visualizations/topic_{topic_num}_wordcloud.png")
        plt.close()

        # Create a bar chart for top terms
        plt.figure(figsize=(10, 5))
        plt.bar(top_terms['term'], top_terms['frequency'], color='skyblue')
        plt.xlabel("Term")
        plt.ylabel("Frequency")
        plt.title(f"Top Terms for Topic {topic_num}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"visualizations/topic_{topic_num}_bar_chart.png")
        plt.close()
    except Exception as e:
        print(f"Error processing topic {topic_num}: {e}")

# Print phrases
print("\nTop Phrases per Topic:")
for topic, phrases in top_phrases.items():
    print(f"\nTopic {topic}:")
    for i, phrase in enumerate(phrases, 1):
        print(f"  {i}. {phrase}")