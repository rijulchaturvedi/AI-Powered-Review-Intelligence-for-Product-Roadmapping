# inspect_top_topics.py

import pickle
import pandas as pd

# Load analyzed review data
with open("data/analyzed_reviews.pkl", "rb") as f:
    data = pickle.load(f)
    df = data["df"]
    topic_labels = data.get("topic_labels")

print("Columns in df:", df.columns.tolist())  # ← Add this

# Define top topics to inspect
top_topics = [0, 1, 2, 3, 4]

print("\n===== TOPIC INSPECTION START =====\n")
for topic in top_topics:
    print(f"--- Topic {topic} ---")
    
    topic_df = df[df['topic'] == topic]

    sample_reviews = topic_df['Text'].dropna().head(5).tolist()  # or adjust based on column name

    for i, review in enumerate(sample_reviews, 1):
        print(f"{i}. {review[:300]}")

    print("\n")

df[df['topic'].isin(top_topics)][['topic', 'Text']].to_csv("top_topic_reviews.csv", index=False)
print("Saved top topic reviews to: top_topic_reviews.csv")