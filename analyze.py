# analyze.py
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pickle

def run_topic_modeling(input_pkl="data/processed_reviews.pkl", output_pkl="data/analyzed_reviews.pkl"):
    df = pd.read_pickle(input_pkl)
    docs = df["clean_text"].tolist()

    # Load sentence embeddings
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(docs, show_progress_bar=True, batch_size=512)

    # Initialize and fit BERTopic
    topic_model = BERTopic(language="english",
                           calculate_probabilities=True,
                           verbose=True)
    topics, probs = topic_model.fit_transform(docs, embeddings)

    # Collect topic info
    info = topic_model.get_topic_info()
    print("Top topics:\n", info.head())

    # Attach to dataframe
    df["topic"] = topics
    df["probability"] = [prob.max() for prob in probs]

    # Compute average sentiment per topic
    topic_stats = df.groupby("topic")["sentiment"].agg(["count","mean"]).reset_index()
    topic_stats.columns = ["topic","num_reviews","avg_sentiment"]

    # Merge stats
    df = df.merge(topic_stats, on="topic", how="left")

    # Save everything
    with open(output_pkl, "wb") as f:
        pickle.dump({"df": df, "model": topic_model}, f)
    print(f"Saved analyzed data with {len(df)} reviews.")

if __name__ == "__main__":
    run_topic_modeling()