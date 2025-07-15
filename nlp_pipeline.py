import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
analyzer = SentimentIntensityAnalyzer()

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def load_and_process(path="data/Reviews.csv", n_rows=None):
    df = pd.read_csv(path, nrows=n_rows)
    df = df.dropna(subset=["Text"])
    if n_rows:
        df = df.sample(n=min(len(df), n_rows), random_state=42)
    df["clean_text"] = df["Text"].apply(preprocess)
    df["sentiment"] = df["clean_text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    print(df[["clean_text", "sentiment"]].head())
    df.to_pickle("data/processed_reviews.pkl")
    return df

if __name__ == "__main__":
    df = load_and_process(n_rows=20000)