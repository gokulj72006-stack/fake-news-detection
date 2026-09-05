import re
import pandas as pd
from load_data import load_news_data

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_news_data():
    df = load_news_data()

    # Combine title + text for richer signal (optional but common)
    df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["clean_text"] = df["full_text"].apply(clean_text)

    return df[["clean_text", "label"]]

if __name__ == "__main__":
    df = preprocess_news_data()
    print(df.shape)
    print(df.head())