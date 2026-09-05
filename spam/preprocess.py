import re
from load_data import load_spam_data

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_spam_data():
    df = load_spam_data()
    df["clean_text"] = df["text"].apply(clean_text)

    return df[["clean_text", "label"]]

if __name__ == "__main__":
    df = preprocess_spam_data()
    print(df.shape)
    print(df.head())