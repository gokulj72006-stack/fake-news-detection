import pandas as pd

def load_spam_data():
    df = pd.read_csv("../data/spam.csv", encoding="latin-1")
    df = df[["v1", "v2"]]  # drop unnamed junk columns
    df.columns = ["label", "text"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    return df

if __name__ == "__main__":
    df = load_spam_data()
    print(df.shape)
    print(df.head())
    print(df["label"].value_counts())