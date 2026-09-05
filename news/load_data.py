import pandas as pd

def load_news_data():
    fake = pd.read_csv("../data/Fake.csv")
    real = pd.read_csv("../data/True.csv")

    fake["label"] = 0  # fake
    real["label"] = 1  # real

    df = pd.concat([fake, real], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    return df

if __name__ == "__main__":
    df = load_news_data()
    print(df.shape)
    print(df.head())
    print(df["label"].value_counts())