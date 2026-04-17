import pandas as pd
from src.config.config import FAKE_PATH, TRUE_PATH

def load_and_merge_data():
    fake = pd.read_csv(FAKE_PATH)
    true = pd.read_csv(TRUE_PATH)

    fake["label"] = 1
    true["label"] = 0

    data = pd.concat([fake, true])
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Combine title + text for richer signal (both columns exist in ISOT dataset)
    if "title" in data.columns and "text" in data.columns:
        data["text"] = data["title"].fillna("") + " " + data["text"].fillna("")

    return data