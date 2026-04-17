import sys
import os
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

# Ensure the project root is in sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.text_cleaner import clean_text

# Load dataset
true_df = pd.read_csv("data/raw/True.csv")
fake_df = pd.read_csv("data/raw/Fake.csv")

# Ensure labels are explicitly bounded to data BEFORE sampling
true_df["label"] = "REAL"
fake_df["label"] = "FAKE"

# Combine 
df = pd.concat([true_df, fake_df])

# 🔥 IMPORTANT: use subset (for performance)
df = df.sample(5000, random_state=42)

# Build Cleaned Text Column
df["combined"] = df["title"] + " " + df["text"]
df["cleaned"] = df["combined"].apply(clean_text)

# Prepare text & metadata lists
texts = df["cleaned"].tolist()
labels = df["label"].tolist()

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# Save everything
os.makedirs("models", exist_ok=True)
with open("models/embeddings.pkl", "wb") as f:
    pickle.dump({
        "texts": texts,
        "embeddings": embeddings,
        "labels": labels
    }, f)

print("Embeddings built and saved")