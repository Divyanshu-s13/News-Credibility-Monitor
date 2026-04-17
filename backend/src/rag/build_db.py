import os
import pandas as pd
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.utils.text_cleaner import clean_text

# Path for ChromaDB relative to the project root
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")
COLLECTION_NAME = "isot_news"

def build_database(batch_size=512):
    sqlite_path = os.path.join(CHROMA_DB_DIR, "chroma.sqlite3")
    if os.path.exists(CHROMA_DB_DIR) and os.path.exists(sqlite_path):
        try:
            existing_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            existing_collection = existing_client.get_collection(name=COLLECTION_NAME)
            existing_count = existing_collection.count()

            if existing_count > 1000:
                print(
                    f"Database already exists at {CHROMA_DB_DIR} "
                    f"with {existing_count} documents. Skipping rebuild."
                )
                return

            print(
                f"Existing database found but only {existing_count} documents "
                "detected. Rebuilding..."
            )
        except Exception:
            print("Existing database found but collection check failed. Rebuilding...")

    print("Loading datasets...")
    fake_path = os.path.join("data", "raw", "Fake.csv")
    true_path = os.path.join("data", "raw", "True.csv")
    
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError("Could not find Fake.csv or True.csv in data/raw/")
        
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    fake_df['label'] = 'FAKE'
    true_df['label'] = 'REAL'
    
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    print("Cleaning and combining text...")
    df['combined_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
    df['clean_text'] = df['combined_text'].apply(clean_text)
    
    # Filter out empty text
    df = df[df['clean_text'].str.strip() != '']
    
    valid_docs = len(df)
    print(f"Total valid documents to embed: {valid_docs}")
    
    df['source'] = 'ISOT'
    
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
        
    collection = client.create_collection(name=COLLECTION_NAME)
    
    texts = df['clean_text'].tolist()
    metadatas = df[['label', 'subject', 'source']].to_dict(orient='records')
    ids = [f"doc_{i}" for i in range(valid_docs)]
    
    print(f"Computing embeddings and indexing into ChromaDB (batch size: {batch_size})...")
    for i in tqdm(range(0, valid_docs, batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        
        batch_embeddings = model.encode(batch_texts, batch_size=32, show_progress_bar=True)
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings.tolist(),
            documents=batch_texts,
            metadatas=batch_metadatas
        )
        
    print("Database built and persisted successfully!")

if __name__ == "__main__":
    build_database()
