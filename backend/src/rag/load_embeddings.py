import os
import pickle
import chromadb

# Optional Streamlit support for server caching
try:
    import streamlit as st
except ImportError:
    st = None

def cache_resource_fallback(func):
    """Fallback decorator if streamlit is not present."""
    if st is not None and hasattr(st, "cache_resource"):
        return st.cache_resource()(func)
    return func

@cache_resource_fallback
def load_chroma_from_embeddings():
    """
    Load precomputed embeddings from pickle and initialize an ephemeral Chroma DB.
    This replaces runtime sentence-transformers calculations.
    """
    print("Loading precomputed embeddings and initializing Ephemeral Chroma DB...")
    
    pkl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models/embeddings.pkl")
    
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found. Please run 'python scripts/build_embeddings.py' first.")
        return None
        
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
        
    texts = data["texts"]
    embeddings = data["embeddings"]
    labels = data["labels"]
    
    # Initialize ephemeral Chroma client
    client = chromadb.EphemeralClient()
    collection_name = "isot_news"
    
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
        
    collection = client.create_collection(name=collection_name)
    
    # Format required insertion fields
    ids = [str(i) for i in range(len(texts))]
    metadatas = [{"label": label, "source": "ISOT Fake News Dataset"} for label in labels]
    
    # Chunk insertion logic
    batch_size = 1000
    for i in range(0, len(texts), batch_size):
        end = i + batch_size
        collection.add(
            documents=texts[i:end],
            embeddings=embeddings[i:end].tolist() if hasattr(embeddings, "tolist") else embeddings[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
        
    print(f"Successfully loaded {len(texts)} embeddings into production DB!")
    return collection

# Global variable fallback caching for pipeline continuity outside of Streamlit
_COLLECTION = None

def get_collection():
    global _COLLECTION
    if _COLLECTION is None:
        # Under streamlit, @cache_resource_fallback caches this function execution safely
        _COLLECTION = load_chroma_from_embeddings()
    return _COLLECTION
