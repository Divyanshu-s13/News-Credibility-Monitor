import os
import sys

# Support direct script invocation
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.text_cleaner import clean_text
from src.rag.load_embeddings import get_collection

def retrieve_similar_news(query, k=3):
    """
    Accepts text input, pulls the precomputed Chroma collection,
    and returns the top-k similar documents utilizing Chroma's lightweight
    default ONNX embedding logic against our cleaned query.
    """
    
    query_clean = clean_text(query)
    if not query_clean.strip():
        # Fallback to raw if cleaner stripped it completely
        query_clean = query
        
    collection = get_collection()
    
    if collection is None:
        print("Error: Could not retrieve embedding collection. Skipping RAG.")
        return []
    
    # NOTE: query_texts utilizes chromadb's DefaultEmbeddingFunction dynamically
    # avoiding the need to load the massive PyTorch sentence-transformers stack!
    results = collection.query(
        query_texts=[query_clean],
        n_results=k
    )
    
    retrieved_docs = []
    if results and 'documents' in results and len(results['documents'][0]) > 0:
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        for doc, meta, dist in zip(docs, metadatas, distances):
            retrieved_docs.append({
                "text": doc,
                "metadata": meta,
                "distance": dist
            })
            
    return retrieved_docs

def test_retriever():
    sample_text = "Donald Trump signs executive order withdrawing US from TPP."
    print(f"Testing retrieval with query: '{sample_text}'\n")
    
    results = retrieve_similar_news(sample_text, k=3)
    if not results:
        print("No results found. Did you run scripts/build_embeddings.py yet?")
        return
        
    for i, res in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Label: {res['metadata'].get('label')}")
        print(f"Source: {res['metadata'].get('source')}")
        print(f"Text Snippet: {res['text'][:150]}...\n")

if __name__ == "__main__":
    test_retriever()
