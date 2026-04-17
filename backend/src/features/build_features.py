from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer():
    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        # Stopwords are already removed in text_cleaner.py — do NOT remove again here
        # to avoid double-filtering and vocabulary drift between train/inference.
    )