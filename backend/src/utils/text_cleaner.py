import re
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are available
try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text)

    # Strip publisher datelines ONLY near the start of text (e.g. "WASHINGTON (Reuters) - ")
    # The old regex '^.*?' was matching across entire sentences containing any parentheses,
    # causing massive content loss for user-supplied text.
    # New pattern: only matches if the parenthesised publisher starts within the first 50 chars.
    text = re.sub(r'^[A-Z][A-Z\s,]{0,40}\s*\([\w\s]+\)\s*[-—]\s*', '', text)

    # Standard cleaning
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)
