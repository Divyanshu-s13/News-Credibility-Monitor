"""
LangGraph node functions for the News Credibility agent.

Each node accepts an AgentState dict and returns a (partial) AgentState dict
with the fields it populates.  LangGraph merges the returned dict into the
running state automatically.

Node execution order (see graph.py):
    preprocess_node  →  ml_node  →  [conditional]
        ├── high confidence (≥85%) → llm_node  → output_node
        └── low confidence  (<85%) → rag_node  → llm_node → output_node
"""

import os
import re
import sys

import joblib

# Ensure project root is always importable regardless of caller CWD
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.agent.state import AgentState
from src.config.config import MODEL_PATH, VECTORIZER_PATH
from src.llm.client import generate_response
from src.llm.prompts import build_analysis_prompt
from src.rag.retriever import retrieve_similar_news
from src.utils.text_cleaner import clean_text

# ---------------------------------------------------------------------------
# Confidence threshold for the conditional routing branch
# ---------------------------------------------------------------------------
HIGH_CONFIDENCE_THRESHOLD = 85.0   # percent

# ---------------------------------------------------------------------------
# Label mapping — the trained Logistic Regression outputs 0 / 1
# Mirroring src/data/load_data.py which assigns 0 = REAL, 1 = FAKE
# (verify with: model.classes_ → [0, 1])
# ---------------------------------------------------------------------------
_LABEL_MAP = {0: "REAL", 1: "FAKE"}

# ---------------------------------------------------------------------------
# Lazy-loaded model / vectorizer (loaded once per process)
# ---------------------------------------------------------------------------
_model = None
_vectorizer = None


def _load_ml_artifacts():
    """Load and cache the trained model and TF-IDF vectorizer."""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run: python -m src.pipeline.training_pipeline"
            )
        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                f"Vectorizer not found at {VECTORIZER_PATH}. "
                "Run: python -m src.pipeline.training_pipeline"
            )
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer


# ═══════════════════════════════════════════════════════════════════════════
# Node 1 — Text preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_node(state: AgentState) -> AgentState:
    """
    Clean the raw article text using the shared text_cleaner utility.

    Populated: cleaned_text
    """
    try:
        raw = state.get("article_text", "")
        cleaned = clean_text(raw)
        # If the cleaner strips everything (e.g. very short input), fall back
        # to the raw text so downstream nodes aren't blocked.
        if not cleaned.strip():
            cleaned = raw
        return {"cleaned_text": cleaned, "error": None}
    except Exception as exc:
        return {"cleaned_text": state.get("article_text", ""), "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════
# Node 2 — ML classification
# ═══════════════════════════════════════════════════════════════════════════

def ml_node(state: AgentState) -> AgentState:
    """
    Vectorise the cleaned text and run the Logistic Regression classifier.

    Populated: ml_prediction, ml_confidence
    """
    try:
        model, vectorizer = _load_ml_artifacts()
        text = state.get("cleaned_text") or state.get("article_text", "")

        vec = vectorizer.transform([text])
        pred_int = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        confidence = float(probs[pred_int]) * 100.0

        prediction = _LABEL_MAP.get(pred_int, str(pred_int))
        return {
            "ml_prediction": prediction,
            "ml_confidence": round(confidence, 2),
            "error": None,
        }
    except Exception as exc:
        # Fall back to neutral defaults so the graph can continue
        return {
            "ml_prediction": "UNKNOWN",
            "ml_confidence": 0.0,
            "error": str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Conditional routing helper (used by graph.py)
# ═══════════════════════════════════════════════════════════════════════════

def route_after_ml(state: AgentState) -> str:
    """
    Return the name of the next node based on ML confidence.

    Called by the conditional_edge in graph.py.
    """
    confidence = state.get("ml_confidence", 0.0)
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        # High confidence — skip heavy RAG retrieval, go straight to LLM
        return "llm_node"
    # Low confidence — run full RAG retrieval first
    return "rag_node"


# ═══════════════════════════════════════════════════════════════════════════
# Node 3 — RAG retrieval
# ═══════════════════════════════════════════════════════════════════════════

def rag_node(state: AgentState) -> AgentState:
    """
    Retrieve top-5 semantically similar articles from the Chroma vector DB.

    Each returned doc has the structure produced by retrieve_similar_news():
        {"text": str, "metadata": {"label": str, "subject": str, "source": str},
         "distance": float}

    Populated: retrieved_docs
    """
    try:
        query = state.get("article_text", "")   # use raw text for richer signal
        docs = retrieve_similar_news(query, k=5)
        return {"retrieved_docs": docs, "error": None}
    except Exception as exc:
        return {"retrieved_docs": [], "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════
# Node 4 — LLM reasoning
# ═══════════════════════════════════════════════════════════════════════════

def llm_node(state: AgentState) -> AgentState:
    """
    Build the structured prompt from ML + RAG signals and call the Groq LLM.

    Populated: llm_response
    """
    try:
        ml_score = (
            f"{state.get('ml_prediction', 'UNKNOWN')} "
            f"({state.get('ml_confidence', 0.0):.1f}%)"
        )
        retrieved_docs = state.get("retrieved_docs", [])

        prompt = build_analysis_prompt(
            article_text=state.get("article_text", ""),
            ml_score=ml_score,
            retrieved_docs=retrieved_docs,
        )

        response = generate_response(prompt)
        return {"llm_response": response, "error": None}
    except Exception as exc:
        return {"llm_response": "", "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════
# Node 5 — Output formatting
# ═══════════════════════════════════════════════════════════════════════════

def _extract_section(text: str, section: str) -> str:
    """
    Pull the content under a named section heading from the LLM response.

    Expects headings like "Summary:\n<content>\n\nAnalysis:\n..."
    """
    pattern = rf"{section}:\s*(.*?)(?=\n\s*(?:Summary|Analysis|Verdict|Disclaimer):|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def output_node(state: AgentState) -> AgentState:
    """
    Parse the LLM response into a clean structured dictionary.

    Populated: final_report (keys: summary, analysis, verdict, disclaimer,
                              ml_score, retrieved_count)
    """
    llm_response = state.get("llm_response", "")
    ml_prediction = state.get("ml_prediction", "UNKNOWN")
    ml_confidence = state.get("ml_confidence", 0.0)
    retrieved_docs = state.get("retrieved_docs", [])

    report = {
        "summary": _extract_section(llm_response, "Summary"),
        "analysis": _extract_section(llm_response, "Analysis"),
        "verdict": _extract_section(llm_response, "Verdict"),
        "disclaimer": _extract_section(llm_response, "Disclaimer"),
        "ml_score": f"{ml_prediction} ({ml_confidence:.1f}%)",
        "retrieved_count": len(retrieved_docs),
        # Keep the raw LLM response for debugging / app.py display
        "raw_llm_response": llm_response,
    }

    return {"final_report": report, "error": state.get("error")}
