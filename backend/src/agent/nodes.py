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
from langchain_nvidia_ai_endpoints import ChatNVIDIA

import joblib

# Ensure project root is always importable regardless of caller CWD
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.agent.state import AgentState
from src.config.config import MODEL_PATH, VECTORIZER_PATH
from src.llm.client import generate_response
from src.llm.prompts import (
    build_conservative_prompt,
    build_skeptical_prompt,
    build_neutral_prompt,
    build_judge_prompt,
)
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
        return "agent_a_node"
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
# Nodes 4, 5, 6 — Reasoning Agents (A, B, C)
# ═══════════════════════════════════════════════════════════════════════════

def agent_a_node(state: AgentState) -> AgentState:
    try:
        ml_score = f"{state.get('ml_prediction', 'UNKNOWN')} ({state.get('ml_confidence', 0.0):.1f}%)"
        prompt = build_conservative_prompt(state.get("article_text", ""), ml_score, state.get("retrieved_docs", []))
        response = generate_response(prompt)
        return {"agent_a_response": response, "error": None}
    except Exception as exc:
        return {"agent_a_response": f"Verdict: UNKNOWN\nConfidence: 0\nReasoning: Error {exc}", "error": str(exc)}


def agent_b_node(state: AgentState) -> AgentState:
    try:
        ml_score = f"{state.get('ml_prediction', 'UNKNOWN')} ({state.get('ml_confidence', 0.0):.1f}%)"
        prompt = build_skeptical_prompt(state.get("article_text", ""), ml_score, state.get("retrieved_docs", []))
        response = generate_response(prompt)
        return {"agent_b_response": response, "error": None}
    except Exception as exc:
        return {"agent_b_response": f"Verdict: UNKNOWN\nConfidence: 0\nReasoning: Error {exc}", "error": str(exc)}


def agent_c_node(state: AgentState) -> AgentState:
    try:
        ml_score = f"{state.get('ml_prediction', 'UNKNOWN')} ({state.get('ml_confidence', 0.0):.1f}%)"
        prompt = build_neutral_prompt(state.get("article_text", ""), ml_score, state.get("retrieved_docs", []))
        response = generate_response(prompt)
        return {"agent_c_response": response, "error": None}
    except Exception as exc:
        return {"agent_c_response": f"Verdict: UNKNOWN\nConfidence: 0\nReasoning: Error {exc}", "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════
# Node 7 — Final Judge Agent
# ═══════════════════════════════════════════════════════════════════════════

def judge_node(state: AgentState) -> AgentState:
    try:
        a_resp = state.get("agent_a_response", "")
        b_resp = state.get("agent_b_response", "")
        c_resp = state.get("agent_c_response", "")
        
        # Precompute agreement to feed into the Judge prompt
        v_a = _extract_field(a_resp, "Verdict").upper()
        v_b = _extract_field(b_resp, "Verdict").upper()
        v_c = _extract_field(c_resp, "Verdict").upper()
        
        real_count = sum(1 for v in [v_a, v_b, v_c] if "REAL" in v)
        fake_count = sum(1 for v in [v_a, v_b, v_c] if "FAKE" in v)
        
        distribution = f"REAL: {real_count}, FAKE: {fake_count}"
        
        if real_count == 3 or fake_count == 3:
            agreement_level = "High (3/3 agents aligned)"
        elif real_count == 2 or fake_count == 2:
            agreement_level = "Medium (2 vs 1 split)"
        else:
            agreement_level = "Low (All agents disagreed / uncertain)"

        ml_score = f"{state.get('ml_prediction', 'UNKNOWN')} ({state.get('ml_confidence', 0.0):.1f}%)"
        prompt = build_judge_prompt(
            ml_score,
            a_resp,
            b_resp,
            c_resp,
            agreement_level,
            distribution
        )
        
        try:
            judge_llm = ChatNVIDIA(
                model="meta/llama-3.3-70b-instruct",
                api_key=os.getenv("NVIDIA_API_KEY"),
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
            )
            judge_response = judge_llm.invoke(prompt)
            response = judge_response.content
        except Exception as e:
            # fallback to existing LLM (Groq)
            response = generate_response(prompt)
            
        return {"judge_response": response, "error": None}
    except Exception as exc:
        return {"judge_response": f"Final Verdict: UNKNOWN\nFinal Confidence: 0\nConsensus Summary: Error {exc}\nDisagreement Reason: Error", "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════
# Node 8 — Output formatting
# ═══════════════════════════════════════════════════════════════════════════

def _extract_field(text: str, field_name: str) -> str:
    """Extract a simple labeled field like 'Verdict: REAL' up to the newline."""
    pattern = rf"{field_name}:\s*(.*?)(?=\n|$)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def _extract_reasoning(text: str) -> str:
    """Extract multi-line reasoning."""
    pattern = rf"Reasoning:\s*(.*?)(?=\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def _extract_consensus(text: str) -> str:
    """Extract multi-line consensus."""
    pattern = rf"Consensus Summary:\s*(.*?)(?=\n\s*Dominant Agent:|\n\s*Conflict Resolution:|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def _extract_conflict(text: str) -> str:
    """Extract multi-line conflict resolution."""
    pattern = rf"Conflict Resolution:\s*(.*?)(?=\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def _parse_agent(response_str: str) -> dict:
    return {
        "verdict": _extract_field(response_str, "Verdict"),
        "confidence": _extract_field(response_str, "Confidence"),
        "reasoning": _extract_reasoning(response_str),
    }

def output_node(state: AgentState) -> AgentState:
    """
    Assemble the LLM outputs and compute explicit rules and summaries.
    """
    # 1. RAG Summary
    retrieved_docs = state.get("retrieved_docs", [])
    rag_real_count = sum(1 for doc in retrieved_docs if "REAL" in doc.get("metadata", {}).get("label", "").upper())
    rag_fake_count = sum(1 for doc in retrieved_docs if "FAKE" in doc.get("metadata", {}).get("label", "").upper())
    
    doc_previews = []
    for doc in retrieved_docs[:2]:
        raw_text = doc.get("text", "")
        # Very basic logic to clean out the prompt-formatting of RAG articles for snippet
        snippet = doc.get("metadata", {}).get("subject", "Reference Article")
        # take the first 150 chars
        doc_previews.append(f"{snippet}: {raw_text[:150]}...")
            
    rag_summary = {
        "total_docs": len(retrieved_docs),
        "real_docs": rag_real_count,
        "fake_docs": rag_fake_count,
        "previews": doc_previews
    }
    
    # 2. Agreement calculation
    a_resp = state.get("agent_a_response", "")
    b_resp = state.get("agent_b_response", "")
    c_resp = state.get("agent_c_response", "")
    
    v_a = _extract_field(a_resp, "Verdict").upper()
    v_b = _extract_field(b_resp, "Verdict").upper()
    v_c = _extract_field(c_resp, "Verdict").upper()
    
    real_count = sum(1 for v in [v_a, v_b, v_c] if "REAL" in v)
    fake_count = sum(1 for v in [v_a, v_b, v_c] if "FAKE" in v)
    
    if real_count == 3 or fake_count == 3:
        agreement_level = "High"
    elif real_count == 2 or fake_count == 2:
        agreement_level = "Medium"
    else:
        agreement_level = "Low"
        
    agreement = {
        "level": agreement_level,
        "distribution": {"REAL": real_count, "FAKE": fake_count}
    }
    
    # 3. Risk Factors computation
    risk_factors = []
    ml_conf = state.get('ml_confidence', 0.0)
    if ml_conf < 70.0:
        risk_factors.append("Low ML confidence (<70%)")
    
    if rag_real_count > 0 and rag_fake_count > 0:
        risk_factors.append("Conflicting evidence (Mixed RAG signals)")
        
    if agreement_level in ["Medium", "Low"]:
        risk_factors.append(f"Weak consensus ({'2 vs 1 split' if agreement_level == 'Medium' else 'all mismatched'})")
        
    if len(retrieved_docs) <= 2:
        risk_factors.append("Limited supporting data (Few docs retrieved)")
        risk_factors.append("Lack of strong supporting evidence")
        
    # Jaccard Check for repetition
    if len(retrieved_docs) > 1:
        text1 = set(retrieved_docs[0].get("text", "").lower().split())
        text2 = set(retrieved_docs[1].get("text", "").lower().split())
        if text1 and text2:
            jaccard = len(text1.intersection(text2)) / float(len(text1.union(text2)))
            if jaccard > 0.4:
                risk_factors.append("Repetitive or similar retrieved content")
        
    # 4. Assemble Final Dict
    judge_resp = state.get("judge_response", "")
    report = {
        "agent_a": _parse_agent(a_resp),
        "agent_b": _parse_agent(b_resp),
        "agent_c": _parse_agent(c_resp),
        "final": {
            "verdict": _extract_field(judge_resp, "Final Verdict"),
            "confidence": _extract_field(judge_resp, "Final Confidence"),
            "consensus": _extract_consensus(judge_resp),
            "dominant_agent": _extract_field(judge_resp, "Dominant Agent"),
            "conflict": _extract_conflict(judge_resp),
        },
        "agreement": agreement,
        "rag_summary": rag_summary,
        "risk_factors": risk_factors,
        "ml_signal": f"{state.get('ml_prediction', 'UNKNOWN')} ({ml_conf:.1f}%)",
        "rag_count": len(retrieved_docs),
    }
    return {"final_report": report, "error": state.get("error")}
