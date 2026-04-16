import os
import sys
import json

import streamlit as st
import joblib
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & path setup
# ---------------------------------------------------------------------------
load_dotenv()  

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.text_cleaner import clean_text
from src.config.config import MODEL_PATH, VECTORIZER_PATH, MODEL_DIR

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="News Credibility Monitor",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — modern dark-friendly design system
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e4e4e7;
    }

    /* ── Minimalist Hero ── */
    .hero-wrapper {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-bottom: 0.25rem;
        background: linear-gradient(135deg, #ffffff 0%, #a1a1aa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        font-size: 1rem;
        color: #a1a1aa;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* ── Buttons and Inputs overriding Streamlit to look like ChatGPT ── */
    div[data-baseweb="textarea"] {
        background-color: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 16px !important;
        transition: all 0.2s ease;
    }
    div[data-baseweb="textarea"]:focus-within {
        border: 1px solid rgba(255,255,255,0.2) !important;
        box-shadow: 0 0 20px rgba(255,255,255,0.03) !important;
    }
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        transition: transform 0.1s ease;
    }
    .stButton > button:active { transform: scale(0.98); }

    /* ── Typography & Headings ── */
    .section-label {
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #71717a;
        margin: 2rem 0 0.75rem 0;
    }

    /* ── Giant Verdict Card ── */
    .master-verdict {
        text-align: center;
        padding: 3.5rem 2rem;
        border-radius: 20px;
        margin: 1.5rem 0 1rem 0;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .master-real {
        background: linear-gradient(180deg, rgba(34,197,94,0.08) 0%, rgba(34,197,94,0.01) 100%);
        border: 1px solid rgba(34,197,94,0.25);
        box-shadow: 0 16px 40px -10px rgba(34,197,94,0.15);
    }
    .master-fake {
        background: linear-gradient(180deg, rgba(239,68,68,0.08) 0%, rgba(239,68,68,0.01) 100%);
        border: 1px solid rgba(239,68,68,0.25);
        box-shadow: 0 16px 40px -10px rgba(239,68,68,0.15);
    }
    .master-unknown {
        background: linear-gradient(180deg, rgba(161,161,170,0.08) 0%, rgba(161,161,170,0.01) 100%);
        border: 1px solid rgba(161,161,170,0.25);
    }

    .mv-verdict {
        font-size: 3.2rem;
        font-weight: 900;
        line-height: 1.1;
        letter-spacing: -1.5px;
        margin: 0.25rem 0;
    }
    .mv-real { color: #4ade80; }
    .mv-fake { color: #f87171; }

    .mv-confidence {
        font-size: 1rem;
        font-weight: 500;
        color: #e4e4e7;
        background: rgba(255,255,255,0.08);
        padding: 4px 16px;
        border-radius: 999px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.05);
    }

    .mv-reasoning {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #d4d4d8;
        max-width: 750px;
        margin: 0 auto;
    }

    /* ── Agent Cards ── */
    .agent-col-card {
        background: rgba(255,255,255,0.015);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1.25rem;
        height: 100%;
        margin-bottom: 0px;
    }
    .agent-header {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #a1a1aa;
        margin-bottom: 0.25rem;
    }
    .agent-v-real { color: #4ade80; font-weight: 700; font-size: 1.1rem; }
    .agent-v-fake { color: #f87171; font-weight: 700; font-size: 1.1rem; }
    .agent-v-unknown { color: #fbbf24; font-weight: 700; font-size: 1.1rem; }
    .agent-short {
        font-size: 0.9rem;
        color: #a1a1aa;
        line-height: 1.5;
        margin-top: 0.75rem;
    }

    /* ── Clean Lists (RAG & Risk) ── */
    .clean-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    .clean-list li {
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 0.95rem;
        color: #e4e4e7;
    }
    .clean-list li:last-child {
        border-bottom: none;
    }
    .risk-text { color: #f87171; }
    .safe-text { color: #4ade80; }

    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* Override metrics */
    div[data-testid="stMetric"] { background: transparent; border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 14px; }
    div[data-testid="stMetric"] label { font-size: 0.75rem !important; opacity: 0.6; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700; }
    
    </style>

    /* ── Per-class metrics table ── */
    .metrics-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.87rem;
    }
    .metrics-table th {
        text-align: left;
        padding: 8px 12px;
        border-bottom: 2px solid rgba(128,128,128,0.25);
        opacity: 0.65;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 0.5px;
    }
    .metrics-table td {
        padding: 10px 12px;
        border-bottom: 1px solid rgba(128,128,128,0.1);
    }
    .metrics-table tr:last-child td { border-bottom: none; }
    .val-cell { font-weight: 600; font-variant-numeric: tabular-nums; }

    /* ── ML score chip inside agent output ── */
    .ml-chip {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(96,165,250,0.1);
        color: #60a5fa;
        border: 1px solid rgba(96,165,250,0.25);
    }

    /* ── Divider ── */
    hr { border-color: rgba(128,128,128,0.15) !important; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1.1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ML models …")
def load_models():
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            return None, None
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    except Exception:
        return None, None


@st.cache_data(show_spinner=False)
def load_metrics():
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


@st.cache_resource(show_spinner="Initialising Agentic AI pipeline …")
def load_graph():
    """Import and warm-up the LangGraph agent graph (cached per session)."""
    try:
        from src.agent.graph import _graph
        return _graph
    except Exception as e:
        return None


model, vectorizer = load_models()
metrics = load_metrics()

if model is None or vectorizer is None:
    st.error("Model artifacts not found. Run the training pipeline first.")
    st.stop()


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict_credibility(text: str):
    """Classic ML: clean → vectorise → predict. Returns (label, confidence%, probs, word_count)."""
    cleaned = clean_text(text)
    if not cleaned:
        return None, None, None, 0
    word_count = len(cleaned.split())
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    confidence = probs[pred] * 100
    return pred, confidence, probs, word_count


def word_count_display(text: str) -> int:
    return len(text.strip().split()) if text.strip() else 0


# ---------------------------------------------------------------------------
# Agent verdict colour helper
# ---------------------------------------------------------------------------
def _verdict_class(verdict_text: str) -> str:
    v = verdict_text.upper()
    if "REAL" in v or "CREDIBLE" in v or "LEGITIMATE" in v:
        return "agent-verdict-real"
    if "FAKE" in v or "FALSE" in v or "FABRICATED" in v or "MISLEADING" in v:
        return "agent-verdict-fake"
    return "agent-verdict-unknown"


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📰 News Credibility Monitor")
    st.caption("GenAI Capstone · Milestone 2")
    st.markdown("---")

    st.markdown("### Analysis Modes")
    st.markdown(
        """
        **Classic ML** uses a Logistic Regression model trained on TF‑IDF features.
        Fast and deterministic.

        **Agentic AI** runs a full LangGraph pipeline: ML → RAG retrieval (Chroma) → LLM reasoning (Groq).
        Richer, explainable output.
        """
    )
    st.markdown("---")

    st.markdown("### User Guide")
    st.markdown(
        """
        **Domain:** US Politics & World News (2016–2018)

        **Tips for best results:**
        - Paste a **full paragraph** (50+ words)
        - Headlines or off-domain topics may be less reliable
        """
    )

    with st.expander("Sample articles to try"):
        st.markdown(
            """
            **REAL:**
            > The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a "fiscal conservative" on Sunday and urged budget cuts in 2018.

            **FAKE:**
            > BREAKING: Hillary Clinton completely melts down after being confronted by angry protesters outside her hotel! You won't believe what she said on camera. Watch the shocking video here before mainstream media takes it down.
            """
        )

    st.markdown("---")
    # System health
    groq_key_set = bool(os.environ.get("GROQ_API_KEY", "").strip())
    st.markdown("### System Status")
    st.markdown(
        f"{'✅' if model is not None else '❌'} **ML Model** — {'Loaded' if model is not None else 'Missing'}"
    )
    st.markdown(
        f"{'✅' if groq_key_set else '⚠️'} **Groq API Key** — {'Configured' if groq_key_set else 'Not set'}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════

# ── Hero header ──
st.markdown(
    """
    <div class="hero-wrapper">
        <h1 class="hero-title">Verify Fact claims.</h1>
        <p class="hero-sub">Enter an article below to run our complete Multi-Agent analysis pipeline.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Section 1 — Model Performance
# ---------------------------------------------------------------------------
if metrics:
    with st.expander("📊 Model Performance Dashboard", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",  f"{metrics['accuracy']  * 100:.2f}%")
        m2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
        m3.metric("Recall",    f"{metrics['recall']    * 100:.2f}%")
        m4.metric("F1-Score",  f"{metrics['f1']        * 100:.2f}%")

        per_class = metrics.get("per_class", {})
        if per_class:
            st.markdown("")
            table_rows = ""
            for cls, vals in per_class.items():
                table_rows += (
                    f"<tr>"
                    f'<td><strong>{cls}</strong></td>'
                    f'<td class="val-cell">{vals["precision"] * 100:.2f}%</td>'
                    f'<td class="val-cell">{vals["recall"] * 100:.2f}%</td>'
                    f'<td class="val-cell">{vals["f1-score"] * 100:.2f}%</td>'
                    f'<td class="val-cell">{int(vals["support"])}</td>'
                    f"</tr>"
                )
            st.markdown(
                f"""
                <table class="metrics-table">
                    <thead>
                        <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
                    </thead>
                    <tbody>{table_rows}</tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Section 2 — Credibility Analyzer
# ---------------------------------------------------------------------------

# ── Mode selection ──
col_mode, col_spacer = st.columns([1, 5])
with col_mode:
    analysis_mode = st.radio(
        "Analysis mode",
        ["Classic ML", "Agentic AI"],
        horizontal=True,
        label_visibility="collapsed",
    )

# Remove old big mode badges completely
st.markdown("<br/>", unsafe_allow_html=True)

# ── Text input ──
news_text = st.text_area(
    "Article Text",
    height=240,
    placeholder=(
        "Paste the full news article text here (50+ words recommended).\n\n"
        "Example: 'The Senate voted on Tuesday to approve a bipartisan infrastructure bill worth...'"
    ),
    label_visibility="collapsed",
)

# Word count logic
wc = word_count_display(news_text)

# ── Short-text warning ──
if news_text.strip() and wc < 20:
    st.warning(
        f"Only **{wc}** words detected. For reliable results, paste a full article (50+ words)."
    )

# ── Analyze button ──
btn_disabled = not news_text.strip()
analyze_clicked = st.button(
    "Analyze Credibility",
    type="primary",
    use_container_width=True,
    disabled=btn_disabled,
)

if not news_text.strip():
    st.caption("Paste an article above to enable analysis.")

# ═══════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════

if analyze_clicked and news_text.strip():
    if wc < 50:
        st.warning("Input text too short. Please provide a longer article (minimum 50 words) so the multi-agent pipeline has enough context to analyze.")
        st.stop()

    st.markdown("---")
    st.markdown('<p class="section-label">Analysis Results</p>', unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────
    # Path A — Classic ML
    # ────────────────────────────────────────────────────────────────────
    if analysis_mode == "Classic ML":
        with st.spinner("Running ML pipeline …"):
            pred, confidence, probs, word_count = predict_credibility(news_text)

        if pred is None:
            st.warning(
                "The input does not contain enough recognizable words after cleaning. "
                "Please provide a more descriptive article."
            )
        else:
            if word_count < 20:
                st.warning(
                    f"Only **{word_count}** meaningful words detected. "
                    "Results may be less reliable with very short inputs."
                )

            res_col, conf_col = st.columns([3, 2])

            with res_col:
                vclass = "master-real" if pred == 0 else "master-fake"
                icon = "✅" if pred == 0 else "🚨"
                title = "Credibility Confirmed" if pred == 0 else "Potentially Fabricated"
                sub = "Language patterns are consistent with factual reporting" if pred == 0 else "Language patterns resemble fabricated content"
                
                st.markdown(
                    f"""
                    <div class="master-verdict {vclass}" style="padding: 2rem;">
                        <div class="mv-verdict {'mv-real' if pred==0 else 'mv-fake'}">{title}</div>
                        <div class="mv-reasoning" style="margin-top:0.5rem;">{sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with conf_col:
                pct_color = "#22c55e" if pred == 0 else "#ef4444"
                st.markdown(
                    f'<div class="conf-box">'
                    f'<div class="conf-label">Confidence</div>'
                    f'<div class="conf-pct" style="color:{pct_color}">{confidence:.1f}%</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.progress(confidence / 100)

                st.markdown("")
                st.markdown(
                    f"""
                    <div class="prob-row">
                        <span class="prob-label">Real probability</span>
                        <span class="prob-val">{probs[0] * 100:.1f}%</span>
                    </div>
                    <div class="prob-row">
                        <span class="prob-label">Fake probability</span>
                        <span class="prob-val">{probs[1] * 100:.1f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ────────────────────────────────────────────────────────────────────
    # Path B — Agentic AI
    # ────────────────────────────────────────────────────────────────────
    else:
        if not os.environ.get("GROQ_API_KEY", "").strip():
            st.error(
                "GROQ_API_KEY is not set. Add it to your `.env` file and restart the app."
            )
        else:
            _graph = load_graph()
            if _graph is None:
                st.error(
                    "Could not load the Agentic AI pipeline. "
                    "Check that all dependencies are installed and try again."
                )
            else:
                report = {}
                initial_state = {
                    "article_text": news_text,
                    "cleaned_text": "",
                    "ml_prediction": "",
                    "ml_confidence": 0.0,
                    "retrieved_docs": [],
                    "llm_response": "",
                    "final_report": {},
                    "error": None,
                }
                
                import time

                with st.status("Agentic AI is evaluating the article...", expanded=True) as status:
                    st.write("**Step 1: Preprocessing input text...**")
                    try:
                        for update in _graph.stream(initial_state):
                            node_name = list(update.keys())[0]
                            state_val = update[node_name]
                            
                            if node_name == "preprocess_node":
                                st.write("✓ Cleaned and tokenised input text.")
                                st.write("")
                                st.write("**Step 2: Checking classic ML prediction...**")
                                # Tiny artificial sleep to make the step readable
                                time.sleep(0.3)
                                
                            elif node_name == "ml_node":
                                ml_pred = state_val.get("ml_prediction", "UNKNOWN")
                                conf = state_val.get("ml_confidence", 0.0)
                                st.write(f"✓ ML suggests **{ml_pred}** ({conf:.1f}% confidence)")
                                st.write("")
                                if conf >= 85.0:
                                    st.write("✓ High confidence threshold met. Skipping RAG retrieval.")
                                    st.write("")
                                    st.write("**Step 3: Multi-Agent Reasoning Pipeline initialized...**")
                                else:
                                    st.write("**Step 3: Searching knowledge base for similar context (RAG)...**")
                                    
                            elif node_name == "rag_node":
                                docs = state_val.get("retrieved_docs", [])
                                st.write(f"✓ Retrieved **{len(docs)}** contextually similar articles.")
                                st.write("")
                                st.write("**Step 4: Multi-Agent Reasoning Pipeline initialized...**")
                                
                            elif node_name == "agent_a_node":
                                st.write("✓ **Agent A (Conservative)** completed analysis.")
                                st.write("")
                            elif node_name == "agent_b_node":
                                st.write("✓ **Agent B (Skeptical)** completed analysis.")
                                st.write("")
                                st.write("**Step 6: Running Agent C (Neutral)...**")

                            elif node_name == "agent_c_node":
                                st.write("✓ **Agent C (Neutral)** completed analysis.")
                                st.write("")
                                st.write("**Step 7: Final Judge Agent evaluating consensus...**")

                            elif node_name == "judge_node":
                                st.write("✓ **Judge Agent** synthesized final verdict.")
                                st.write("")
                                st.write("**Finalizing report...**")
                                
                            elif node_name == "output_node":
                                report = state_val.get("final_report", {})
                                
                        status.update(label="Analysis complete!", state="complete", expanded=False)
                    except Exception as e:
                        status.update(label="Analysis failed.", state="error", expanded=False)
                        st.error(f"Agent pipeline error: {e}")

                if not report:
                    st.error(
                        "The agent did not return a result. "
                        "Check the terminal for error details."
                    )
                else:
                    agent_a   = report.get("agent_a", {})
                    agent_b   = report.get("agent_b", {})
                    agent_c   = report.get("agent_c", {})
                    final     = report.get("final", {})
                    agreement = report.get("agreement", {})
                    rag_sum   = report.get("rag_summary", {})
                    risks     = report.get("risk_factors", [])
                    ml_score  = report.get("ml_signal", "")
                    retrieved = report.get("rag_count", 0)

                    # ── 1. Final Verdict (Master Glowing Card) ──
                    f_verdict = final.get("verdict", "UNKNOWN")
                    vclass = "master-real" if "REAL" in f_verdict.upper() else "master-fake" if "FAKE" in f_verdict.upper() else "master-unknown"
                    vtext = "mv-real" if "REAL" in f_verdict.upper() else "mv-fake" if "FAKE" in f_verdict.upper() else ""
                    
                    st.markdown(
                        f"""
                        <div class="master-verdict {vclass}">
                            <div class="mv-verdict {vtext}">{f_verdict}</div>
                            <div class="mv-confidence">{final.get("confidence", "0")}% Confidence</div>
                            <div class="mv-reasoning">{final.get("consensus", "No consensus provided.")}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # ── 2. Unified Inline Agreement ──
                    a_level = agreement.get("level", "UNKNOWN")
                    dist = agreement.get("distribution", {})
                    dist_str = f"REAL: {dist.get('REAL', 0)} • FAKE: {dist.get('FAKE', 0)}"

                    conflict_str = ""
                    if "High" not in a_level and final.get("conflict") and "none" not in final.get("conflict", "").lower():
                        conflict_str = f"&nbsp;|&nbsp; <strong>Conflict Resolution:</strong> {final.get('conflict')}"

                    st.markdown(
                        f"""
                        <div style="text-align: center; margin-bottom: 2.5rem; padding: 0 1rem;">
                            <span style="font-size: 0.85rem; padding: 6px 14px; background: rgba(255,255,255,0.03); border-radius: 99px; color: #a1a1aa; border: 1px solid rgba(255,255,255,0.08); display: inline-block;">
                                <strong>Agreement:</strong> {a_level} &nbsp;|&nbsp; {dist_str} {conflict_str}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # ── 3. Compact Agent Panel (3 Columns) ──
                    st.markdown('<div class="section-label">Expert Panel Breakdown</div>', unsafe_allow_html=True)
                    
                    c1, c2, c3 = st.columns(3, gap="medium")
                    
                    for col, agent_name, agent_data in zip(
                        [c1, c2, c3],
                        ["Agent A (Conservative)", "Agent B (Skeptical)", "Agent C (Neutral)"],
                        [agent_a, agent_b, agent_c]
                    ):
                        with col:
                            v = agent_data.get("verdict", "UNKNOWN").upper()
                            conf = agent_data.get("confidence", "0")
                            reason = agent_data.get("reasoning", "No reasoning returned.")
                            short_reason = reason.split(".")[0] + "." if "." in reason else reason[:80] + "..."
                            vcolor = "agent-v-real" if "REAL" in v else "agent-v-fake" if "FAKE" in v else "agent-v-unknown"
                            
                            st.markdown(
                                f"""
                                <div class="agent-col-card">
                                    <div class="agent-header">{agent_name}</div>
                                    <div class="{vcolor}">{v} <span style="font-size:0.85rem; color:#71717a; font-weight:400;">({conf}%)</span></div>
                                    <div class="agent-short">{short_reason}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            with st.expander("View detailed reasoning"):
                                st.write(reason)
                                
                    st.markdown("<br/>", unsafe_allow_html=True)

                    # ── 4 & 5. Evidence Summary & Risk Factors ──
                    left_col, right_col = st.columns(2, gap="medium")
                    
                    with left_col:
                        st.markdown('<div class="section-label">Evidence Summary (RAG)</div>', unsafe_allow_html=True)
                        previews = rag_sum.get("previews", [])
                        preview_html = ""
                        if previews:
                            preview_html += "<div style='font-size:0.75rem; color:#71717a; text-transform:uppercase; margin:14px 0 6px 0; font-weight:600; letter-spacing:1px;'>Top Snippets</div>"
                            for p in previews:
                                preview_html += f"<div style='font-size:0.85rem; color:#a1a1aa; border-left: 2px solid rgba(255,255,255,0.1); padding-left: 10px; margin-bottom: 8px;'><i>{p}</i></div>"
                            
                        st.markdown(
                            f"""
                            <ul class="clean-list">
                                <li><strong>Total retrieved documents:</strong> {rag_sum.get("total_docs", 0)}</li>
                                <li><span class="safe-text">REAL:</span> {rag_sum.get("real_docs", 0)} &nbsp;|&nbsp; <span class="risk-text">FAKE:</span> {rag_sum.get("fake_docs", 0)}</li>
                            </ul>
                            {preview_html}
                            """,
                            unsafe_allow_html=True
                        )

                    with right_col:
                        st.markdown('<p class="section-label">4. Risk Factors</p>', unsafe_allow_html=True)
                        if risks:
                            risk_bullets = "".join([f'<li style="margin-bottom:4px;">{r}</li>' for r in risks])
                            st.markdown(
                                f"""
                                <div class="agent-section" style="height:100%;">
                                    <ul style="margin:0;padding-left:20px;font-size:0.95rem;color:#ef4444;">
                                        {risk_bullets}
                                    </ul>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                """
                                <div class="agent-section" style="height:100%; display:flex; align-items:center; justify-content:center;">
                                    <span style="color:#10b981;font-weight:600;">✓ No critical risk factors detected</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "News Credibility Monitor · Milestone 2 · "
    "Built with Scikit-Learn, LangGraph, ChromaDB RAG & Groq LLM reasoning"
)
