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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global typography ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App header hero ── */
    .hero-wrapper {
        padding: 2rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(128,128,128,0.15);
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0 0 6px 0;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 60%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        font-size: 0.97rem;
        opacity: 0.6;
        margin: 0;
        font-weight: 400;
    }

    /* ── Section headings ── */
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        opacity: 0.45;
        margin-bottom: 10px;
    }

    /* ── Mode badge ── */
    .mode-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-bottom: 1.2rem;
    }
    .mode-classic {
        background: rgba(251, 191, 36, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    .mode-agent {
        background: rgba(167, 139, 250, 0.12);
        color: #a78bfa;
        border: 1px solid rgba(167, 139, 250, 0.3);
    }

    /* ── Word count chip ── */
    .wc-chip {
        display: inline-block;
        padding: 3px 11px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        background: rgba(96, 165, 250, 0.1);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.25);
        margin-top: 6px;
    }

    /* ── Verdict cards ── */
    .verdict-card {
        padding: 28px 28px 22px 28px;
        border-radius: 16px;
        text-align: center;
        margin: 6px 0 18px 0;
        transition: box-shadow 0.2s ease;
    }
    .verdict-real {
        background: rgba(34, 197, 94, 0.08);
        border: 2px solid rgba(34, 197, 94, 0.40);
        box-shadow: 0 0 32px rgba(34, 197, 94, 0.08);
    }
    .verdict-real .vcard-icon  { font-size: 2.4rem; margin-bottom: 8px; }
    .verdict-real .vcard-title { font-size: 1.5rem; font-weight: 800; color: #22c55e; margin: 0 0 6px 0; }
    .verdict-real .vcard-sub   { opacity: 0.65; margin: 0; font-size: 0.9rem; }

    .verdict-fake {
        background: rgba(239, 68, 68, 0.08);
        border: 2px solid rgba(239, 68, 68, 0.40);
        box-shadow: 0 0 32px rgba(239, 68, 68, 0.08);
    }
    .verdict-fake .vcard-icon  { font-size: 2.4rem; margin-bottom: 8px; }
    .verdict-fake .vcard-title { font-size: 1.5rem; font-weight: 800; color: #ef4444; margin: 0 0 6px 0; }
    .verdict-fake .vcard-sub   { opacity: 0.65; margin: 0; font-size: 0.9rem; }

    /* ── Confidence gauge ── */
    .conf-box {
        text-align: center;
        padding: 24px 20px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.18);
        margin-top: 6px;
        background: rgba(128,128,128,0.04);
    }
    .conf-pct {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .conf-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        opacity: 0.45;
    }

    /* ── Agent output sections ── */
    .agent-section {
        padding: 20px 22px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.18);
        background: rgba(128,128,128,0.04);
        margin-bottom: 14px;
    }
    .agent-section-title {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.3px;
        text-transform: uppercase;
        opacity: 0.45;
        margin-bottom: 10px;
    }
    .agent-section-body {
        font-size: 0.95rem;
        line-height: 1.75;
        font-weight: 400;
    }

    /* ── Agent verdict inline ── */
    .agent-verdict-real {
        padding: 22px 24px;
        border-radius: 14px;
        background: rgba(34, 197, 94, 0.08);
        border: 2px solid rgba(34, 197, 94, 0.35);
        margin-bottom: 14px;
        text-align: center;
    }
    .agent-verdict-fake {
        padding: 22px 24px;
        border-radius: 14px;
        background: rgba(239, 68, 68, 0.08);
        border: 2px solid rgba(239, 68, 68, 0.35);
        margin-bottom: 14px;
        text-align: center;
    }
    .agent-verdict-unknown {
        padding: 22px 24px;
        border-radius: 14px;
        background: rgba(251, 191, 36, 0.08);
        border: 2px solid rgba(251, 191, 36, 0.35);
        margin-bottom: 14px;
        text-align: center;
    }

    /* ── Disclaimer ── */
    .disclaimer-box {
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px dashed rgba(128,128,128,0.25);
        font-size: 0.78rem;
        opacity: 0.55;
        line-height: 1.6;
        margin-top: 4px;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: transparent;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 14px 18px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.6;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 700;
    }

    /* ── Probabilities table ── */
    .prob-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid rgba(128,128,128,0.1);
        font-size: 0.88rem;
    }
    .prob-row:last-child { border-bottom: none; }
    .prob-label { opacity: 0.75; }
    .prob-val   { font-weight: 600; font-variant-numeric: tabular-nums; }

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
def load_agent():
    """Import and warm-up the LangGraph agent (cached per session)."""
    try:
        from src.agent.graph import run_agent as _run_agent
        return _run_agent
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
        <h1 class="hero-title">News Credibility Monitor</h1>
        <p class="hero-sub">
            Intelligent credibility analysis powered by Agentic AI — LangGraph · RAG · LLM reasoning
        </p>
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
st.markdown("---")
st.markdown('<p class="section-label">Credibility Analyzer</p>', unsafe_allow_html=True)

# ── Mode selection ──
col_mode, col_spacer = st.columns([2, 3])
with col_mode:
    analysis_mode = st.radio(
        "Analysis mode",
        ["Classic ML", "Agentic AI"],
        horizontal=True,
        label_visibility="collapsed",
    )

# Show current mode badge
if analysis_mode == "Classic ML":
    st.markdown(
        '<span class="mode-badge mode-classic">⚡ Classic ML — Logistic Regression + TF‑IDF</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span class="mode-badge mode-agent">🤖 Agentic AI — LangGraph · RAG · Groq LLM</span>',
        unsafe_allow_html=True,
    )

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

# Word count display
wc = word_count_display(news_text)
if news_text.strip():
    wc_color = "#22c55e" if wc >= 50 else "#fbbf24" if wc >= 20 else "#ef4444"
    st.markdown(
        f'<span class="wc-chip" style="color:{wc_color};border-color:{wc_color}40;'
        f'background:{wc_color}12">{wc} words</span>',
        unsafe_allow_html=True,
    )

st.markdown("")

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
                if pred == 0:
                    st.markdown(
                        """
                        <div class="verdict-card verdict-real">
                            <div class="vcard-icon">✅</div>
                            <div class="vcard-title">Credible News</div>
                            <p class="vcard-sub">Language patterns are consistent with verified, factual reporting.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="verdict-card verdict-fake">
                            <div class="vcard-icon">🚨</div>
                            <div class="vcard-title">Potentially Fabricated</div>
                            <p class="vcard-sub">Language patterns resemble those commonly found in unreliable sources.</p>
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
            run_agent = load_agent()
            if run_agent is None:
                st.error(
                    "Could not load the Agentic AI pipeline. "
                    "Check that all dependencies are installed and try again."
                )
            else:
                with st.spinner("Running Agentic AI pipeline (ML → RAG → LLM) …"):
                    try:
                        report = run_agent(news_text)
                    except Exception as e:
                        st.error(f"Agent pipeline error: {e}")
                        report = {}

                if not report:
                    st.error(
                        "The agent did not return a result. "
                        "Check the terminal for error details."
                    )
                else:
                    summary   = report.get("summary", "")
                    analysis  = report.get("analysis", "")
                    verdict   = report.get("verdict", "")
                    disclaimer = report.get("disclaimer", "")
                    ml_score  = report.get("ml_score", "")
                    retrieved = report.get("retrieved_count", 0)

                    # ── Verdict (highlighted) ──
                    vclass = _verdict_class(verdict)
                    verdict_icon = (
                        "✅" if "real" in vclass else
                        "🚨" if "fake" in vclass else "⚠️"
                    )
                    st.markdown(
                        f"""
                        <div class="{vclass}">
                            <div style="font-size:2rem;margin-bottom:6px">{verdict_icon}</div>
                            <div style="font-size:1.05rem;font-weight:700;margin-bottom:6px">Verdict</div>
                            <div style="font-size:0.97rem;line-height:1.65;opacity:0.9">{verdict or "No verdict returned."}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # ── ML signal pill ──
                    if ml_score:
                        st.markdown(
                            f'<span class="ml-chip">ML signal: {ml_score} · {retrieved} RAG docs retrieved</span>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("")

                    # ── Summary + Analysis side by side (or stacked on small screens) ──
                    left, right = st.columns([1, 1], gap="medium")

                    with left:
                        st.markdown(
                            f"""
                            <div class="agent-section">
                                <div class="agent-section-title">Summary</div>
                                <div class="agent-section-body">{summary or "No summary returned."}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with right:
                        st.markdown(
                            f"""
                            <div class="agent-section">
                                <div class="agent-section-title">Analysis</div>
                                <div class="agent-section-body">{analysis or "No analysis returned."}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # ── Disclaimer ──
                    if disclaimer:
                        st.markdown(
                            f'<div class="disclaimer-box">{disclaimer}</div>',
                            unsafe_allow_html=True,
                        )

                    # ── Debug expander ──
                    raw = report.get("raw_llm_response", "")
                    if raw:
                        with st.expander("Raw LLM response (debug view)"):
                            st.text(raw)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "News Credibility Monitor · Milestone 2 · "
    "Built with Scikit-Learn, LangGraph, ChromaDB RAG & Groq LLM reasoning"
)
