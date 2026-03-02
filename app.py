import os
import sys
import json

import streamlit as st
import joblib
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
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
# Theme-aware CSS — no hardcoded white backgrounds
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Metric cards: transparent, theme-aware ---- */
    div[data-testid="stMetric"] {
        background: transparent;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 14px 18px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.7;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 700;
    }

    /* ---- Pill-style metric badges (used in dashboard) ---- */
    .metric-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 3px 4px;
    }
    .pill-green  { background: #0e6b3620; color: #22c55e; border: 1px solid #22c55e40; }
    .pill-blue   { background: #3b82f620; color: #60a5fa; border: 1px solid #60a5fa40; }
    .pill-amber  { background: #f59e0b20; color: #fbbf24; border: 1px solid #fbbf2440; }
    .pill-purple { background: #8b5cf620; color: #a78bfa; border: 1px solid #a78bfa40; }

    /* ---- Per-class table ---- */
    .metrics-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 8px;
        font-size: 0.88rem;
    }
    .metrics-table th {
        text-align: left;
        padding: 8px 12px;
        border-bottom: 2px solid rgba(128,128,128,0.3);
        opacity: 0.7;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.5px;
    }
    .metrics-table td {
        padding: 10px 12px;
        border-bottom: 1px solid rgba(128,128,128,0.15);
    }
    .metrics-table tr:last-child td { border-bottom: none; }
    .val-cell { font-weight: 600; font-variant-numeric: tabular-nums; }

    /* ---- Result cards ---- */
    .verdict-card {
        padding: 28px 24px;
        border-radius: 14px;
        text-align: center;
        margin: 8px 0 16px 0;
    }
    .verdict-real {
        background: rgba(34, 197, 94, 0.1);
        border: 2px solid rgba(34, 197, 94, 0.4);
    }
    .verdict-real h3 { color: #22c55e; margin: 0 0 6px 0; }
    .verdict-real p  { opacity: 0.8; margin: 0; }

    .verdict-fake {
        background: rgba(239, 68, 68, 0.1);
        border: 2px solid rgba(239, 68, 68, 0.4);
    }
    .verdict-fake h3 { color: #ef4444; margin: 0 0 6px 0; }
    .verdict-fake p  { opacity: 0.8; margin: 0; }

    /* ---- Confidence gauge box ---- */
    .conf-box {
        text-align: center;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.2);
        margin-top: 8px;
    }
    .conf-pct {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .conf-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.55;
    }

    /* ---- Sidebar polish ---- */
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1.1rem !important;
    }
    .roadmap-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-left: 4px;
    }
    .badge-active  { background: #22c55e25; color: #22c55e; border: 1px solid #22c55e50; }
    .badge-upcoming { background: #8b5cf625; color: #a78bfa; border: 1px solid #a78bfa50; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model & metrics loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading NLP models …")
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


model, vectorizer = load_models()
metrics = load_metrics()

if model is None or vectorizer is None:
    st.error("⚠️ Model artifacts not found. Run the training pipeline first.")
    st.stop()


# ---------------------------------------------------------------------------
# URL scraper helper
# ---------------------------------------------------------------------------
def scrape_article(url: str) -> str:
    """Extract the main text content from a news article URL."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url.strip(), headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()

        article = soup.find("article")
        if article:
            paragraphs = article.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        text = " ".join(
            p.get_text(strip=True)
            for p in paragraphs
            if len(p.get_text(strip=True)) > 40
        )
        return text if text else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def predict_credibility(text: str):
    """Clean, vectorize, predict. Returns (label, confidence%, probabilities, word_count)."""
    cleaned = clean_text(text)
    if not cleaned:
        return None, None, None, 0

    word_count = len(cleaned.split())
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    confidence = probs[pred] * 100
    return pred, confidence, probs, word_count


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📰 News Credibility Monitor")
    st.caption("GenAI Capstone · Milestone 1")
    st.markdown("---")

    # ── Project Roadmap ──
    st.markdown("### 🗺️ Project Roadmap")

    st.markdown(
        '**Milestone 1** <span class="roadmap-badge badge-active">Active</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        Classical NLP Pipeline (Scikit-Learn)
        - TF-IDF Vectorization · 10k features · bigrams
        - Logistic Regression · balanced class weights
        - Streamlit deployment · confidence scoring
        """
    )

    st.markdown(
        '**Milestone 2** <span class="roadmap-badge badge-upcoming">Upcoming</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        Agentic AI Fact-Checker (LangGraph)
        - Multi-agent autonomous verification
        - LLM-powered claim extraction
        - Real-time source credibility scoring
        """
    )
    st.markdown("---")

    # ── User Guide ──
    st.markdown("### 📖 User Guide")
    st.markdown(
        """
        **Domain:** US Politics & World News (2016–2018)

        **Tips for best results:**
        - Paste a **full paragraph** (50+ words)
        - Headlines or off-domain topics may be unreliable
        """
    )

    with st.expander("💡 Sample articles to try"):
        st.markdown(
            """
            **REAL:**
            > The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a "fiscal conservative" on Sunday and urged budget cuts in 2018.

            **FAKE:**
            > BREAKING: Hillary Clinton completely melts down after being confronted by angry protesters outside her hotel! You won't believe what she said on camera. Watch the shocking video here before mainstream media takes it down.
            """
        )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════

st.title("📰 News Credibility Monitor")
st.caption("Intelligent Fake-News Detection powered by Classical NLP · Milestone 1")

# ---------------------------------------------------------------------------
# Section 1 — Model Performance (compact)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Model Performance")

if metrics:
    # Compact metric row using st.metric — no white boxes now
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{metrics['accuracy']  * 100:.2f}%")
    m2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    m3.metric("Recall",    f"{metrics['recall']    * 100:.2f}%")
    m4.metric("F1-Score",  f"{metrics['f1']        * 100:.2f}%")

    # Per-class as a clean HTML table inside an expander
    with st.expander("View per-class breakdown"):
        per_class = metrics.get("per_class", {})
        table_rows = ""
        for cls, vals in per_class.items():
            table_rows += (
                f'<tr>'
                f'<td><strong>{cls}</strong></td>'
                f'<td class="val-cell">{vals["precision"] * 100:.2f}%</td>'
                f'<td class="val-cell">{vals["recall"] * 100:.2f}%</td>'
                f'<td class="val-cell">{vals["f1-score"] * 100:.2f}%</td>'
                f'<td class="val-cell">{int(vals["support"])}</td>'
                f'</tr>'
            )
        st.markdown(
            f"""
            <table class="metrics-table">
                <thead>
                    <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
                </thead>
                <tbody>{table_rows}</tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("Metrics not found. Run `python -m src.pipeline.training_pipeline` to generate.")

# ---------------------------------------------------------------------------
# Section 2 — Credibility Analyzer (Dual Input)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 Credibility Analyzer")

input_mode = st.radio(
    "Choose input method:",
    ["📝 Paste Article Text", "🔗 Enter Article URL"],
    horizontal=True,
)

news_text = ""

if input_mode == "📝 Paste Article Text":
    news_text = st.text_area(
        "Article Text",
        height=220,
        placeholder="Paste the full news article text here …",
    )
else:
    url_input = st.text_input(
        "Article URL",
        placeholder="https://www.example.com/news-article",
    )
    if url_input:
        with st.spinner("Fetching article from URL …"):
            news_text = scrape_article(url_input)
            if news_text:
                st.success(f"Extracted {len(news_text.split())} words from the article.")
                with st.expander("Preview extracted text"):
                    st.write(news_text[:2000] + (" …" if len(news_text) > 2000 else ""))
            else:
                st.error(
                    "Could not extract text from this URL. The site may block automated access. "
                    "Please copy-paste the article text manually instead."
                )

# ── Predict Button ──
if st.button("⚡ Analyze Credibility", type="primary", use_container_width=True):
    if not news_text.strip():
        st.warning("Please provide article text or a valid URL.")
    else:
        with st.spinner("Running NLP pipeline …"):
            pred, confidence, probs, word_count = predict_credibility(news_text)

            if pred is None:
                st.warning(
                    "The input does not contain enough recognizable words after cleaning. "
                    "Please provide a more descriptive article."
                )
            else:
                # Word-count warning
                if word_count < 20:
                    st.warning(
                        f"Only **{word_count}** meaningful words detected. "
                        "For reliable results, paste full articles (50+ words)."
                    )

                st.markdown("---")

                # ── Result Display ──
                res_col, conf_col = st.columns([3, 2])

                with res_col:
                    if pred == 0:
                        st.success("✅  **Verdict: REAL**")
                        st.markdown(
                            '<div class="verdict-card verdict-real">'
                            "<h3>✅ Credible News</h3>"
                            "<p>Language patterns are consistent with verified, factual reporting.</p>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error("🚨  **Verdict: FAKE**")
                        st.markdown(
                            '<div class="verdict-card verdict-fake">'
                            "<h3>🚨 Potentially Fabricated</h3>"
                            "<p>Language patterns resemble those commonly found in unreliable sources.</p>"
                            "</div>",
                            unsafe_allow_html=True,
                        )

                with conf_col:
                    # Confidence gauge
                    pct_color = "#22c55e" if pred == 0 else "#ef4444"
                    st.markdown(
                        f'<div class="conf-box">'
                        f'<div class="conf-label">Confidence</div>'
                        f'<div class="conf-pct" style="color:{pct_color}">{confidence:.1f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(confidence / 100)

                    st.markdown("")
                    st.markdown(f"**Real probability:** {probs[0] * 100:.1f}%")
                    st.markdown(f"**Fake probability:** {probs[1] * 100:.1f}%")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("News Credibility Monitor · Milestone 1 · Built with Scikit-Learn & Streamlit")
