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
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Metric card tweaks */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label {font-size: 0.85rem !important;}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important; font-weight: 700;
    }
    /* Prediction result cards */
    .result-card {
        padding: 24px; border-radius: 12px; text-align: center; margin: 12px 0;
    }
    .result-real {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724; border: 2px solid #a3d9a5;
    }
    .result-fake {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24; border: 2px solid #f1aeb5;
    }
    .confidence-bar {
        height: 8px; border-radius: 4px; margin-top: 8px;
    }
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

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()

        # Try <article> first, then fall back to <p> tags
        article = soup.find("article")
        if article:
            paragraphs = article.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40)
        return text if text else ""
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def predict_credibility(text: str):
    """Clean, vectorize, predict. Returns (label, confidence%, probabilities)."""
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
    st.image("docs/system_design.png", use_container_width=True)
    st.markdown("---")

    # ── Project Roadmap ──
    st.markdown("## 🗺️ Project Roadmap")

    st.markdown("#### ✅ Milestone 1 — *Current*")
    st.markdown(
        """
        **Classical NLP Pipeline (Scikit-Learn)**
        - TF-IDF Vectorization (10 000 features, bigrams)
        - Logistic Regression with balanced class weights
        - Streamlit deployment with confidence scoring
        """
    )

    st.markdown("#### 🔮 Milestone 2 — *Upcoming*")
    st.markdown(
        """
        **Agentic AI Fact-Checker (LangGraph)**
        - Multi-agent autonomous verification pipeline
        - LLM-powered claim extraction & cross-referencing
        - Real-time source credibility scoring
        - LangGraph orchestration for agent collaboration
        """
    )
    st.markdown("---")

    # ── User Guide ──
    st.markdown("## 📖 User Guide")
    st.markdown(
        """
        **Best Practices:**
        - **Domain:** Trained on **US Politics & World News** (2016–2018).
        - **Length:** Paste a full paragraph (50+ words) for accurate analysis.
        - Short headlines or off-domain topics (sports, finance) may yield unreliable results.

        **Try a REAL example:**
        > The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a "fiscal conservative" on Sunday and urged budget cuts in 2018.

        **Try a FAKE example:**
        > BREAKING: Hillary Clinton completely melts down after being confronted by angry protesters outside her hotel! You won't believe what she said on camera. Watch the shocking video here before mainstream media takes it down.
        """
    )

# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════

st.title("📰 News Credibility Monitor")
st.caption("Intelligent Fake-News Detection powered by Classical NLP · Milestone 1")

# ---------------------------------------------------------------------------
# Section 1 — Model Performance Dashboard
# ---------------------------------------------------------------------------
st.markdown("---")
st.header("📊 Model Performance Dashboard")

if metrics:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{metrics['accuracy']  * 100:.2f}%")
    m2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    m3.metric("Recall",    f"{metrics['recall']    * 100:.2f}%")
    m4.metric("F1-Score",  f"{metrics['f1']        * 100:.2f}%")

    with st.expander("📈 Per-Class Metrics & Confusion Matrix", expanded=False):
        pc1, pc2 = st.columns(2)

        with pc1:
            st.markdown("##### Per-Class Breakdown")
            per_class = metrics.get("per_class", {})
            for cls, vals in per_class.items():
                st.markdown(f"**{cls}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Precision", f"{vals['precision'] * 100:.2f}%")
                c2.metric("Recall",    f"{vals['recall']    * 100:.2f}%")
                c3.metric("F1",        f"{vals['f1-score']  * 100:.2f}%")

        with pc2:
            st.markdown("##### Confusion Matrix")
            cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
            else:
                st.info("Confusion matrix image not found. Run the training pipeline to generate it.")
else:
    st.info("Metrics file not found. Run `python -m src.pipeline.training_pipeline` to generate it.")

# ---------------------------------------------------------------------------
# Section 2 — Credibility Analyzer (Dual Input)
# ---------------------------------------------------------------------------
st.markdown("---")
st.header("🔍 Credibility Analyzer")

input_mode = st.radio(
    "Choose input method:",
    ["📝 Paste Article Text", "🔗 Enter Article URL"],
    horizontal=True,
)

news_text = ""

if input_mode == "📝 Paste Article Text":
    news_text = st.text_area(
        "Article Text",
        height=250,
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
                        "For reliable results, paste full articles (50+ words). "
                        "Headlines and short text tend to be classified as Fake due to "
                        "insufficient vocabulary signal."
                    )

                # ── Result Display ──
                res_col, conf_col = st.columns([3, 2])

                with res_col:
                    if pred == 0:
                        st.success("✅ **Verdict: REAL**  — This article's language patterns are consistent with credible news sources.")
                        st.markdown(
                            '<div class="result-card result-real">'
                            "<h3>✅ Credible News</h3>"
                            "<p>The linguistic features of this article align with patterns found in verified, factual news reporting.</p>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error("🚨 **Verdict: FAKE**  — This article's language patterns resemble fabricated or unreliable content.")
                        st.markdown(
                            '<div class="result-card result-fake">'
                            "<h3>🚨 Potentially Fabricated</h3>"
                            "<p>The linguistic features of this article resemble patterns commonly found in unreliable or fabricated news.</p>"
                            "</div>",
                            unsafe_allow_html=True,
                        )

                with conf_col:
                    st.markdown("##### Confidence Score")
                    st.markdown(f"### {confidence:.1f}%")
                    st.progress(confidence / 100)

                    st.markdown("##### Class Probabilities")
                    st.markdown(f"- **Real:** {probs[0] * 100:.1f}%")
                    st.markdown(f"- **Fake:** {probs[1] * 100:.1f}%")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6c757d; font-size:0.85em;'>"
    "News Credibility Monitor · Milestone 1 · Built with Scikit-Learn & Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
