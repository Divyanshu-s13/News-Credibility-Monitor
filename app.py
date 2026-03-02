import os
import streamlit as st
import joblib
import pandas as pd
import os
import sys

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.text_cleaner import clean_text
from src.config.config import MODEL_PATH, VECTORIZER_PATH

# Set page configuration
st.set_page_config(
    page_title="News Credibility Monitor",
    page_icon="📰",
    layout="wide"
)

# Header
st.title("📰 News Credibility Monitor")
st.markdown("### Detect Fake News using Machine Learning")


# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #ced4da;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        font-size: 16px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .result-box-real {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
        text-align: center;
        margin-top: 20px;
    }
    .result-box-fake {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
        text-align: center;
        margin-top: 20px;
    }
    .confidence-text {
        font-size: 14px;
        margin-top: 10px;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource(show_spinner="Loading NLP Models...")
def load_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_models()

if model is None or vectorizer is None:
    st.error("⚠️ Models not found! Please run the training pipeline first.")
    st.stop()

# Add User Guide to the Sidebar (Left-hand side)
with st.sidebar:
    st.markdown("## 📖 User Guide & Best Practices")
    st.markdown("---")
    st.markdown("""
    **How to get the best results:**
    *   **Topic Limitation:** This model was primarily trained on **US Politics** and **World News** from 2016-2018. It is highly accurate on these topics.
    *   **Not for all news:** If you paste Sports, Financial (like RBI repo rates), or Entertainment news, the model may incorrectly guess "Fake" because it doesn't recognize the vocabulary.
    *   **Length:** Paste a full paragraph or an entire article (50+ words) for the best analysis. Single sentences lack the context needed for the NLP model to find credibility patterns.
    
    ### Examples
    *Try this **REAL** example:*
    > The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a "fiscal conservative" on Sunday and urged budget cuts in 2018.

    *Try this **FAKE** example:*
    > BREAKING: Hillary Clinton completely melts down after being confronted by angry protesters outside her hotel! You won't believe what she said on camera. Watch the shocking video here before mainstream media takes it down.
    """)

st.markdown("Enter the text of a news article below to predict whether it is **Real** or **Fake**.")

# Input area
news_text = st.text_area("Article Text:", height=250, placeholder="Paste the news article text here...")

# Predict button
if st.button("Predict Credibility"):
    if not news_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            try:
                # Preprocess the text
                cleaned_input = clean_text(news_text)
                
                if not cleaned_input:
                    st.warning("⚠️ The input text does not contain enough recognizable words after cleaning. Please provide a more descriptive article.")
                else:
                    word_count = len(cleaned_input.split())
                    if word_count < 20:
                        st.warning(
                            f"⚠️ Only **{word_count} meaningful words** found after cleaning. "
                            "This model was trained on full news articles (50+ words). "
                            "Paste a full paragraph for accurate results — headlines and short sentences will almost always be classified as **Fake** due to insufficient vocabulary signal."
                        )

                    # Vectorize the text
                    vectorized_input = vectorizer.transform([cleaned_input])
                    
                    # Predict
                    prediction = model.predict(vectorized_input)[0]
                    probabilities = model.predict_proba(vectorized_input)[0]
                    
                    # Display results
                    st.markdown("### Analysis Result")
                    if prediction == 0:
                        st.markdown("""
                        <div class="result-box-real">
                            <h4>✅ This news appears to be <b>REAL</b>.</h4>
                            <p>Our model has classified this text as credible news based on its language patterns.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="result-box-fake">
                            <h4>🚨 This news appears to be <b>FAKE</b>.</h4>
                            <p>Our model has classified this text as potentially fabricated or unreliable.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show confidence
                    confidence = probabilities[prediction] * 100
                    st.markdown(f"<p class='confidence-text'>Confidence Score: {confidence:.2f}%</p>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #6c757d; font-size: 0.9em;'>Built with Streamlit & Scikit-Learn</p>", unsafe_allow_html=True)
