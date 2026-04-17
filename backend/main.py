from fastapi import FastAPI
from src.agent.graph import run_agent

app = FastAPI(title="News Credibility Monitor API")

@app.get("/")
def home():
    return {"status": "ok", "message": "News Credibility Monitor API is running"}

@app.post("/analyze")
def analyze(data: dict):
    text = data.get("text", "")

    if not text or len(text.split()) < 50:
        return {"error": "Text too short. Please provide at least 50 words for analysis."}

    try:
        # We just invoke the pipeline asynchronously or synchronously based on defined rules
        result = run_agent(text)
        return result
    except Exception as e:
        return {"error": str(e)}
