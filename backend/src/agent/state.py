"""
AgentState definition for the News Credibility LangGraph agent.

Every node in the graph receives and returns an instance of this TypedDict.
Fields are populated progressively as the article moves through the pipeline.
"""

from typing import Any, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state dictionary passed between all LangGraph nodes."""

    # Input
    article_text: str               # Raw article text supplied by the caller

    # After preprocess_node
    cleaned_text: str               # Text after clean_text() is applied

    # After ml_node
    ml_prediction: str              # "FAKE" or "REAL"
    ml_confidence: float            # Confidence percentage, e.g. 82.3

    # After rag_node (may be skipped on high-confidence paths)
    retrieved_docs: List[dict]      # List of dicts from retrieve_similar_news()

    # After respective agents
    agent_a_response: str
    agent_b_response: str
    agent_c_response: str
    judge_response: str

    # After output_node
    final_report: dict              # Parsed dict with keys: summary, analysis,
                                    # verdict, disclaimer, ml_score, retrieved_count

    # Error handling — any node may populate this on failure
    error: Optional[str]
