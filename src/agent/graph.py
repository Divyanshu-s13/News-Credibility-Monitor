"""
LangGraph StateGraph for the News Credibility agent.

Graph topology:
    preprocess_node
        └── ml_node
                ├── (confidence ≥ 85%) ──────────────────→ llm_node
                └── (confidence  < 85%) → rag_node ───→ llm_node
                                                              └── output_node

Public API:
    run_agent(article_text: str) -> dict
        Returns the final_report dictionary produced by output_node.
"""

import os
import sys

# Ensure project root is on the path when run as a script
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import (
    llm_node,
    ml_node,
    output_node,
    preprocess_node,
    rag_node,
    route_after_ml,
)


def _build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph."""

    builder = StateGraph(AgentState)

    # ── Register nodes ──
    builder.add_node("preprocess_node", preprocess_node)
    builder.add_node("ml_node",         ml_node)
    builder.add_node("rag_node",         rag_node)
    builder.add_node("llm_node",         llm_node)
    builder.add_node("output_node",      output_node)

    # ── Entry point ──
    builder.set_entry_point("preprocess_node")

    # ── Linear edges ──
    builder.add_edge("preprocess_node", "ml_node")

    # ── Conditional branch after ml_node ──
    builder.add_conditional_edges(
        "ml_node",
        route_after_ml,
        {
            "rag_node": "rag_node",   # low confidence  → full RAG + LLM
            "llm_node": "llm_node",   # high confidence → skip RAG
        },
    )

    # Both paths converge at llm_node → output_node → END
    builder.add_edge("rag_node",    "llm_node")
    builder.add_edge("llm_node",    "output_node")
    builder.add_edge("output_node", END)

    return builder.compile()


# Compile once at module load — reused across all run_agent() calls
_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent(article_text: str) -> dict:
    """
    Run the full credibility analysis pipeline as a LangGraph agent.

    Parameters
    ----------
    article_text : str
        Raw news article text to analyse.

    Returns
    -------
    dict
        Final report dictionary with keys:
            summary, analysis, verdict, disclaimer, ml_score,
            retrieved_count, raw_llm_response
    """
    initial_state: AgentState = {
        "article_text":  article_text,
        "cleaned_text":  "",
        "ml_prediction": "",
        "ml_confidence": 0.0,
        "retrieved_docs": [],
        "llm_response":  "",
        "final_report":  {},
        "error":         None,
    }

    final_state = _graph.invoke(initial_state)
    return final_state.get("final_report", {})


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_article = (
        "BREAKING: The government has secretly banned all social media platforms "
        "overnight. Anonymous sources inside the administration confirm the move "
        "was made without Congressional approval. Citizens waking up this morning "
        "found Twitter, Facebook, and Instagram completely inaccessible. The White "
        "House has not issued any official statement, and mainstream media appears "
        "to be intentionally suppressing the story."
    )

    print("=" * 65)
    print("  News Credibility Agent — LangGraph Pipeline Test")
    print("=" * 65)
    print(f"\nArticle: {sample_article[:100]}…\n")

    report = run_agent(sample_article)

    required_keys = {
        "summary", "analysis", "verdict", "disclaimer",
        "ml_score", "retrieved_count",
    }
    missing = required_keys - report.keys()
    if missing:
        print(f"⚠️  Missing keys in report: {missing}")
    else:
        print("✅ All required keys present in final report.\n")

    print("-" * 65)
    for key, value in report.items():
        if key == "raw_llm_response":
            continue   # skip verbose raw output in the summary print
        label = key.replace("_", " ").title()
        print(f"\n{label}:\n{value}")

    print("\n" + "=" * 65)
    print("Full report dict:")
    print("=" * 65)
    import pprint
    pprint.pprint(report)
