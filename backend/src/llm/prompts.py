"""
Prompt templates for the news credibility analysis agent.

Combines ML prediction signals, RAG-retrieved evidence, and the
article text into a structured prompt for the LLM.
"""


def _truncate(text: str, max_chars: int = 500) -> str:
    """Truncate text to *max_chars* on a word boundary."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"


def _format_retrieved_docs(retrieved_docs: list) -> str:
    """
    Format the list of retrieved documents returned by
    ``src.rag.retriever.retrieve_similar_news`` into a readable block
    for the prompt.
    """
    if not retrieved_docs:
        return "No similar articles were found in the reference database."

    lines = []
    for idx, doc in enumerate(retrieved_docs, 1):
        meta = doc.get("metadata", {})
        label = meta.get("label", "UNKNOWN")
        subject = meta.get("subject", "N/A")
        source = meta.get("source", "N/A")
        snippet = _truncate(doc.get("text", ""), max_chars=300)

        lines.append(
            f"[{idx}] Label: {label} | Subject: {subject} | Source: {source}\n"
            f"    Snippet: {snippet}"
        )

    return "\n".join(lines)


def _build_agent_prompt(
    role_description: str,
    article_text: str,
    ml_score: str,
    retrieved_docs: list,
) -> str:
    evidence_block = _format_retrieved_docs(retrieved_docs)
    article_snippet = _truncate(article_text, max_chars=1500)

    return f"""{role_description}

═══════════════════════════════════════
ARTICLE UNDER REVIEW
═══════════════════════════════════════
{article_snippet}

═══════════════════════════════════════
MACHINE LEARNING SIGNAL
═══════════════════════════════════════
ML Prediction: {ml_score}

═══════════════════════════════════════
RETRIEVED REFERENCE ARTICLES (from verified dataset)
═══════════════════════════════════════
{evidence_block}

═══════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════
Using ONLY the information above, produce a structured credibility report in EXACTLY the following format:

Verdict: <REAL or FAKE>
Confidence: <A single number between 0 and 100 representing your confidence>
Reasoning: <Provide a concise explanation (maximum 3 to 5 lines) justifying your verdict based on the evidence>

IMPORTANT RULES:
- Do NOT hallucinate or invent facts not present in the evidence above.
- If the retrieved evidence is insufficient, say so clearly within your Reasoning.
- Strictly adhere to the requested keys (Verdict, Confidence, Reasoning).
"""


def build_conservative_prompt(article_text: str, ml_score: str, retrieved_docs: list) -> str:
    role = (
        "You are Agent A, a Conservative News Analyst. You strongly rely on the provided Machine Learning "
        "prediction and the retrieved reference documents. You trust statistical patterns and prior data, "
        "tending to follow the combined ML and RAG signals heavily in your reasoning."
    )
    return _build_agent_prompt(role, article_text, ml_score, retrieved_docs)


def build_skeptical_prompt(article_text: str, ml_score: str, retrieved_docs: list) -> str:
    role = (
        "You are Agent B, a Skeptical News Analyst. Your primary goal is to challenge the validity of the "
        "claim. Intensely question the evidence quality and completeness. Highlight any uncertainty and "
        "weaknesses in the article."
    )
    return _build_agent_prompt(role, article_text, ml_score, retrieved_docs)


def build_neutral_prompt(article_text: str, ml_score: str, retrieved_docs: list) -> str:
    role = (
        "You are Agent C, a Neutral News Analyst. You provide balanced reasoning, carefully weighing "
        "similarities and discrepancies objectively. Avoid strong bias unless the evidence is extremely clear."
    )
    return _build_agent_prompt(role, article_text, ml_score, retrieved_docs)


def build_judge_prompt(ml_score: str, agent_a: str, agent_b: str, agent_c: str, agreement_level: str, verdict_distribution: str) -> str:
    return f"""You are the Final Judge Agent. You are reviewing the independent analyses of three distinct expert agents (Conservative, Skeptical, and Neutral) regarding a news article's credibility.

═══════════════════════════════════════
MACHINE LEARNING SIGNAL
═══════════════════════════════════════
ML Prediction: {ml_score}

═══════════════════════════════════════
AGENT AGREEMENT LEVEL
═══════════════════════════════════════
Agreement Level: {agreement_level}
Distribution: {verdict_distribution}

═══════════════════════════════════════
AGENT A (Conservative Analyst - Trusts ML/Data)
═══════════════════════════════════════
{agent_a}

═══════════════════════════════════════
AGENT B (Skeptical Analyst - Questions Evidence)
═══════════════════════════════════════
{agent_b}

═══════════════════════════════════════
AGENT C (Neutral Analyst - Balanced Objective Review)
═══════════════════════════════════════
{agent_c}

═══════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════
Your task is to compare all three agent outputs, resolve any disagreements, and produce the final, authoritative verdict.
Keep instructions strictly focused on synthesis, avoiding summarizing the raw data excessively.

Produce your output in EXACTLY the following format:

Final Verdict: <REAL or FAKE>
Final Confidence: <A single number between 0 and 100>
Consensus Summary: <Explanation of how you synthesized the evidence to reach the final verdict>
Dominant Agent: <Specify WHICH agent (A, B, or C) had the strongest influence on the decision and WHY the others were deemed less reliable>
Conflict Resolution: <Explain exactly how you handled any conflicting viewpoints between the agents. If the agreement was High (Unanimous), explicitly output "None, all agents agreed.">
"""
