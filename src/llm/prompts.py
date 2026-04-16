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


def build_analysis_prompt(
    article_text: str,
    ml_score: str,
    retrieved_docs: list,
) -> str:
    """
    Build the full analysis prompt sent to the LLM.

    Parameters
    ----------
    article_text : str
        The raw (or lightly cleaned) article text to analyse.
    ml_score : str
        Human-readable ML prediction, e.g. ``"FAKE (82.3%)"`` or
        ``"REAL (94.1%)"``.
    retrieved_docs : list[dict]
        Output of ``retrieve_similar_news()`` — each dict has keys
        ``text``, ``metadata``, and ``distance``.

    Returns
    -------
    str
        Fully assembled prompt string.
    """
    evidence_block = _format_retrieved_docs(retrieved_docs)
    article_snippet = _truncate(article_text, max_chars=1500)

    prompt = f"""You are a news credibility analyst. Your job is to evaluate whether the article below is REAL or FAKE based ONLY on the evidence provided. Do NOT invent facts or make unsupported claims.

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

Summary:
<Write a brief 2-3 sentence summary of the article under review.>

Analysis:
<Compare the article with the retrieved reference articles. Note similarities or differences in language, topic, and style. Comment on how the ML prediction aligns with the retrieved evidence. If the evidence is weak or limited, explicitly state that.>

Verdict:
<State "Likely REAL" or "Likely FAKE" with a brief justification. Include a confidence qualifier (High / Moderate / Low) based on the strength of evidence.>

Disclaimer:
<State that this analysis is automated and based on pattern matching against a reference dataset. It should not be treated as a definitive fact-check. Recommend consulting authoritative fact-checking organisations for verification.>

IMPORTANT RULES:
- Do NOT hallucinate or invent facts not present in the evidence above.
- If the retrieved evidence is insufficient, say so clearly.
- Keep the response concise and well-structured.
"""
    return prompt
