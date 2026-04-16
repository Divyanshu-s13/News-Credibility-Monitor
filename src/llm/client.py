"""
Groq API client for LLM-powered credibility reasoning.

Uses the free-tier Groq API with Llama 3 models.
Requires the GROQ_API_KEY environment variable to be set.
"""

import os
import time
import requests
from dotenv import load_dotenv
load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


def _get_api_key() -> str:
    """Retrieve the Groq API key from environment variables."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set. "
            "Get a free key at https://console.groq.com and set it:\n"
            "  export GROQ_API_KEY='gsk_...'"
        )
    return key


def generate_response(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1500,
) -> str:
    """
    Send a prompt to the Groq API and return the generated text.

    Parameters
    ----------
    prompt : str
        The full prompt (system + user content is combined here as a
        single user message for simplicity).
    model : str
        Groq model identifier. Defaults to ``llama3-8b-8192``.
    temperature : float
        Sampling temperature (lower = more deterministic).
    max_tokens : int
        Maximum number of tokens in the response.

    Returns
    -------
    str
        The LLM-generated response text.

    Raises
    ------
    RuntimeError
        If the API call fails after all retries.
    """
    api_key = _get_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional news credibility analyst. "
                    "Respond ONLY based on the evidence provided. "
                    "Never invent or hallucinate facts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )

            # ── Rate-limit handling ──
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", RETRY_DELAY_SECONDS))
                print(f"[Groq] Rate-limited. Retrying in {retry_after}s (attempt {attempt}/{MAX_RETRIES})…")
                time.sleep(retry_after)
                continue

            response.raise_for_status()

            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except requests.exceptions.Timeout:
            last_error = "Request timed out"
            print(f"[Groq] Timeout on attempt {attempt}/{MAX_RETRIES}.")
            time.sleep(RETRY_DELAY_SECONDS)

        except requests.exceptions.HTTPError as http_err:
            last_error = f"HTTP {response.status_code}: {response.text}"
            print(f"[Groq] HTTP error on attempt {attempt}/{MAX_RETRIES}: {last_error}")
            # Don't retry on auth errors — they won't self-resolve
            if response.status_code in (401, 403):
                break
            time.sleep(RETRY_DELAY_SECONDS)

        except requests.exceptions.RequestException as req_err:
            last_error = str(req_err)
            print(f"[Groq] Request error on attempt {attempt}/{MAX_RETRIES}: {last_error}")
            time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"Groq API call failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )
