"""
config/env_loader.py
====================

Loads environment variables from the project-root .env file and
provides a single accessor for the Anthropic API key.

Usage
-----
The LLM modules call get_api_key() lazily (only when use_llm=True),
so no import-time errors occur during normal simulation runs.
"""

import os
from pathlib import Path


def load_env() -> None:
    """
    Read filter_bubble_sim/.env and inject key=value pairs into
    os.environ using setdefault (never overwrites existing env vars).
    """
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


def get_groq_api_key() -> str:
    """
    Return the Groq API key, loading .env first if needed.

    Raises
    ------
    ValueError
        If GROQ_API_KEY is absent or still set to the placeholder.
    """
    load_env()
    key = os.environ.get("GROQ_API_KEY", "")
    if not key or key == "your-groq-key-here":
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Add it to filter_bubble_sim/.env file."
        )
    return key
