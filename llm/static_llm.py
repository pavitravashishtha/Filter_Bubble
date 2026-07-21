import json
from typing import Any

from groq import Groq
from config.env_loader import get_groq_api_key

def _get_client():
    """Returns a fresh Groq client loaded from .env each call."""
    return Groq(api_key=get_groq_api_key())


def run_static_llm(agents: list, config: Any) -> dict:
    """
    Runs the static LLM pass to ground simulation parameters in psychology research.

    Args:
        agents: The full list of agent objects.
        config: SimulationConfig controlling whether real API calls are made.

    Returns:
        A dict with parameter adjustments and citations, or a placeholder if use_llm=False.
    """
    if not config.use_llm:
        return {"status": "skipped", "reason": "use_llm=False", "adjustments": {}}

    archetypes = list({a.archetype for a in agents})

    prompt = f"""
    You are grounding simulation parameters in psychology research 
    for a filter bubble simulation.
    
    Platforms: IMDB (taste similarity bubbles), Reddit (community 
    absorption bubbles), YouTube (algorithmic drift bubbles).
    
    Agent population: {len(agents)} agents
    Archetypes present: {archetypes}
    
    Current parameter means by archetype:
    heavy_imdb:    confidence_threshold=0.5, update_rate=0.3, 
                   susceptibility=0.4, open_mindedness=0.6, 
                   critical_thinking=0.6
    heavy_reddit:  confidence_threshold=0.4, update_rate=0.4, 
                   susceptibility=0.5, open_mindedness=0.5, 
                   critical_thinking=0.5
    heavy_youtube: confidence_threshold=0.6, update_rate=0.5, 
                   susceptibility=0.6, open_mindedness=0.4, 
                   critical_thinking=0.4
    balanced:      confidence_threshold=0.5, update_rate=0.3, 
                   susceptibility=0.4, open_mindedness=0.6, 
                   critical_thinking=0.6
    cross_platform:confidence_threshold=0.5, update_rate=0.4, 
                   susceptibility=0.5, open_mindedness=0.7, 
                   critical_thinking=0.5
    
    Review these parameters against empirical psychology research 
    on confirmation bias, attitude change, and social conformity.
    
    Return a JSON object with this exact structure:
    {{
      "adjustments": {{
        "heavy_imdb": {{"confidence_threshold": float, 
                        "update_rate": float,
                        "susceptibility": float,
                        "open_mindedness": float,
                        "critical_thinking": float}},
        ... (same for each archetype)
      }},
      "citations": ["citation 1", "citation 2"],
      "reasoning": "brief explanation"
    }}
    
    Return only valid JSON. No markdown. No preamble.
    """

    response = _get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )

    raw = response.choices[0].message.content
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"status": "parse_error", "raw": raw, "adjustments": {}}

    return result
