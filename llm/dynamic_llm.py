import json
from typing import Any, Optional

from groq import Groq
from config.env_loader import get_groq_api_key

def _get_client():
    """Returns a fresh Groq client loaded from .env each call."""
    return Groq(api_key=get_groq_api_key())


def build_dynamic_context(agent: Any, state: Any, timestep: int, trigger: str) -> dict:
    """
    Builds a context dict summarizing agent state for a dynamic LLM prompt.

    Args:
        agent: The agent triggering a dynamic LLM call.
        state: The current SimulationState.
        timestep: The current simulation timestep.
        trigger: The string describing what triggered this call.

    Returns:
        A dictionary of current agent state and history.
    """
    return {
        "agent_id": agent.id,
        "archetype": agent.archetype,
        "belief_position": agent.belief_position,
        "arousal_level": agent.arousal_level,
        "interaction_count": agent.interaction_count,
        "trigger": trigger,
        "timestep": timestep,
        "recent_belief_history": agent.belief_position_history[-20:],
        "recent_content_history": agent.content_position_history[-20:],
        "platform_weights": agent.platform_weights,
        "primary_community": agent.primary_community,
    }


def run_dynamic_llm(agent: Any, state: Any, timestep: int,
                    trigger: str, config: Any) -> dict:
    """
    Runs a mid-simulation dynamic LLM call to analyze an anomalous agent event.

    Args:
        agent: The agent to analyze.
        state: The current SimulationState.
        timestep: The current simulation timestep.
        trigger: The trigger event description.
        config: SimulationConfig controlling whether real API calls are made.

    Returns:
        A dict with analysis, trajectory prediction, and optional parameter adjustments.
    """
    if not config.use_llm:
        return {
            "status": "skipped",
            "trigger": trigger,
            "parameter_adjustments": None,
            "explanation": ""
        }

    context = build_dynamic_context(agent, state, timestep, trigger)

    prompt = f"""
    You are analyzing a mid-simulation event in a filter bubble study.
    
    Agent context:
    {json.dumps(context, indent=2)}
    
    Trigger event: {trigger}
    
    Answer specifically in JSON:
    {{
      "primary_driver": "algorithm_pressure|social_influence|content_valence|state_interaction",
      "trajectory_likely": "continue|self_correct|uncertain",
      "parameter_adjustments": null or {{"susceptibility": float, "critical_thinking": float}},
      "explanation": "one paragraph explanation",
      "bubble_vulnerability": "low|medium|high|critical"
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
        result = {"status": "parse_error", "raw": raw, "parameter_adjustments": None}

    return result
