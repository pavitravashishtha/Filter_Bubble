import json
import numpy as np
from typing import Any

from groq import Groq
from config.env_loader import get_groq_api_key

def _get_client():
    """Returns a fresh Groq client loaded from .env each call."""
    return Groq(api_key=get_groq_api_key())


def build_trajectory_summary(state: Any, checkpoint_t: int) -> dict:
    """
    Builds a population-level metrics summary at a checkpoint timestep.

    Args:
        state: The current SimulationState.
        checkpoint_t: The checkpoint timestep to summarize.

    Returns:
        A dict of aggregate population metrics.
    """
    agents = state.agents
    store = state.measurement_store
    tidx = checkpoint_t - 1

    return {
        "timestep": checkpoint_t,
        "n_agents": len(agents),
        "mean_belief_position": float(np.mean(
            [a.belief_position for a in agents]
        )),
        "std_belief_position": float(np.std(
            [a.belief_position for a in agents]
        )),
        "mean_diversity_score": float(np.mean(
            store.diversity_scores[:, tidx]
        )),
        "mean_echo_chamber_index": float(np.mean(
            store.echo_chamber_index[:, tidx]
        )),
        "mean_exposure_bias": float(np.mean(
            store.exposure_bias[:, tidx]
        )),
        "fraction_high_exposure_bias": float(np.mean(
            store.exposure_bias[:, tidx] > 0.7
        )),
        "n_anomalies": len(store.anomaly_log),
        "n_skips": len(store.skip_log),
        "youtube_weight_history": (
            state.platform_factory.get_platform("youtube")
            .weight_history[-5:]
        ),
        "intervention_active": (
            state.intervention_manager.get_active_intervention()
        ),
    }


def run_sequential_llm(state: Any, checkpoint_t: int,
                        prior_findings: list, config: Any) -> dict:
    """
    Runs a checkpoint sequential LLM analysis to generate research findings and hypotheses.

    Args:
        state: The current SimulationState.
        checkpoint_t: The checkpoint timestep being analyzed.
        prior_findings: List of findings from previous checkpoints.
        config: SimulationConfig controlling whether real API calls are made.

    Returns:
        A dict with findings, anomaly explanations, and testable hypotheses.
    """
    if not config.use_llm:
        return {
            "status": "skipped",
            "checkpoint": checkpoint_t,
            "findings": [],
            "hypotheses": []
        }

    summary = build_trajectory_summary(state, checkpoint_t)

    if checkpoint_t == 400:
        focus = """Focus on: which platform drives fastest 
        bubble formation, algorithm vs social influence 
        contribution, unexpected archetype patterns."""
    elif checkpoint_t == 800:
        focus = """Focus on: bubble pattern evolution since 
        timestep 400, most deeply bubbled agents and why, 
        predicted intervention effectiveness, self-correcting 
        agents and what protected them."""
    else:
        focus = """Focus on: intervention effectiveness by 
        agent type and platform, adaptation rate and speed, 
        backfire events, top 5 findings, hypotheses for 
        next experiment batch."""

    prompt = f"""
    You are analyzing a filter bubble simulation.
    Platforms: IMDB + Reddit + YouTube
    Checkpoint: Timestep {checkpoint_t}
    
    Prior findings: {json.dumps(prior_findings[-3:], indent=2)}
    
    Current metrics summary:
    {json.dumps(summary, indent=2)}
    
    {focus}
    
    Return JSON:
    {{
      "findings": ["finding 1", "finding 2", "finding 3"],
      "anomaly_explanations": ["explanation 1"],
      "hypotheses": [
        {{
          "statement": "hypothesis text",
          "parameter_to_vary": "parameter name",
          "test_value": value,
          "expected_if_true": "description",
          "expected_if_false": "description"
        }}
      ],
      "research_gap_identified": "description or null"
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
        result["checkpoint"] = checkpoint_t
    except json.JSONDecodeError:
        result = {
            "status": "parse_error",
            "raw": raw,
            "findings": [],
            "hypotheses": []
        }

    return result
