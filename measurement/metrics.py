import numpy as np

def calc_belief_position(agent) -> float:
    """
    Returns agent.belief_position directly.
    Range: 0.0-10.0
    """
    return float(agent.belief_position)

def calc_content_diversity_score(agent, window: int = 20) -> float:
    """
    Calculates standard deviation of content positions seen in the last window timesteps.
    Returns float 0.0-5.0
    """
    history = agent.content_position_history
    if len(history) < 2:
        return 0.5
    
    recent = history[-window:]
    if len(recent) < 2:
        return 0.5
        
    return float(np.std(recent))

def calc_belief_drift_rate(agent, window: int = 10) -> float:
    """
    Rate of belief position change over last window timesteps.
    Returns float 0.0-1.0, clamped.
    """
    history = agent.belief_position_history
    if len(history) < 2:
        return 0.0
        
    recent = history[-window:]
    drift = abs(recent[-1] - recent[0]) / float(len(recent))
    return float(max(0.0, min(1.0, drift)))

def calc_echo_chamber_index(agent, networks: dict) -> float:
    """
    How similar are this agent's social neighbors to the agent?
    Returns float 0.0-1.0 (higher = stronger echo chamber)
    """
    distances = []
    
    for graph in networks.values():
        if agent.id not in graph:
            distances.append(5.0)
            continue
            
        neighbors = list(graph.neighbors(agent.id))
        if not neighbors:
            distances.append(5.0)
            continue
            
        layer_dists = []
        for n in neighbors:
            neighbor_belief = graph.nodes[n].get("belief_position", 5.0)
            layer_dists.append(abs(agent.belief_position - neighbor_belief))
        
        distances.append(float(np.mean(layer_dists)))
        
    mean_distance = float(np.mean(distances)) if distances else 5.0
    echo_chamber_index = 1.0 - (mean_distance / 10.0)
    
    return float(max(0.0, min(1.0, echo_chamber_index)))

def calc_exposure_bias(agent, window: int = 20) -> float:
    """
    Fraction of content served in last window timesteps that was within 2.0 of agent's current belief position.
    Returns float 0.0-1.0
    """
    history = agent.content_position_history
    if not history:
        return 0.5
        
    recent = history[-window:]
    within_range = sum(1 for pos in recent if abs(pos - agent.belief_position) <= 2.0)
    return float(within_range / len(recent))

def calc_intervention_response_rate(agent) -> float:
    """
    Fraction of intervention content that agent engaged with.
    Returns float 0.0-1.0
    """
    if agent.intervention_content_served == 0:
        return 0.0
        
    return float(agent.intervention_content_engaged / agent.intervention_content_served)
