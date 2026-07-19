import numpy as np
from dataclasses import dataclass
from typing import List, Any

@dataclass
class Anomaly:
    """
    Represents an anomalous event detected during the simulation.
    """
    agent_id: int
    anomaly_type: str
    description: str
    timestep: int
    agent_belief_position: float
    agent_arousal_level: float

def detect_anomalies(agents: List[Any], measurement_store: Any, timestep: int) -> List[Anomaly]:
    """
    Checks all agents for anomaly conditions.
    
    Args:
        agents: List of agent objects.
        measurement_store: The MeasurementStore instance.
        timestep: The current simulation timestep.
        
    Returns:
        A list of Anomaly objects.
    """
    anomalies = []
    
    for agent in agents:
        latest = measurement_store.get_latest_metrics(agent.id)
        diversity_score = latest.get("diversity_score", 0.0)
        drift_rate = latest.get("drift_rate", 0.0)
        exposure_bias = latest.get("exposure_bias", 0.5)
        
        # ANOMALY 1 - unexpected_resilience
        if agent.arousal_level > 0.7 and diversity_score > 0.6:
            anomalies.append(Anomaly(
                agent_id=agent.id,
                anomaly_type="unexpected_resilience",
                description="High arousal but maintained content diversity",
                timestep=timestep,
                agent_belief_position=agent.belief_position,
                agent_arousal_level=agent.arousal_level
            ))
            
        # ANOMALY 2 - rapid_radicalization
        if drift_rate > 0.8:
            anomalies.append(Anomaly(
                agent_id=agent.id,
                anomaly_type="rapid_radicalization",
                description=f"Drift rate {drift_rate:.2f} exceeds expected maximum",
                timestep=timestep,
                agent_belief_position=agent.belief_position,
                agent_arousal_level=agent.arousal_level
            ))
            
        # ANOMALY 3 - intervention_backfire
        if timestep > 800:
            pre_intervention_bias = measurement_store.get_metric_at(agent.id, "exposure_bias", 800)
            if exposure_bias > pre_intervention_bias:
                anomalies.append(Anomaly(
                    agent_id=agent.id,
                    anomaly_type="intervention_backfire",
                    description="Exposure bias increased after diversity injection",
                    timestep=timestep,
                    agent_belief_position=agent.belief_position,
                    agent_arousal_level=agent.arousal_level
                ))
                
    return anomalies
