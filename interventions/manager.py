import random
from typing import Dict, Any, List, Optional

from interventions.null_intervention import apply_null_intervention
from interventions.algorithm_friction import apply_algorithm_friction
from interventions.social_bridge import apply_social_bridges
from interventions.diversity_injection import (
    activate_diversity_injection,
    deactivate_diversity_injection
)

class InterventionManager:
    """
    Manages and applies interventions during the simulation.
    """
    def __init__(self, config: Any, platforms: Dict[str, Any], networks: Dict[str, Any]):
        self.config = config
        self.platforms = platforms
        self.networks = networks
        self.active_intervention: Optional[str] = None
        self.intervention_log: List[Dict[str, Any]] = []
        self.agent_response_tracker: Dict[int, List[Dict[str, Any]]] = {}
        self.cooldown_per_agent: Dict[int, int] = {}

    def activate(self, timestep: int) -> None:
        """
        Activates the configured intervention.
        """
        intervention_type = self.config.intervention_type
        self.active_intervention = intervention_type
        
        if intervention_type == "diversity_injection":
            activate_diversity_injection(self.config)
        elif intervention_type == "algorithm_friction":
            apply_algorithm_friction(
                self.platforms, 
                self.config.intervention_strength
            )
        elif intervention_type == "social_bridge":
            apply_social_bridges(self.networks, [])
        elif intervention_type == "null":
            apply_null_intervention(None)
        
        self.intervention_log.append({
            "timestep": timestep,
            "type": intervention_type,
            "activated": True
        })

    def track_response(self, agent: Any, content: Any, engaged: bool, timestep: int) -> None:
        """
        Records an agent's response to intervention content.
        """
        if not getattr(content, "is_intervention_content", False):
            return
        
        response = {
            "agent_id": agent.id,
            "timestep": timestep,
            "content_id": content.content_id,
            "content_position": content.position_in_belief_space,
            "agent_position": agent.belief_position,
            "distance": abs(content.position_in_belief_space - agent.belief_position),
            "engaged": engaged,
            "arousal_at_time": agent.arousal_level,
        }
        
        if agent.id not in self.agent_response_tracker:
            self.agent_response_tracker[agent.id] = []
        self.agent_response_tracker[agent.id].append(response)

    def get_active_intervention(self) -> Optional[str]:
        """
        Returns the currently active intervention.
        """
        return self.active_intervention

    def calculate_adaptation_rate(self) -> Dict[str, Any]:
        """
        Calculates how agents adapted to the intervention.
        """
        adapted_count = 0
        per_agent = {}
        
        for agent_id, responses in self.agent_response_tracker.items():
            intervention_start = self.config.intervention_start
            early_end = intervention_start + 50
            late_start = intervention_start + 150

            early_responses = [r for r in responses
                               if r["timestep"] < early_end]
            late_responses = [r for r in responses
                              if r["timestep"] >= late_start]
            
            early_rate = sum(1 for r in early_responses if r["engaged"]) / len(early_responses) if early_responses else 0.0
            late_rate = sum(1 for r in late_responses if r["engaged"]) / len(late_responses) if late_responses else 0.0
            
            adaptation_rate = early_rate - late_rate
            adapted = adaptation_rate > 0.2
            
            if adapted:
                adapted_count += 1
                
            per_agent[agent_id] = {
                "early_rate": early_rate,
                "late_rate": late_rate,
                "adaptation_rate": adaptation_rate,
                "adapted": adapted
            }
            
        adapted_fraction = adapted_count / len(self.agent_response_tracker) if self.agent_response_tracker else 0.0
        
        return {
            "adapted_fraction": float(adapted_fraction),
            "per_agent": per_agent
        }
