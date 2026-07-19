import numpy as np
from typing import Dict, Any, List

from measurement.metrics import (
    calc_belief_position,
    calc_content_diversity_score,
    calc_belief_drift_rate,
    calc_echo_chamber_index,
    calc_exposure_bias,
    calc_intervention_response_rate
)
from measurement.anomaly import detect_anomalies, Anomaly

class MeasurementStore:
    """
    Central storage and measurement for all simulation metrics.
    """
    def __init__(self, n_agents: int, n_timesteps: int, checkpoints: List[int]):
        """
        Initializes metric arrays.
        """
        self.belief_positions = np.zeros((n_agents, n_timesteps))
        self.diversity_scores = np.zeros((n_agents, n_timesteps))
        self.drift_rates = np.zeros((n_agents, n_timesteps))
        self.echo_chamber_index = np.zeros((n_agents, n_timesteps))
        self.exposure_bias = np.zeros((n_agents, n_timesteps))
        self.intervention_response = np.zeros((n_agents, n_timesteps))
        
        self.n_agents = n_agents
        self.n_timesteps = n_timesteps
        self.checkpoints = checkpoints
        self.burnin_end = 100
        self.snapshots: Dict[int, Any] = {}
        self.skip_log: List[Dict[str, Any]] = []
        self.anomaly_log: List[Anomaly] = []

    def record(self, agent: Any, networks: Dict[str, Any], timestep: int) -> None:
        """
        Records all six metrics for one agent at one timestep.
        """
        idx = agent.id
        self._ensure_capacity(idx)
        tidx = timestep - 1
        
        self.belief_positions[idx, tidx] = calc_belief_position(agent)
        self.diversity_scores[idx, tidx] = calc_content_diversity_score(agent)
        self.drift_rates[idx, tidx] = calc_belief_drift_rate(agent)
        self.echo_chamber_index[idx, tidx] = calc_echo_chamber_index(agent, networks)
        self.exposure_bias[idx, tidx] = calc_exposure_bias(agent)
        self.intervention_response[idx, tidx] = calc_intervention_response_rate(agent)

    def record_skip(self, agent: Any, content: Any, timestep: int) -> None:
        """
        Records an interaction skip event.
        """
        self.skip_log.append({
            "agent_id": agent.id,
            "content_position": content.position_in_belief_space,
            "agent_position": agent.belief_position,
            "timestep": timestep,
            "is_intervention": getattr(content, 'is_intervention_content', False)
        })

    def save_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Saves a snapshot state dict at a specific timestep.
        """
        self.snapshots[snapshot["timestep"]] = snapshot

    def flag_anomalies(self, anomalies: List[Anomaly], timestep: int) -> None:
        """
        Adds detected anomalies to the anomaly log.
        """
        for anomaly in anomalies:
            self.anomaly_log.append(anomaly)

    def get_latest_metrics(self, agent_id: int) -> Dict[str, float]:
        """
        Returns the most recently recorded metric values for this agent.
        """
        self._ensure_capacity(agent_id)
        # Find the last non-zero column index for this agent (or just the latest recorded one).
        # We can find this by looking at belief_positions which shouldn't be identically 0.0 everywhere.
        # Alternatively, we can find the last non-zero index by checking where belief_positions != 0
        # However, 0.0 is a valid belief_position. A more robust way is to find the last index where it was modified.
        # But we'll just check the last non-zero or take the maximum non-zero index across some metric that is mostly non-zero.
        # Let's find the last index by checking any metric that is non-zero, e.g. diversity score or belief position.
        
        # A simpler way: we can just find the last non-zero in belief_positions. If it's all zeros, use index 0.
        row = self.belief_positions[agent_id, :]
        non_zero_indices = np.nonzero(row)[0]
        if len(non_zero_indices) > 0:
            last_idx = non_zero_indices[-1]
        else:
            # Maybe the belief position is exactly 0.0. Let's check another array.
            if np.any(self.diversity_scores[agent_id, :]):
                last_idx = np.nonzero(self.diversity_scores[agent_id, :])[0][-1]
            else:
                last_idx = 0
                
        return {
            "diversity_score": float(self.diversity_scores[agent_id, last_idx]),
            "drift_rate": float(self.drift_rates[agent_id, last_idx]),
            "exposure_bias": float(self.exposure_bias[agent_id, last_idx]),
            "belief_position": float(self.belief_positions[agent_id, last_idx]),
            "echo_chamber_index": float(self.echo_chamber_index[agent_id, last_idx]),
            "intervention_response": float(self.intervention_response[agent_id, last_idx])
        }

    def get_metric_at(self, agent_id: int, metric_name: str, timestep: int) -> float:
        """
        Returns value of named metric for agent at specific timestep.
        """
        self._ensure_capacity(agent_id)
        tidx = timestep - 1
        metric_arrays = {
            "belief_position": self.belief_positions,
            "diversity_score": self.diversity_scores,
            "drift_rate": self.drift_rates,
            "echo_chamber_index": self.echo_chamber_index,
            "exposure_bias": self.exposure_bias,
            "intervention_response": self.intervention_response
        }
        return float(metric_arrays[metric_name][agent_id, tidx])

    def mark_burnin(self, timestep: int) -> None:
        """
        Placeholder method for explicit burn-in handling.
        """
        pass

    def get_analysis_data(self, exclude_burnin: bool = True) -> Dict[str, Any]:
        """
        Returns all arrays sliced to exclude burn-in if requested.
        """
        start = self.burnin_end if exclude_burnin else 0
        return {
            "belief_positions": self.belief_positions[:, start:],
            "diversity_scores": self.diversity_scores[:, start:],
            "drift_rates": self.drift_rates[:, start:],
            "echo_chamber_index": self.echo_chamber_index[:, start:],
            "exposure_bias": self.exposure_bias[:, start:],
            "intervention_response": self.intervention_response[:, start:],
            "snapshots": self.snapshots,
            "anomaly_log": self.anomaly_log,
            "skip_log": self.skip_log
        }

    def _ensure_capacity(self, agent_id: int) -> None:
        """
        Dynamically resizes arrays to accommodate new agent IDs added during simulation.
        """
        current_capacity = self.belief_positions.shape[0]
        if agent_id >= current_capacity:
            new_capacity = agent_id + 1
            n_timesteps = self.belief_positions.shape[1]
            extra_rows = new_capacity - current_capacity
            
            self.belief_positions = np.vstack([self.belief_positions, np.zeros((extra_rows, n_timesteps))])
            self.diversity_scores = np.vstack([self.diversity_scores, np.zeros((extra_rows, n_timesteps))])
            self.drift_rates = np.vstack([self.drift_rates, np.zeros((extra_rows, n_timesteps))])
            self.echo_chamber_index = np.vstack([self.echo_chamber_index, np.zeros((extra_rows, n_timesteps))])
            self.exposure_bias = np.vstack([self.exposure_bias, np.zeros((extra_rows, n_timesteps))])
            self.intervention_response = np.vstack([self.intervention_response, np.zeros((extra_rows, n_timesteps))])
