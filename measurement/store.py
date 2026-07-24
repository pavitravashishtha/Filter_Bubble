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
    
    Optimizations implemented:
    1. Chunked Allocation: Instead of growing the underlying arrays by exactly 1 row (which
       causes O(N) copies on every single new agent addition), we grow arrays by a fixed chunk
       size (e.g. 256 rows). This reduces the frequency of reallocations to O(N / chunk_size).
    2. float32 Data Type: Metric arrays are stored as float32 instead of float64, reducing
       memory footprints by 50% without affecting the required precision for research metrics.
    3. Property Views: Properties are exposed for all public arrays, returning a sliced view
       up to the logical number of agents (`_active_agents`). This ensures external code and
       experiments get identical shapes and contents as they did in the unoptimized version.
    """
    def __init__(self, n_agents: int, n_timesteps: int, checkpoints: List[int]):
        """
        Initializes metric arrays.
        """
        self._chunk_size = 256
        self._active_agents = n_agents
        
        # Initialize arrays with float32 to reduce memory footprint.
        # We start with the capacity equal to the initial number of agents.
        physical_capacity = n_agents
        
        self._belief_positions = np.zeros((physical_capacity, n_timesteps), dtype=np.float32)
        self._diversity_scores = np.zeros((physical_capacity, n_timesteps), dtype=np.float32)
        self._drift_rates = np.zeros((physical_capacity, n_timesteps), dtype=np.float32)
        self._echo_chamber_index = np.zeros((physical_capacity, n_timesteps), dtype=np.float32)
        self._exposure_bias = np.zeros((physical_capacity, n_timesteps), dtype=np.float32)
        self._intervention_response = np.zeros((physical_capacity, n_timesteps), dtype=np.float32)
        self._last_recorded_timestep = np.full(physical_capacity, -1, dtype=np.int32)
        
        self.n_agents = n_agents
        self.n_timesteps = n_timesteps
        self.checkpoints = checkpoints
        self.burnin_end = 100
        self.snapshots: Dict[int, Any] = {}
        self.skip_log: List[Dict[str, Any]] = []
        self.anomaly_log: List[Anomaly] = []

    @property
    def belief_positions(self) -> np.ndarray:
        return self._belief_positions[:self._active_agents]

    @belief_positions.setter
    def belief_positions(self, value: np.ndarray) -> None:
        self._belief_positions = value
        self._active_agents = value.shape[0]

    @property
    def diversity_scores(self) -> np.ndarray:
        return self._diversity_scores[:self._active_agents]

    @diversity_scores.setter
    def diversity_scores(self, value: np.ndarray) -> None:
        self._diversity_scores = value
        self._active_agents = value.shape[0]

    @property
    def drift_rates(self) -> np.ndarray:
        return self._drift_rates[:self._active_agents]

    @drift_rates.setter
    def drift_rates(self, value: np.ndarray) -> None:
        self._drift_rates = value
        self._active_agents = value.shape[0]

    @property
    def echo_chamber_index(self) -> np.ndarray:
        return self._echo_chamber_index[:self._active_agents]

    @echo_chamber_index.setter
    def echo_chamber_index(self, value: np.ndarray) -> None:
        self._echo_chamber_index = value
        self._active_agents = value.shape[0]

    @property
    def exposure_bias(self) -> np.ndarray:
        return self._exposure_bias[:self._active_agents]

    @exposure_bias.setter
    def exposure_bias(self, value: np.ndarray) -> None:
        self._exposure_bias = value
        self._active_agents = value.shape[0]

    @property
    def intervention_response(self) -> np.ndarray:
        return self._intervention_response[:self._active_agents]

    @intervention_response.setter
    def intervention_response(self, value: np.ndarray) -> None:
        self._intervention_response = value
        self._active_agents = value.shape[0]

    @property
    def last_recorded_timestep(self) -> np.ndarray:
        return self._last_recorded_timestep[:self._active_agents]

    @last_recorded_timestep.setter
    def last_recorded_timestep(self, value: np.ndarray) -> None:
        self._last_recorded_timestep = value
        self._active_agents = value.shape[0]

    def record(self, agent: Any, networks: Dict[str, Any], timestep: int) -> None:
        """
        Records all six metrics for one agent at one timestep.
        """
        idx = agent.id
        self._ensure_capacity(idx)
        tidx = timestep - 1
        
        self._belief_positions[idx, tidx] = calc_belief_position(agent)
        self._diversity_scores[idx, tidx] = calc_content_diversity_score(agent)
        self._drift_rates[idx, tidx] = calc_belief_drift_rate(agent)
        self._echo_chamber_index[idx, tidx] = calc_echo_chamber_index(agent, networks)
        self._exposure_bias[idx, tidx] = calc_exposure_bias(agent)
        self._intervention_response[idx, tidx] = calc_intervention_response_rate(agent)
        self._last_recorded_timestep[idx] = tidx

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

    def get_latest_metrics(self, agent_id: int) -> dict:
        self._ensure_capacity(agent_id)
        tidx = self._last_recorded_timestep[agent_id]
        if tidx < 0:
            return {
                "diversity_score": 0.5,
                "drift_rate": 0.0,
                "exposure_bias": 0.5,
                "belief_position": 5.0,
                "echo_chamber_index": 0.0,
                "intervention_response": 0.0
            }
        return {
            "diversity_score": float(
                self._diversity_scores[agent_id, tidx]
            ),
            "drift_rate": float(
                self._drift_rates[agent_id, tidx]
            ),
            "exposure_bias": float(
                self._exposure_bias[agent_id, tidx]
            ),
            "belief_position": float(
                self._belief_positions[agent_id, tidx]
            ),
            "echo_chamber_index": float(
                self._echo_chamber_index[agent_id, tidx]
            ),
            "intervention_response": float(
                self._intervention_response[agent_id, tidx]
            )
        }

    def get_metric_at(self, agent_id: int, metric_name: str, timestep: int) -> float:
        """
        Returns value of named metric for agent at specific timestep.
        """
        self._ensure_capacity(agent_id)
        tidx = timestep - 1
        metric_arrays = {
            "belief_position": self._belief_positions,
            "diversity_score": self._diversity_scores,
            "drift_rate": self._drift_rates,
            "echo_chamber_index": self._echo_chamber_index,
            "exposure_bias": self._exposure_bias,
            "intervention_response": self._intervention_response
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

    def _resize_metric_array(self, arr: np.ndarray, new_capacity: int, n_timesteps: int, fill_value: float) -> np.ndarray:
        """
        Helper function that resizes a metric array consistently to a new capacity.
        Preserves all existing data during expansion.
        """
        new_arr = np.full((new_capacity, n_timesteps), fill_value, dtype=arr.dtype)
        new_arr[:arr.shape[0], :] = arr
        return new_arr

    def _resize_last_recorded_array(self, arr: np.ndarray, new_capacity: int, fill_value: int) -> np.ndarray:
        """
        Helper function that resizes the last_recorded_timestep array consistently.
        Preserves all existing data during expansion.
        """
        new_arr = np.full(new_capacity, fill_value, dtype=arr.dtype)
        new_arr[:arr.shape[0]] = arr
        return new_arr

    def _ensure_capacity(self, agent_id: int) -> None:
        """
        Dynamically resizes arrays to accommodate new agent IDs added during simulation.
        Uses chunked growth strategy to minimize costly copies.
        """
        current_capacity = self._belief_positions.shape[0]
        if agent_id >= current_capacity:
            # Calculate the required capacity, growing by at least self._chunk_size.
            # Round up to a multiple of self._chunk_size.
            growth = max(self._chunk_size, agent_id - current_capacity + 1)
            growth = ((growth + self._chunk_size - 1) // self._chunk_size) * self._chunk_size
            new_capacity = current_capacity + growth
            n_timesteps = self._belief_positions.shape[1]
            
            # Reallocate and copy data using the helper functions.
            self._belief_positions = self._resize_metric_array(self._belief_positions, new_capacity, n_timesteps, 0.0)
            self._diversity_scores = self._resize_metric_array(self._diversity_scores, new_capacity, n_timesteps, 0.0)
            self._drift_rates = self._resize_metric_array(self._drift_rates, new_capacity, n_timesteps, 0.0)
            self._echo_chamber_index = self._resize_metric_array(self._echo_chamber_index, new_capacity, n_timesteps, 0.0)
            self._exposure_bias = self._resize_metric_array(self._exposure_bias, new_capacity, n_timesteps, 0.0)
            self._intervention_response = self._resize_metric_array(self._intervention_response, new_capacity, n_timesteps, 0.0)
            self._last_recorded_timestep = self._resize_last_recorded_array(self._last_recorded_timestep, new_capacity, -1)
            
        if agent_id >= self._active_agents:
            self._active_agents = agent_id + 1
