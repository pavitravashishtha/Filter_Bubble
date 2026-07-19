import pytest
import numpy as np
import networkx as nx

from measurement.store import MeasurementStore
from measurement.metrics import (
    calc_belief_position,
    calc_content_diversity_score,
    calc_belief_drift_rate,
    calc_echo_chamber_index,
    calc_exposure_bias,
    calc_intervention_response_rate
)
from measurement.anomaly import detect_anomalies

class MockAgent:
    def __init__(self, agent_id=0):
        self.id = agent_id
        self.belief_position = 5.0
        self.content_position_history = []
        self.belief_position_history = []
        self.intervention_content_served = 0
        self.intervention_content_engaged = 0
        self.arousal_level = 0.5

class MockContent:
    def __init__(self, pos):
        self.position_in_belief_space = pos
        self.is_intervention_content = False

def test_measurement_store_initialization():
    # 1. MeasurementStore initializes with correct array shapes
    store = MeasurementStore(n_agents=10, n_timesteps=100, checkpoints=[50, 100])
    assert store.belief_positions.shape == (10, 100)
    assert store.diversity_scores.shape == (10, 100)
    assert store.drift_rates.shape == (10, 100)

def test_record_fills_correct_positions():
    # 2. record() fills correct array positions for agent 0 at timestep 1 and agent 5 at timestep 50
    store = MeasurementStore(n_agents=10, n_timesteps=100, checkpoints=[])
    
    agent0 = MockAgent(0)
    agent0.belief_position = 2.0
    
    agent5 = MockAgent(5)
    agent5.belief_position = 8.0
    
    networks = {"imdb": nx.Graph(), "reddit": nx.Graph(), "youtube": nx.Graph()}
    networks["imdb"].add_node(0)
    networks["imdb"].add_node(5)
    networks["reddit"].add_node(0)
    networks["reddit"].add_node(5)
    networks["youtube"].add_node(0)
    networks["youtube"].add_node(5)
    
    store.record(agent0, networks, timestep=1)
    store.record(agent5, networks, timestep=50)
    
    assert store.belief_positions[0, 0] == 2.0
    assert store.belief_positions[5, 49] == 8.0

def test_record_skip():
    # 3. record_skip() appends to skip_log
    store = MeasurementStore(n_agents=10, n_timesteps=100, checkpoints=[])
    agent = MockAgent(1)
    content = MockContent(5.0)
    store.record_skip(agent, content, timestep=10)
    
    assert len(store.skip_log) == 1
    assert store.skip_log[0]["agent_id"] == 1
    assert store.skip_log[0]["timestep"] == 10

def test_save_snapshot():
    # 4. save_snapshot() stores snapshot at correct timestep key
    store = MeasurementStore(10, 100, [])
    snapshot = {"timestep": 50, "data": "test"}
    store.save_snapshot(snapshot)
    
    assert 50 in store.snapshots
    assert store.snapshots[50]["data"] == "test"

def test_get_latest_metrics():
    # 5. get_latest_metrics() returns dict with all six keys
    store = MeasurementStore(10, 100, [])
    agent = MockAgent(0)
    networks = {"imdb": nx.Graph(), "reddit": nx.Graph(), "youtube": nx.Graph()}
    networks["imdb"].add_node(0)
    networks["reddit"].add_node(0)
    networks["youtube"].add_node(0)
    
    store.record(agent, networks, timestep=10)
    metrics = store.get_latest_metrics(0)
    
    expected_keys = {"diversity_score", "drift_rate", "exposure_bias", 
                     "belief_position", "echo_chamber_index", "intervention_response"}
    assert set(metrics.keys()) == expected_keys

def test_get_metric_at():
    # 6. get_metric_at() returns correct value at specified timestep
    store = MeasurementStore(10, 100, [])
    store.belief_positions[2, 49] = 7.5
    val = store.get_metric_at(agent_id=2, metric_name="belief_position", timestep=50)
    assert val == 7.5

def test_get_analysis_data():
    # 7. get_analysis_data(exclude_burnin=True) returns arrays starting from column 100
    store = MeasurementStore(10, 200, [])
    store.burnin_end = 100
    data = store.get_analysis_data(exclude_burnin=True)
    
    assert data["belief_positions"].shape == (10, 100) # 200 - 100 = 100 columns

def test_calc_diversity_score_few_items():
    # 8. calc_content_diversity_score() returns 0.5 when agent has fewer than 2 items in history
    agent = MockAgent()
    agent.content_position_history = [5.0]
    assert calc_content_diversity_score(agent) == 0.5

def test_calc_exposure_bias_high():
    # 9. calc_exposure_bias() returns high value when all content is within 2.0 of agent belief position
    agent = MockAgent()
    agent.belief_position = 5.0
    agent.content_position_history = [4.0, 5.5, 6.0, 4.5] # All within 2.0 of 5.0
    assert calc_exposure_bias(agent) == 1.0

def test_calc_echo_chamber_index():
    # 10. calc_echo_chamber_index() returns value between 0.0 and 1.0
    agent = MockAgent(0)
    networks = {"imdb": nx.Graph(), "reddit": nx.Graph(), "youtube": nx.Graph()}
    for g in networks.values():
        g.add_node(0, belief_position=5.0)
        g.add_node(1, belief_position=6.0)
        g.add_edge(0, 1)
        
    index = calc_echo_chamber_index(agent, networks)
    assert 0.0 <= index <= 1.0

def test_detect_anomalies_empty():
    # 11. detect_anomalies() returns empty list when no conditions are triggered
    store = MeasurementStore(10, 100, [])
    agent = MockAgent(0)
    
    # Needs a baseline so latest metrics can be returned as defaults (all safe)
    anomalies = detect_anomalies([agent], store, timestep=10)
    assert len(anomalies) == 0

def test_detect_anomalies_rapid_radicalization():
    # 12. detect_anomalies() returns Anomaly with type rapid_radicalization when drift_rate > 0.8
    store = MeasurementStore(10, 100, [])
    store.drift_rates[0, 9] = 0.9 # drift rate > 0.8 at tidx=9 (timestep=10)
    store.belief_positions[0, 9] = 1.0 # ensure it's found as latest metric
    
    agent = MockAgent(0)
    anomalies = detect_anomalies([agent], store, timestep=10)
    
    assert len(anomalies) > 0
    assert any(a.anomaly_type == "rapid_radicalization" for a in anomalies)
