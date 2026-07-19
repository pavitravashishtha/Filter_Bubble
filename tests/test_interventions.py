import pytest
import networkx as nx

from config.simulation_config import SimulationConfig
from interventions.manager import InterventionManager
from interventions.social_bridge import apply_social_bridges

class MockAgent:
    def __init__(self, agent_id=0):
        self.id = agent_id
        self.belief_position = 5.0
        self.arousal_level = 0.5

class MockContent:
    def __init__(self, content_id=0):
        self.content_id = content_id
        self.position_in_belief_space = 5.0
        self.is_intervention_content = False

class MockYouTube:
    def __init__(self):
        self.friction_applied = False
        
    def apply_friction(self, factor):
        self.friction_applied = True
        self.factor = factor

def test_manager_instantiation():
    # 1. InterventionManager instantiates correctly
    config = SimulationConfig()
    manager = InterventionManager(config, {}, {})
    assert manager.active_intervention is None
    assert len(manager.intervention_log) == 0

def test_activate_null():
    # 2. activate() with "null" runs without error
    config = SimulationConfig()
    config.intervention_type = "null"
    manager = InterventionManager(config, {}, {})
    manager.activate(timestep=10)
    assert manager.active_intervention == "null"
    assert manager.intervention_log[-1]["type"] == "null"

def test_activate_diversity():
    # 3. activate() with "diversity_injection" sets config.active_intervention correctly
    config = SimulationConfig()
    config.intervention_type = "diversity_injection"
    manager = InterventionManager(config, {}, {})
    manager.activate(timestep=20)
    assert config.active_intervention == "diversity_injection"

def test_activate_friction():
    # 4. activate() with "algorithm_friction" reduces YouTube W_V weight
    config = SimulationConfig()
    config.intervention_type = "algorithm_friction"
    youtube = MockYouTube()
    manager = InterventionManager(config, {"youtube": youtube}, {})
    manager.activate(timestep=30)
    assert youtube.friction_applied is True

def test_activate_logs():
    # 5. activate() logs to intervention_log
    config = SimulationConfig()
    config.intervention_type = "null"
    manager = InterventionManager(config, {}, {})
    manager.activate(timestep=40)
    assert len(manager.intervention_log) == 1
    assert manager.intervention_log[0]["timestep"] == 40
    assert manager.intervention_log[0]["activated"] is True

def test_track_response_ignores_non_intervention():
    # 6. track_response() ignores non-intervention content
    config = SimulationConfig()
    manager = InterventionManager(config, {}, {})
    agent = MockAgent(0)
    content = MockContent(0) # is_intervention_content is False
    manager.track_response(agent, content, engaged=True, timestep=10)
    assert 0 not in manager.agent_response_tracker

def test_track_response_records():
    # 7. track_response() records intervention content response
    config = SimulationConfig()
    manager = InterventionManager(config, {}, {})
    agent = MockAgent(1)
    content = MockContent(1)
    content.is_intervention_content = True
    manager.track_response(agent, content, engaged=True, timestep=10)
    assert 1 in manager.agent_response_tracker
    assert len(manager.agent_response_tracker[1]) == 1
    assert manager.agent_response_tracker[1][0]["engaged"] is True

def test_calculate_adaptation_rate():
    # 8. calculate_adaptation_rate() returns dict with adapted_fraction key
    config = SimulationConfig()
    manager = InterventionManager(config, {}, {})
    
    agent = MockAgent(0)
    
    # Early responses (timestep < 850)
    c1 = MockContent(1)
    c1.is_intervention_content = True
    manager.track_response(agent, c1, engaged=True, timestep=800)
    manager.track_response(agent, c1, engaged=True, timestep=810)
    
    # Late responses (timestep >= 950)
    c2 = MockContent(2)
    c2.is_intervention_content = True
    manager.track_response(agent, c2, engaged=False, timestep=960)
    manager.track_response(agent, c2, engaged=False, timestep=970)
    
    # early_rate = 1.0, late_rate = 0.0 -> adaptation_rate = 1.0 -> adapted = True
    stats = manager.calculate_adaptation_rate()
    assert "adapted_fraction" in stats
    assert stats["adapted_fraction"] == 1.0 # 1 out of 1 agent adapted

def test_apply_social_bridges():
    # 9. apply_social_bridges() adds edges to Reddit network between different communities
    # 10. apply_social_bridges() marks added edges with is_bridge=True attribute
    g = nx.Graph()
    g.add_node(1, community_id=0)
    g.add_node(2, community_id=0)
    g.add_node(3, community_id=1)
    g.add_node(4, community_id=1)
    
    # Initially no edges
    networks = {"reddit": g}
    
    apply_social_bridges(networks, [], bridge_weight=0.5)
    
    # Check if a bridge edge was added
    # Since there are two communities (0 and 1), it should pick one random node from 0 and one from 1
    assert g.number_of_edges() == 1
    
    # Check edge attributes
    u, v = list(g.edges())[0]
    edge_data = g.get_edge_data(u, v)
    
    assert edge_data["is_bridge"] is True
    assert edge_data["weight"] == 0.5
    
    # Verify they are across different communities
    assert g.nodes[u]["community_id"] != g.nodes[v]["community_id"]
