import pytest
import networkx as nx
import numpy as np

from config.simulation_config import SimulationConfig
from network.multiplex import (
    build_imdb_network,
    build_reddit_network,
    build_youtube_network,
    build_all_networks,
    calculate_social_influence,
    update_node_belief_positions,
    rebuild_imdb_network
)

class MockAgent:
    def __init__(self, agent_id, belief_position):
        self.id = agent_id
        self.belief_position = belief_position
        self.archetype = "balanced"
        self.preference_vector = np.random.normal(0, 1, 25)
        self.primary_community = None
        self.subscribed_communities = []
        self.platform_weights = {"imdb": 0.4, "reddit": 0.3, "youtube": 0.3}

    def draw_effective_parameters(self):
        return {"confidence_threshold": 2.0}

def get_test_agents(n=50):
    return [MockAgent(i, float(i % 10)) for i in range(n)]

def test_build_imdb_network():
    # 1. build_imdb_network() returns graph with correct number of nodes equal to len(agents)
    agents = get_test_agents()
    graph = build_imdb_network(agents, k=5)
    assert len(graph.nodes) == len(agents)

def test_build_reddit_network():
    # 2. build_reddit_network() returns graph with correct number of nodes
    agents = get_test_agents()
    graph = build_reddit_network(agents)
    assert len(graph.nodes) == len(agents)

def test_build_youtube_network():
    # 3. build_youtube_network() returns graph with correct number of nodes
    agents = get_test_agents()
    graph = build_youtube_network(agents)
    assert len(graph.nodes) == len(agents)

def test_reddit_community_assignment():
    # 4 & 5. After build_reddit_network(), every agent has primary_community set and subscribed_communities non-empty
    agents = get_test_agents()
    build_reddit_network(agents, n_communities=5)
    for agent in agents:
        assert agent.primary_community is not None
        assert len(agent.subscribed_communities) > 0

def test_youtube_sparse_graph():
    # 6. build_youtube_network() produces sparse graph — average degree is less than 20 for 50 agents
    agents = get_test_agents(50)
    # connection_probability 0.05 on 50 agents gives an expected degree around 2.5
    graph = build_youtube_network(agents, connection_probability=0.05)
    avg_degree = sum(dict(graph.degree()).values()) / len(agents)
    assert avg_degree < 20

def test_imdb_neighborhoods():
    # 7. build_imdb_network() produces connected neighborhoods — most nodes have at least k/2 neighbors
    agents = get_test_agents(20)
    k = 4
    graph = build_imdb_network(agents, k=k)
    # Count how many nodes have >= k/2 neighbors
    count = sum(1 for node in graph.nodes() if graph.degree(node) >= k/2)
    assert count > len(agents) / 2

def test_belief_position_attribute():
    # 8. All three networks have belief_position attribute on every node
    agents = get_test_agents(10)
    config = SimulationConfig()
    networks = build_all_networks(agents, config)
    
    for graph in networks.values():
        for node in graph.nodes():
            assert "belief_position" in graph.nodes[node]

def test_calculate_social_influence():
    # 9. calculate_social_influence() returns a float within [0.0, 10.0]
    agents = get_test_agents(20)
    config = SimulationConfig()
    networks = build_all_networks(agents, config)
    
    agent = agents[0]
    pull = calculate_social_influence(agent, networks, config)
    assert isinstance(pull, float)
    assert 0.0 <= pull <= 10.0

def test_update_node_belief_positions():
    # 10. update_node_belief_positions() correctly syncs changed belief positions to all three graphs
    agents = get_test_agents(10)
    config = SimulationConfig()
    networks = build_all_networks(agents, config)
    
    agent = agents[0]
    new_belief = 9.99
    agent.belief_position = new_belief
    
    update_node_belief_positions(agents, networks)
    
    for graph in networks.values():
        assert graph.nodes[agent.id]["belief_position"] == new_belief
