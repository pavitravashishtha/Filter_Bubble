import numpy as np
import networkx as nx
import random
from typing import List, Dict, Any

def build_imdb_network(agents: List[Any], k: int = 10) -> nx.Graph:
    """
    Builds a k-nearest neighbors graph based on agent preference_vector cosine similarity.
    
    Args:
        agents: List of agent objects.
        k: Number of nearest neighbors to connect.
        
    Returns:
        A NetworkX graph representing the IMDB network layer.
    """
    graph = nx.Graph()
    for agent in agents:
        graph.add_node(
            agent.id,
            belief_position=agent.belief_position,
            archetype=agent.archetype,
            community_id=None
        )
    
    # Calculate similarities and find nearest neighbors
    for i, agent_a in enumerate(agents):
        similarities = []
        norm_a = np.linalg.norm(agent_a.preference_vector)
        for j, agent_b in enumerate(agents):
            if i == j:
                continue
            
            norm_b = np.linalg.norm(agent_b.preference_vector)
            
            # Cosine similarity formula
            sim = np.dot(agent_a.preference_vector, agent_b.preference_vector) / ((norm_a * norm_b) + 1e-8)
            similarities.append((sim, agent_b.id))
            
        # Sort by similarity descending and pick top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[:k]
        
        for _, neighbor_id in top_k:
            # Undirected graph handles deduplication / bi-directional edges implicitly
            graph.add_edge(agent_a.id, neighbor_id)
            
    return graph

def build_reddit_network(agents: List[Any], n_communities: int = 20,
                         p_within: float = 0.7, p_across: float = 0.02) -> nx.Graph:
    """
    Builds a stochastic block model with dense connections within communities and sparse across.
    
    Args:
        agents: List of agent objects.
        n_communities: Number of communities to divide agents into.
        p_within: Probability of edge between agents in the same community.
        p_across: Probability of edge between agents in different communities.
        
    Returns:
        A NetworkX graph representing the Reddit network layer.
    """
    graph = nx.Graph()
    
    # Step 1: Assign agents to communities
    sorted_agents = sorted(agents, key=lambda a: a.belief_position)
    
    # Divide into n_communities groups of equal size
    n_agents = len(sorted_agents)
    chunk_size = max(1, n_agents // n_communities)
    
    for i, agent in enumerate(sorted_agents):
        community_id = min(i // chunk_size, n_communities - 1)
        agent.primary_community = community_id
        agent.subscribed_communities = [community_id]
        
        graph.add_node(
            agent.id,
            belief_position=agent.belief_position,
            archetype=agent.archetype,
            community_id=agent.primary_community
        )
        
    # Step 2: Build edges
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            agent_a = agents[i]
            agent_b = agents[j]
            
            if agent_a.primary_community == agent_b.primary_community:
                prob = p_within
            else:
                prob = p_across
                
            if random.random() < prob:
                graph.add_edge(agent_a.id, agent_b.id)
                
    return graph

def build_youtube_network(agents: List[Any], connection_probability: float = 0.05) -> nx.Graph:
    """
    Builds a sparse random graph where each pair is connected with low probability.
    
    Args:
        agents: List of agent objects.
        connection_probability: The probability of an edge between any two agents.
        
    Returns:
        A NetworkX graph representing the YouTube network layer.
    """
    graph = nx.Graph()
    for agent in agents:
        graph.add_node(
            agent.id,
            belief_position=agent.belief_position,
            archetype=agent.archetype,
            community_id=None
        )
        
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if random.random() < connection_probability:
                graph.add_edge(agents[i].id, agents[j].id)
                
    return graph

def build_all_networks(agents: List[Any], config: Any) -> Dict[str, nx.Graph]:
    """
    Calls all three network builders using config values.
    
    Args:
        agents: List of agent objects.
        config: Simulation configuration containing network parameters.
        
    Returns:
        A dictionary mapping platform names to their respective NetworkX graphs.
    """
    return {
        "imdb": build_imdb_network(agents, k=config.imdb_knn_k),
        "reddit": build_reddit_network(
            agents, 
            n_communities=config.n_reddit_communities,
            p_within=config.reddit_p_within,
            p_across=config.reddit_p_across
        ),
        "youtube": build_youtube_network(
            agents, 
            connection_probability=config.youtube_subscription_probability
        )
    }

def calculate_social_influence(agent: Any, networks: Dict[str, nx.Graph], config: Any,
                               active_platform_name: str = None,
                               confidence_threshold: float = None) -> float:
    """
    Calculates the total social pull on an agent from all three network layers combined.
    
    Args:
        agent: The agent to calculate influence for.
        networks: Dictionary of the three network layers.
        config: Simulation configuration containing layer weights.
        active_platform_name: Optional name of the active platform context.
        confidence_threshold: Pre-drawn threshold to avoid redundant parameter sampling.
        
    Returns:
        The combined social pull value as a float.
    """
    pulls = {}
    if confidence_threshold is None:
        confidence_threshold = agent.draw_effective_parameters()["confidence_threshold"]
    
    belief = agent.belief_position
    
    for name, graph in networks.items():
        if agent.id not in graph:
            pulls[name] = belief
            continue
            
        neighbors = list(graph.neighbors(agent.id))
        if not neighbors:
            pulls[name] = belief
            continue
        
        neighbor_beliefs = np.array(
            [graph.nodes[n].get("belief_position", 5.0) for n in neighbors]
        )
        mask = np.abs(neighbor_beliefs - belief) <= confidence_threshold
        valid = neighbor_beliefs[mask]
            
        pulls[name] = float(np.mean(valid)) if len(valid) > 0 else belief
        
    # Find active or dominant platform
    if active_platform_name and active_platform_name in config.layer_weights:
        weights = config.layer_weights[active_platform_name]
    else:
        dominant = max(agent.platform_weights, key=agent.platform_weights.get)
        weights = config.layer_weights.get(
            dominant,
            config.layer_weights["balanced"]
        )
    
    total_social_pull = (
        pulls["imdb"] * weights["imdb"] +
        pulls["reddit"] * weights["reddit"] +
        pulls["youtube"] * weights["youtube"]
    )
    
    return float(total_social_pull)


def update_node_belief_positions(agents: List[Any], networks: Dict[str, nx.Graph]) -> None:
    """
    Syncs changed agent belief positions to all three graphs.
    
    Args:
        agents: List of agent objects with updated belief positions.
        networks: Dictionary of network graphs to update.
    """
    for agent in agents:
        for name, graph in networks.items():
            if agent.id in graph.nodes:
                graph.nodes[agent.id]["belief_position"] = agent.belief_position

def rebuild_imdb_network(agents: List[Any], k: int = 10) -> nx.Graph:
    """
    Recalculates knn neighbors based on current taste profiles.
    
    Args:
        agents: List of agent objects.
        k: Number of nearest neighbors.
        
    Returns:
        A new NetworkX graph representing the updated IMDB network layer.
    """
    return build_imdb_network(agents, k)
