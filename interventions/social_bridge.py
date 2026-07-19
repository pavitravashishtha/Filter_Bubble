import random
import networkx as nx
from typing import Dict, Any, List

def apply_social_bridges(networks: Dict[str, nx.Graph], agents: List[Any], bridge_weight: float = 0.1) -> None:
    """
    Adds cross-community edges to Reddit network layer.
    """
    reddit_graph = networks.get("reddit")
    if reddit_graph is None:
        return
        
    communities = set()
    for node, data in reddit_graph.nodes(data=True):
        comm_id = data.get("community_id")
        if comm_id is not None:
            communities.add(comm_id)
            
    communities_list = list(communities)
    
    for i in range(len(communities_list)):
        for j in range(i + 1, len(communities_list)):
            comm_i = communities_list[i]
            comm_j = communities_list[j]
            
            # Find agents in these communities
            agents_i = [n for n, d in reddit_graph.nodes(data=True) if d.get("community_id") == comm_i]
            agents_j = [n for n, d in reddit_graph.nodes(data=True) if d.get("community_id") == comm_j]
            
            if not agents_i or not agents_j:
                continue
                
            agent_i = random.choice(agents_i)
            agent_j = random.choice(agents_j)
            
            if not reddit_graph.has_edge(agent_i, agent_j):
                reddit_graph.add_edge(agent_i, agent_j, weight=bridge_weight, is_bridge=True)

def get_bridge_description() -> str:
    """
    Returns description string of what this intervention does.
    """
    return "Creates social bridges by randomly connecting individuals across different communities."
