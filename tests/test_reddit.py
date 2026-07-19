import pytest
import math
import networkx as nx
from platforms.reddit import RedditPlatform
from config.simulation_config import SimulationConfig
from core.content import RedditContent

class MockAgent:
    def __init__(self):
        self.subscribed_communities = [1]
        self.primary_community = 1
        self.belief_position = 5.0
        self.community_just_migrated = False
        
    def draw_effective_parameters(self):
        return {"confidence_threshold": 2.0}

def get_test_graph():
    g = nx.Graph()
    g.add_node(1, community_id=1, belief_position=5.0)
    g.add_node(2, community_id=1, belief_position=5.0)
    g.add_node(3, community_id=2, belief_position=8.0)
    g.add_node(4, community_id=2, belief_position=8.0)
    return g

def test_reddit_instantiation():
    # 1. RedditPlatform instantiates with a NetworkX graph
    config = SimulationConfig()
    g = nx.Graph()
    platform = RedditPlatform(config, g)
    assert platform.platform_name == "reddit"
    assert isinstance(platform.graph, nx.Graph)

def test_serve_content_returns_one():
    # 2. serve_content() returns one RedditContent object
    config = SimulationConfig()
    platform = RedditPlatform(config, get_test_graph())
    agent = MockAgent()
    pool = [RedditContent(content_id=i, birth_timestep=0, community_id=1) for i in range(10)]
    content = platform.serve_content(agent, pool, timestep=1)
    assert isinstance(content, RedditContent)

def test_serve_content_empty_community():
    # 3. serve_content() returns a safe result even with empty community (falls back to random)
    config = SimulationConfig()
    platform = RedditPlatform(config, get_test_graph())
    agent = MockAgent()
    agent.subscribed_communities = [99] # No items in this community
    
    pool = [RedditContent(content_id=i, birth_timestep=0, community_id=1) for i in range(10)]
    content = platform.serve_content(agent, pool, timestep=1)
    assert isinstance(content, RedditContent)
    assert content in pool

def test_hot_score_age_penalty():
    # 4. _hot_score() returns lower score for older content at same vote count
    config = SimulationConfig()
    platform = RedditPlatform(config, get_test_graph())
    c1 = RedditContent(content_id=1, birth_timestep=0, community_id=1)
    c1.upvotes = 100
    c1.downvotes = 10
    c1.timestamp = 10
    
    c2 = RedditContent(content_id=2, birth_timestep=0, community_id=1)
    c2.upvotes = 100
    c2.downvotes = 10
    c2.timestamp = 0 # Older
    
    score1 = platform._hot_score(c1, timestep=100)
    score2 = platform._hot_score(c2, timestep=100)
    assert score2 < score1

def test_wilson_score_penalty():
    # 5. _wilson_score() returns lower score for content with 2 up 0 down than 1000 up 0 down
    config = SimulationConfig()
    platform = RedditPlatform(config, get_test_graph())
    
    c1 = RedditContent(content_id=1, birth_timestep=0, community_id=1)
    c1.upvotes = 2
    c1.downvotes = 0
    
    c2 = RedditContent(content_id=2, birth_timestep=0, community_id=1)
    c2.upvotes = 1000
    c2.downvotes = 0
    
    score1 = platform._wilson_score(c1)
    score2 = platform._wilson_score(c2)
    assert score1 < score2

def test_ideological_alignment_equal():
    # 6. _ideological_alignment() returns 1.0 when content position equals community mean
    config = SimulationConfig()
    platform = RedditPlatform(config, get_test_graph())
    agent = MockAgent()
    
    c = RedditContent(content_id=1, birth_timestep=0, community_id=1)
    c.position_in_belief_space = 5.0 # matches community mean of 5.0
    
    alignment = platform._ideological_alignment(c, agent)
    assert abs(alignment - 1.0) < 1e-6

def test_ideological_alignment_distant():
    # 7. _ideological_alignment() returns 0.0 when distance is 10
    config = SimulationConfig()
    platform = RedditPlatform(config, nx.Graph())
    agent = MockAgent() # MockAgent has no members in empty graph, mean falls back to 5.0
    
    c = RedditContent(content_id=1, birth_timestep=0, community_id=1)
    c.position_in_belief_space = -5.0 # Distance = | -5.0 - 5.0 | = 10.0
    
    alignment = platform._ideological_alignment(c, agent)
    assert abs(alignment - 0.0) < 1e-6

def test_learn_from_feedback_increment_upvotes():
    # 8. learn_from_feedback() increments upvotes when engaged=True
    config = SimulationConfig()
    platform = RedditPlatform(config, get_test_graph())
    agent = MockAgent()
    
    c = RedditContent(content_id=1, birth_timestep=0, community_id=1)
    initial_upvotes = c.upvotes
    
    platform.learn_from_feedback(agent, c, engaged=True, engagement_strength=1.0)
    assert c.upvotes == initial_upvotes + 1

def test_learn_from_feedback_increment_downvotes():
    # 9. learn_from_feedback() increments downvotes when engaged=False
    config = SimulationConfig()
    platform = RedditPlatform(config, get_test_graph())
    agent = MockAgent()
    
    c = RedditContent(content_id=1, birth_timestep=0, community_id=1)
    initial_downvotes = c.downvotes
    
    platform.learn_from_feedback(agent, c, engaged=False, engagement_strength=0.0)
    assert c.downvotes == initial_downvotes + 1

def test_check_community_migration():
    # 10. check_community_migration() sets community_just_migrated=True when agent drifted far
    config = SimulationConfig()
    platform = RedditPlatform(config, get_test_graph())
    agent = MockAgent()
    
    # Community 1 mean = 5.0
    # Threshold = 2.0
    agent.belief_position = 8.0 # Drift = |8.0 - 5.0| = 3.0 > 2.0 -> migrate
    
    # Community 2 mean = 8.0 (perfect match)
    platform.check_community_migration(agent, timestep=1)
    
    assert agent.community_just_migrated is True
    assert agent.primary_community == 2
    assert 2 in agent.subscribed_communities
    assert 1 not in agent.subscribed_communities
