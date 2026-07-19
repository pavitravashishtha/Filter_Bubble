import pytest
import numpy as np
from platforms.imdb import IMDBPlatform
from config.simulation_config import SimulationConfig
from core.content import IMDBContent

class MockAgent:
    def __init__(self):
        self.interaction_count = 0
        self.seen_content_ids = set()
        self.belief_position = 5.0
        self.preference_vector = np.zeros(25)
        self.user_bias = 0.0
        
    def draw_effective_parameters(self):
        return {"update_rate": 0.5}

def test_imdb_instantiation():
    # 1. IMDBPlatform instantiates correctly with default config
    config = SimulationConfig()
    platform = IMDBPlatform(config)
    assert platform.platform_name == "imdb"
    assert platform.learning_rate == config.imdb_learning_rate

def test_serve_content_returns_one():
    # 2. serve_content() returns one IMDBContent object
    config = SimulationConfig()
    platform = IMDBPlatform(config)
    agent = MockAgent()
    pool = [IMDBContent(content_id=i, birth_timestep=0) for i in range(50)]
    for c in pool:
        c.latent_features = np.zeros(25)
        c.ideological_loading_vector = np.zeros(25)
        c.item_bias = 0.0
        
    content = platform.serve_content(agent, pool, timestep=1)
    assert isinstance(content, IMDBContent)

def test_serve_content_never_none():
    # 3. serve_content() never returns None when pool has items
    config = SimulationConfig()
    platform = IMDBPlatform(config)
    agent = MockAgent()
    pool = [IMDBContent(content_id=i, birth_timestep=0) for i in range(1)]
    for c in pool:
        c.latent_features = np.zeros(25)
        c.ideological_loading_vector = np.zeros(25)
        c.item_bias = 0.0
    
    # Even if seen, if there are no candidates, it falls back to random.choice(content_pool)
    agent.seen_content_ids.add(0)
    agent.interaction_count = 15
    content = platform.serve_content(agent, pool, timestep=1)
    assert content is not None
    assert content.content_id == 0

def test_cold_start():
    # 4. Cold start (interaction_count < 10) still returns content
    config = SimulationConfig()
    platform = IMDBPlatform(config)
    agent = MockAgent()
    agent.interaction_count = 5
    pool = [IMDBContent(content_id=i, birth_timestep=0) for i in range(10)]
    for c in pool:
        c.latent_features = np.zeros(25)
        c.ideological_loading_vector = np.zeros(25)
        c.item_bias = 0.0
        
    candidates = platform.generate_candidates(agent, pool)
    assert len(candidates) > 0
    assert len(candidates) <= 200

def test_predict_rating():
    # 5. _predict_rating() returns a float
    config = SimulationConfig()
    platform = IMDBPlatform(config)
    agent = MockAgent()
    content = IMDBContent(content_id=1, birth_timestep=0)
    content.latent_features = np.ones(25)
    content.item_bias = 0.1
    rating = platform._predict_rating(agent, content)
    assert isinstance(rating, float)

def test_learn_from_feedback_preference_update():
    # 6. learn_from_feedback() changes agent.preference_vector
    config = SimulationConfig()
    platform = IMDBPlatform(config)
    agent = MockAgent()
    content = IMDBContent(content_id=1, birth_timestep=0)
    content.latent_features = np.ones(25)
    content.ideological_loading_vector = np.ones(25)
    content.item_bias = 0.0
    
    old_vector = agent.preference_vector.copy()
    platform.learn_from_feedback(agent, content, engaged=True, engagement_strength=1.0)
    
    assert not np.array_equal(old_vector, agent.preference_vector)

def test_learn_from_feedback_ideology_bleed():
    # 7. learn_from_feedback() changes agent.belief_position slightly over 50 calls
    config = SimulationConfig()
    platform = IMDBPlatform(config)
    agent = MockAgent()
    agent.preference_vector = np.ones(25) * 0.1 # Need non-zero preference to bleed
    old_belief = agent.belief_position
    
    content = IMDBContent(content_id=1, birth_timestep=0)
    content.latent_features = np.ones(25)
    content.ideological_loading_vector = np.ones(25) * 0.5
    content.item_bias = 0.0
    
    for _ in range(50):
        platform.learn_from_feedback(agent, content, engaged=True, engagement_strength=1.0)
        
    assert agent.belief_position != old_belief

def test_belief_position_bounds():
    # 8. agent.belief_position stays within [0.0, 10.0] after 1000 learn_from_feedback() calls
    config = SimulationConfig()
    platform = IMDBPlatform(config)
    agent = MockAgent()
    agent.preference_vector = np.ones(25)
    agent.belief_position = 9.9
    
    content = IMDBContent(content_id=1, birth_timestep=0)
    content.latent_features = np.ones(25)
    content.ideological_loading_vector = np.ones(25) * 1.0
    content.item_bias = 0.0
    
    for _ in range(1000):
        platform.learn_from_feedback(agent, content, engaged=True, engagement_strength=5.0)
        
    assert 0.0 <= agent.belief_position <= 10.0

def test_diversity_injection():
    # 9. diversity_injection intervention causes some returned candidates to have is_intervention_content = True
    config = SimulationConfig()
    platform = IMDBPlatform(config)
    agent = MockAgent()
    agent.interaction_count = 20
    agent.belief_position = 5.0
    
    pool = []
    for i in range(250):
        c = IMDBContent(content_id=i, birth_timestep=0)
        # Make exactly some far away
        if i < 50:
            c.position_in_belief_space = 9.0 # distance = 4.0 (>3.0)
        else:
            c.position_in_belief_space = 5.0 # distance = 0.0
        c.latent_features = np.zeros(25)
        c.ideological_loading_vector = np.zeros(25)
        pool.append(c)
        
    candidates = platform.generate_candidates(agent, pool, intervention="diversity_injection")
    
    # Check that at least one candidate has is_intervention_content = True
    assert any(c.is_intervention_content for c in candidates)

def test_weight_snapshot():
    # 10. weight_history has one entry after record_weight_snapshot()
    config = SimulationConfig()
    platform = IMDBPlatform(config)
    
    assert len(platform.weight_history) == 0
    platform.record_weight_snapshot(timestep=100)
    
    assert len(platform.weight_history) == 1
    assert platform.weight_history[0]["timestep"] == 100
    assert platform.weight_history[0]["global_mean_rating"] == config.imdb_global_mean_rating
    assert platform.weight_history[0]["learning_rate"] == config.imdb_learning_rate
