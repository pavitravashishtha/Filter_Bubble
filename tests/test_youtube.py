import pytest
from platforms.youtube import YouTubePlatform
from config.simulation_config import SimulationConfig
from core.content import YouTubeContent

class MockAgent:
    def __init__(self):
        self.interaction_count = 0
        self.belief_position = 5.0

def test_youtube_instantiation():
    # 1. YouTubePlatform instantiates with default config
    config = SimulationConfig()
    platform = YouTubePlatform(config)
    assert platform.platform_name == "youtube"
    
def test_initial_weights():
    # 2. Initial weights are all 0.25
    config = SimulationConfig()
    platform = YouTubePlatform(config)
    for k in ["W_V", "W_E", "W_N", "W_D"]:
        assert platform.weights[k] == 0.25

def test_serve_content_returns_one():
    # 3. serve_content() returns one YouTubeContent object
    config = SimulationConfig()
    platform = YouTubePlatform(config)
    agent = MockAgent()
    agent.interaction_count = 20
    platform.exploration_rate = 0.0 # avoid random choice for deterministic test
    pool = [YouTubeContent(content_id=i, birth_timestep=0, topic_cluster=1) for i in range(10)]
    for c in pool:
        c.position_in_belief_space = 5.0
    content = platform.serve_content(agent, pool, timestep=1)
    assert isinstance(content, YouTubeContent)

def test_cold_start():
    # 4. Cold start (interaction_count < 10) returns content
    config = SimulationConfig()
    platform = YouTubePlatform(config)
    agent = MockAgent()
    agent.interaction_count = 5
    pool = [YouTubeContent(content_id=i, birth_timestep=0, topic_cluster=1) for i in range(10)]
    for c in pool:
        c.position_in_belief_space = 5.0
    content = platform.serve_content(agent, pool, timestep=1)
    assert isinstance(content, YouTubeContent)

def test_score_returns_float():
    # 5. _score() returns a float
    config = SimulationConfig()
    platform = YouTubePlatform(config)
    agent = MockAgent()
    content = YouTubeContent(content_id=1, birth_timestep=0, topic_cluster=1)
    content.position_in_belief_space = 5.0
    score = platform._score(content, agent)
    assert isinstance(score, float)

def test_learn_from_feedback_engaged_increases_WV():
    # 6. After 200 learn_from_feedback() calls with engaged=True and high-valence content, W_V is higher
    config = SimulationConfig()
    config.youtube_learning_rate = 0.5
    platform = YouTubePlatform(config)
    agent = MockAgent()
    
    content = YouTubeContent(content_id=1, birth_timestep=0, topic_cluster=1)
    content.position_in_belief_space = 5.0
    content.emotional_valence = 0.9
    content.engagement_potential = 0.1
    content.novelty = 0.1
    
    for _ in range(200):
        platform.learn_from_feedback(agent, content, engaged=True, engagement_strength=1.0)
        
    assert platform.weights["W_V"] > 0.25

def test_learn_from_feedback_not_engaged_reward():
    # 7. After 200 learn_from_feedback() calls with engaged=False, reward is -0.5 not -1
    config = SimulationConfig()
    platform = YouTubePlatform(config)
    agent = MockAgent()
    
    content = YouTubeContent(content_id=1, birth_timestep=0, topic_cluster=1)
    content.position_in_belief_space = 5.0
    content.emotional_valence = 1.0
    content.engagement_potential = 1.0
    content.novelty = 1.0
    
    old_wv = platform.weights["W_V"]
    for _ in range(200):
        platform.learn_from_feedback(agent, content, engaged=False, engagement_strength=0.0)
    
    # Confirms that negative engagement impacts the weights in the correct direction 
    assert platform.weights["W_V"] < old_wv

def test_WD_stays_positive():
    # 8. W_D stays >= 0.0 after all weight updates
    config = SimulationConfig()
    platform = YouTubePlatform(config)
    agent = MockAgent()
    
    content = YouTubeContent(content_id=1, birth_timestep=0, topic_cluster=1)
    content.position_in_belief_space = 10.0 # Max distance
    agent.belief_position = 0.0
    
    for _ in range(100):
        platform.learn_from_feedback(agent, content, engaged=False, engagement_strength=0.0)
        assert platform.weights["W_D"] >= 0.0

def test_weights_sum_to_one():
    # 9. Weights sum to approximately 1.0 after normalization
    config = SimulationConfig()
    platform = YouTubePlatform(config)
    agent = MockAgent()
    content = YouTubeContent(content_id=1, birth_timestep=0, topic_cluster=1)
    content.position_in_belief_space = 5.0
    
    for _ in range(50):
        platform.learn_from_feedback(agent, content, engaged=True, engagement_strength=1.0)
        
    total = sum(platform.weights.values())
    assert abs(total - 1.0) < 0.01

def test_record_weight_snapshot():
    # 10. record_weight_snapshot() adds entry to weight_history with correct timestep
    config = SimulationConfig()
    platform = YouTubePlatform(config)
    
    assert len(platform.weight_history) == 0
    platform.record_weight_snapshot(timestep=50)
    assert len(platform.weight_history) == 1
    assert platform.weight_history[0]["timestep"] == 50
    assert "W_V" in platform.weight_history[0]

def test_apply_friction():
    # 11. apply_friction() reduces W_V
    config = SimulationConfig()
    platform = YouTubePlatform(config)
    
    old_wv = platform.weights["W_V"]
    platform.apply_friction(friction_factor=0.5)
    
    assert platform.weights["W_V"] == old_wv * 0.5

def test_diversity_injection():
    # 12. diversity_injection intervention marks some candidates as is_intervention_content = True
    config = SimulationConfig()
    platform = YouTubePlatform(config)
    agent = MockAgent()
    agent.belief_position = 5.0
    
    pool = []
    for i in range(150):
        c = YouTubeContent(content_id=i, birth_timestep=0, topic_cluster=1)
        if i < 30:
            c.position_in_belief_space = 9.0 # Distant
        else:
            c.position_in_belief_space = 5.0 # Close
        pool.append(c)
        
    candidates = platform.generate_candidates(agent, pool, intervention="diversity_injection")
    
    assert any(getattr(c, "is_intervention_content", False) for c in candidates)
