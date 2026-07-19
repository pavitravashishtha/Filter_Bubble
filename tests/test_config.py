import pytest
from config.simulation_config import SimulationConfig, load_default_config

def test_load_default_config():
    # 1. load_default_config() returns SimulationConfig without error
    config = load_default_config()
    assert isinstance(config, SimulationConfig)

def test_agent_distribution_sum():
    # 2. agent_distribution values sum to 300
    config = load_default_config()
    assert sum(config.agent_distribution.values()) == 300
    assert config.n_agents == 300

def test_invalid_intervention_type():
    # 3. Invalid intervention_type raises ValueError
    with pytest.raises(ValueError, match="Invalid intervention_type"):
        SimulationConfig(intervention_type="invalid_type")

def test_invalid_agent_distribution():
    # 4. Agent distribution not summing to n_agents raises ValueError
    # Defaults sum to 300, so passing n_agents=400 should trigger the validation error
    with pytest.raises(ValueError, match="Agent distribution values sum to 300, but n_agents is 400"):
        SimulationConfig(n_agents=400)

def test_weights_sum_to_one():
    # 5. algo_weight + social_weight = 1.0 for all three platforms
    config = load_default_config()
    assert abs((config.imdb_algo_weight + config.imdb_social_weight) - 1.0) < 1e-6
    assert abs((config.reddit_algo_weight + config.reddit_social_weight) - 1.0) < 1e-6
    assert abs((config.youtube_algo_weight + config.youtube_social_weight) - 1.0) < 1e-6

def test_invalid_weights_sum():
    with pytest.raises(ValueError, match="IMDB algo and social weights must sum to 1.0"):
        SimulationConfig(imdb_algo_weight=0.5, imdb_social_weight=0.6)
    
    with pytest.raises(ValueError, match="Reddit algo and social weights must sum to 1.0"):
        SimulationConfig(reddit_algo_weight=0.1, reddit_social_weight=0.1)
        
    with pytest.raises(ValueError, match="YouTube algo and social weights must sum to 1.0"):
        SimulationConfig(youtube_algo_weight=0.1, youtube_social_weight=0.1)

def test_to_dict():
    # 6. to_dict() returns a dict with at least 30 keys
    config = load_default_config()
    d = config.to_dict()
    assert isinstance(d, dict)
    assert len(d) >= 30

def test_from_dict():
    # 7. from_dict(config.to_dict()) produces identical config
    original_config = load_default_config()
    # Modify a value to ensure we are testing a real assignment
    original_config.run_id = "test_run"
    
    d = original_config.to_dict()
    new_config = SimulationConfig.from_dict(d)
    
    assert new_config.run_id == "test_run"
    assert new_config.to_dict() == original_config.to_dict()

def test_kwargs_override():
    # 8. Custom values passed as kwargs override defaults correctly
    config = SimulationConfig(
        run_id="override_run", 
        n_agents=100, 
        agent_distribution={
            "heavy_imdb": 100, 
            "heavy_reddit": 0, 
            "heavy_youtube": 0, 
            "balanced": 0, 
            "cross_platform": 0
        }
    )
    assert config.run_id == "override_run"
    assert config.n_agents == 100
    assert config.agent_distribution["heavy_imdb"] == 100

def test_checkpoints_bounds():
    # 9. All checkpoint values are within [1, 1000]
    config = load_default_config()
    for cp in config.checkpoints:
        assert 1 <= cp <= 1000

def test_sequential_llm_checkpoints_subset():
    # 10. sequential_llm_checkpoints is subset of checkpoints
    config = load_default_config()
    assert set(config.sequential_llm_checkpoints).issubset(set(config.checkpoints))
