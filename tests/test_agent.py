import pytest
import numpy as np
from core.agent import Agent

def test_instantiate_archetypes():
    archetypes = ['heavy_imdb', 'heavy_reddit', 'heavy_youtube', 'balanced', 'cross_platform']
    for arch in archetypes:
        agent = Agent(agent_id=1, archetype=arch)
        assert agent.archetype == arch
        assert agent.agent_id == 1

def test_create_300_agents():
    agents = []
    archetypes = ['heavy_imdb', 'heavy_reddit', 'heavy_youtube', 'balanced', 'cross_platform']
    for i in range(300):
        agents.append(Agent(agent_id=i, archetype=np.random.choice(archetypes)))
    assert len(agents) == 300

def test_belief_position_distribution():
    agents = [Agent(i, 'balanced') for i in range(300)]
    beliefs = [a.base_belief_position for a in agents]
    mean_belief = np.mean(beliefs)
    std_belief = np.std(beliefs)
    assert 4.0 <= mean_belief <= 6.0
    assert 1.5 <= std_belief <= 2.5

def test_effective_parameters_stochasticity():
    agent = Agent(1, 'balanced')
    params1 = agent.draw_effective_parameters()
    params2 = agent.draw_effective_parameters()
    assert params1 != params2

def test_effective_parameters_bounds():
    agent = Agent(1, 'heavy_youtube')
    agent.arousal_level = 1.0 # Maximize modifiers
    params = agent.draw_effective_parameters()
    for key, val in params.items():
        assert 0.0 <= val <= 1.0
        
def test_update_emotional_state_high_valence():
    agent = Agent(1, 'balanced')
    initial_arousal = agent.arousal_level
    agent.update_emotional_state(0.9)
    assert agent.arousal_level > initial_arousal
    assert agent.consecutive_high_valence_count == 1
    
def test_update_emotional_state_low_valence():
    agent = Agent(1, 'balanced')
    agent.arousal_level = 0.5
    agent.current_cognitive_load = 0.5
    agent.consecutive_high_valence_count = 5
    agent.update_emotional_state(0.3)
    assert agent.arousal_level < 0.5
    assert agent.consecutive_high_valence_count == 0

def test_arousal_never_exceeds_1():
    agent = Agent(1, 'balanced')
    for _ in range(100):
        agent.update_emotional_state(0.9)
    assert agent.arousal_level == 1.0

def test_get_state_name():
    agent = Agent(1, 'balanced')
    
    agent.arousal_level = 0.1
    assert agent.get_state_name() == "CALM"
    
    agent.arousal_level = 0.5
    assert agent.get_state_name() == "ENGAGED"
    
    agent.arousal_level = 0.7
    assert agent.get_state_name() == "AGITATED"
    
    agent.arousal_level = 0.9
    assert agent.get_state_name() == "REACTIVE"
    
def test_community_just_migrated_initialization():
    agent = Agent(1, 'balanced')
    assert agent.community_just_migrated is False
    agent.community_just_migrated = True
    agent.reset_migration_flag()
    assert agent.community_just_migrated is False
