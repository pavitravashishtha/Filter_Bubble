import pytest
import numpy as np

from config.simulation_config import SimulationConfig
from experiments.run_simulation import (
    create_agents,
    initialize_simulation,
    run_core_timestep,
    run_checkpoint,
    SimulationState,
)


def get_small_config() -> SimulationConfig:
    """Returns a minimal config for fast testing."""
    config = SimulationConfig()
    # Scale everything down for fast test runs
    config.n_agents = 10
    config.agent_distribution = {
        "heavy_imdb": 2,
        "heavy_reddit": 2,
        "heavy_youtube": 2,
        "balanced": 2,
        "cross_platform": 2
    }
    config.total_timesteps = 10
    config.burnin_end = 5
    config.intervention_start = 8
    config.checkpoints = [5, 10]
    config.imdb_knn_k = 3
    config.n_reddit_communities = 2
    config.imdb_network_recalc_interval = 5
    config.reddit_migration_check_interval = 5
    return config


def test_initialize_completes():
    # 1. initialize_simulation() completes without error
    config = get_small_config()
    state = initialize_simulation(config)
    assert state is not None


def test_correct_agent_count():
    # 2. initialize_simulation() returns SimulationState with correct number of agents (10)
    config = get_small_config()
    state = initialize_simulation(config)
    assert len(state.agents) == 10


def test_all_content_pools_loaded():
    # 3. initialize_simulation() returns state with all three content pools loaded
    config = get_small_config()
    state = initialize_simulation(config)
    assert "imdb" in state.content_pools
    assert "reddit" in state.content_pools
    assert "youtube" in state.content_pools
    for key, pool in state.content_pools.items():
        assert len(pool) > 0, f"Content pool '{key}' is empty"


def test_all_networks_built():
    # 4. initialize_simulation() returns state with all three networks built
    config = get_small_config()
    state = initialize_simulation(config)
    assert "imdb" in state.networks
    assert "reddit" in state.networks
    assert "youtube" in state.networks
    for key, g in state.networks.items():
        assert g.number_of_nodes() > 0, f"Network '{key}' has no nodes"


def test_run_core_timestep_no_error():
    # 5. run_core_timestep() completes for all agents without error at timestep 1
    config = get_small_config()
    state = initialize_simulation(config)
    run_core_timestep(state, timestep=1)  # Should not raise


def test_run_core_timestep_updates_beliefs():
    # 6. run_core_timestep() updates at least some agent belief positions
    config = get_small_config()
    state = initialize_simulation(config)
    initial_positions = [a.belief_position for a in state.agents]
    
    for t in range(1, 6):
        run_core_timestep(state, timestep=t)
    
    final_positions = [a.belief_position for a in state.agents]
    # At least one agent should have moved
    assert initial_positions != final_positions


def test_run_checkpoint_saves_snapshot():
    # 7. run_checkpoint() saves snapshot to measurement_store
    config = get_small_config()
    state = initialize_simulation(config)
    run_core_timestep(state, timestep=1)
    run_checkpoint(state, timestep=1)
    assert 1 in state.measurement_store.snapshots
    assert state.measurement_store.snapshots[1]["n_agents"] == 10


def test_10_timesteps_no_error():
    # 8. Running 10 timesteps completes without error
    config = get_small_config()
    state = initialize_simulation(config)
    for t in range(1, 11):
        state.current_timestep = t
        run_core_timestep(state, t)
        if t in config.checkpoints:
            run_checkpoint(state, t)
        from network.multiplex import update_node_belief_positions
        update_node_belief_positions(state.agents, state.networks)


def test_belief_positions_in_range():
    # 9. All agent belief positions stay within [0.0, 10.0] after 10 timesteps
    config = get_small_config()
    state = initialize_simulation(config)
    for t in range(1, 11):
        run_core_timestep(state, t)
    for agent in state.agents:
        assert 0.0 <= agent.belief_position <= 10.0, (
            f"Agent {agent.id} belief_position {agent.belief_position} out of range"
        )


def test_measurement_store_has_nonzero_values():
    # 10. measurement_store has non-zero values after 10 timesteps of recording
    config = get_small_config()
    state = initialize_simulation(config)
    for t in range(1, 11):
        run_core_timestep(state, t)
    
    # belief_positions should be non-zero for at least some agents / timesteps
    assert np.any(state.measurement_store.belief_positions != 0), \
        "belief_positions array is all zeros after 10 timesteps"
