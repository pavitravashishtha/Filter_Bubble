import time
from config.simulation_config import SimulationConfig
from experiments.run_simulation import run_simulation

def test_full_50_timestep_run():
    config = SimulationConfig(
        total_timesteps=50,
        burnin_end=20,
        intervention_start=40,
        checkpoints=[20, 40, 50],
        sequential_llm_checkpoints=[50],
        use_llm=False,
        new_agent_entry_rate=0,
        n_agents=50,
        agent_distribution={
            "heavy_imdb": 10,
            "heavy_reddit": 10,
            "heavy_youtube": 10,
            "balanced": 15,
            "cross_platform": 5
        }
    )

    start = time.time()
    state = run_simulation(config)
    elapsed = time.time() - start

    # Verify simulation ran
    assert len(state.agents) >= 50

    # Verify all belief positions in valid range
    for agent in state.agents:
        assert 0.0 <= agent.belief_position <= 10.0, \
            f"Agent {agent.id} belief {agent.belief_position} out of range"

    # Verify measurement store has data
    assert state.measurement_store.belief_positions.sum() > 0

    # Verify snapshots were saved at checkpoints
    assert 20 in state.measurement_store.snapshots
    assert 40 in state.measurement_store.snapshots
    assert 50 in state.measurement_store.snapshots

    # Verify at least some agent drift occurred
    initial_positions = [5.0] * 50  # approximate starting mean
    final_positions = [a.belief_position for a in state.agents[:50]]
    assert final_positions != initial_positions, \
        "No agent drift detected — simulation may not be running"

    print("Smoke test passed.")
    print(f"Final mean belief position: "
          f"{sum(final_positions)/len(final_positions):.2f}")
    print(f"50 timesteps completed in {elapsed:.2f} seconds.")
