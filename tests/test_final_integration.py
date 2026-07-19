from config.simulation_config import SimulationConfig
from experiments.run_simulation import run_simulation

def test_full_integration_no_llm():
    config = SimulationConfig(
        total_timesteps=100,
        burnin_end=20,
        intervention_start=80,
        checkpoints=[20, 50, 80, 100],
        sequential_llm_checkpoints=[50, 100],
        use_llm=False,
        new_agent_entry_rate=1,
        n_agents=100,
        agent_distribution={
            "heavy_imdb": 20,
            "heavy_reddit": 20,
            "heavy_youtube": 20,
            "balanced": 25,
            "cross_platform": 15
        },
        intervention_type="diversity_injection"
    )

    state = run_simulation(config)

    # Core assertions
    assert len(state.agents) >= 100
    for agent in state.agents:
        assert 0.0 <= agent.belief_position <= 10.0

    # Measurement assertions
    assert state.measurement_store.belief_positions.sum() > 0
    assert len(state.measurement_store.snapshots) == 4

    # LLM coordinator wired in
    assert state.llm_coordinator is not None
    assert len(state.llm_coordinator.research_log) >= 1

    # Intervention activated at timestep 81
    assert (state.intervention_manager.get_active_intervention()
            == "diversity_injection")

    # Platform weight history recorded
    youtube = state.platform_factory.get_platform("youtube")
    assert len(youtube.weight_history) > 0

    # YouTube W_V has changed from starting value of 0.25
    # (RL learning happened)
    final_wv = youtube.weight_history[-1]["W_V"]
    assert final_wv != 0.25, "YouTube RL weights did not update"

    # Content diversity — agents should have history
    for agent in state.agents[:10]:
        assert len(agent.belief_position_history) > 0
        assert len(agent.content_position_history) > 0

    # Skip log has entries (some agents rejected content)
    assert len(state.measurement_store.skip_log) > 0

    print("\n=== INTEGRATION TEST RESULTS ===")
    print(f"Agents: {len(state.agents)}")
    print(f"Final mean belief: {sum(a.belief_position for a in state.agents)/len(state.agents):.3f}")
    print(f"Snapshots saved: {list(state.measurement_store.snapshots.keys())}")
    print(f"LLM research log entries: {len(state.llm_coordinator.research_log)}")
    print(f"YouTube W_V final: {final_wv:.4f}")
    print(f"Skip events recorded: {len(state.measurement_store.skip_log)}")
    print(f"Anomalies detected: {len(state.measurement_store.anomaly_log)}")
    print(f"Intervention responses tracked: {len(state.intervention_manager.agent_response_tracker)}")
    print("=================================")
