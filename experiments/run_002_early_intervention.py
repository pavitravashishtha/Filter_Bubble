import sys
import os
# Allow running as both `python experiments/run_002_early_intervention.py` and `python -m experiments.run_002_early_intervention`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
from config.simulation_config import SimulationConfig
from experiments.run_simulation import run_simulation

def run_full_research_simulation():
    config = SimulationConfig(
        run_id = "research_run_002",
        random_seed = 42,
        total_timesteps = 1000,
        burnin_end = 100,
        intervention_start = 400,  # changed from 800 — early intervention
        checkpoints = [100, 200, 400, 600, 800, 900, 1000],
        sequential_llm_checkpoints = [400, 800, 1000],
        use_llm = False,
        n_agents = 300,
        agent_distribution = {
            "heavy_imdb": 60,
            "heavy_reddit": 60,
            "heavy_youtube": 60,
            "balanced": 80,
            "cross_platform": 40
        },
        intervention_type = "diversity_injection",
        intervention_strength = 0.2,
        new_agent_entry_rate = 1,
    )
    
    print("=== FILTER BUBBLE SIMULATION — RESEARCH RUN 002 ===")
    print(f"Agents: {config.n_agents}")
    print(f"Timesteps: {config.total_timesteps}")
    print(f"Intervention: {config.intervention_type} "
          f"at timestep {config.intervention_start}")
    print(f"Random seed: {config.random_seed}")
    print("===================================================\n")
    
    start_time = time.time()
    state = run_simulation(config)
    elapsed = time.time() - start_time
    
    # ── RESULTS ANALYSIS ──
    agents = state.agents
    store = state.measurement_store
    
    belief_positions = [a.belief_position for a in agents]
    mean_final = float(np.mean(belief_positions))
    std_final = float(np.std(belief_positions))
    
    # Belief position at key checkpoints
    checkpoint_means = {}
    for t in config.checkpoints:
        tidx = t - 1
        if tidx < store.belief_positions.shape[1]:
            checkpoint_means[t] = float(
                np.mean(store.belief_positions[:config.n_agents, tidx])
            )
    
    # YouTube weight evolution
    youtube = state.platform_factory.get_platform("youtube")
    weight_snapshots = youtube.weight_history
    
    if weight_snapshots:
        first_snapshot = weight_snapshots[0]
        last_snapshot = weight_snapshots[-1]
        wv_change = last_snapshot["W_V"] - first_snapshot["W_V"]
        wd_change = last_snapshot["W_D"] - first_snapshot["W_D"]
    else:
        wv_change = 0.0
        wd_change = 0.0
    
    # Bubble formation — agents with high exposure bias
    if store.belief_positions.shape[1] >= 1000:
        early_bias = float(np.mean(
            store.exposure_bias[:config.n_agents, 199]
        ))
        late_bias = float(np.mean(
            store.exposure_bias[:config.n_agents, 799]
        ))
        intervention_bias = float(np.mean(
            store.exposure_bias[:config.n_agents, 999]
        ))
    else:
        early_bias = late_bias = intervention_bias = 0.0
    
    # Intervention adaptation
    adaptation = (
        state.intervention_manager.calculate_adaptation_rate()
    )
    adapted_fraction = adaptation.get("adapted_fraction", 0.0)
    
    # Community distribution (Reddit)
    community_counts = {}
    for agent in agents:
        if agent.primary_community is not None:
            c = agent.primary_community
            community_counts[c] = community_counts.get(c, 0) + 1
    
    # ── PRINT FULL RESULTS ──
    print("\n=== SIMULATION COMPLETE ===")
    print(f"Total runtime: {elapsed:.1f} seconds "
          f"({elapsed/60:.1f} minutes)")
    print(f"Final agent count: {len(agents)}")
    print(f"\n-- BELIEF POSITIONS --")
    print(f"Final mean:  {mean_final:.3f} (started ~5.0)")
    print(f"Final std:   {std_final:.3f}")
    print(f"\nCheckpoint means:")
    for t, mean in checkpoint_means.items():
        label = ""
        if t == 100:  label = " <- burn-in end"
        if t == 400:  label = " <- intervention start"
        if t == 800:  label = " <- pre-final"
        if t == 1000: label = " <- final"
        print(f"  Timestep {t:4d}: {mean:.3f}{label}")
    
    print(f"\n-- BUBBLE FORMATION --")
    print(f"Exposure bias at t=200:  {early_bias:.3f} "
          f"(early simulation)")
    print(f"Exposure bias at t=800:  {late_bias:.3f} "
          f"(pre-intervention)")
    print(f"Exposure bias at t=1000: {intervention_bias:.3f} "
          f"(post-intervention)")
    bias_change = late_bias - early_bias
    intervention_effect = intervention_bias - late_bias
    print(f"Bubble formation rate:   {bias_change:+.3f} "
          f"(positive = bubbles forming)")
    print(f"Intervention effect:     {intervention_effect:+.3f} "
          f"(negative = intervention working)")
    
    print(f"\n-- YOUTUBE RL WEIGHTS --")
    if weight_snapshots:
        print(f"W_V (valence):    "
              f"{first_snapshot['W_V']:.3f} -> "
              f"{last_snapshot['W_V']:.3f} "
              f"({wv_change:+.3f})")
        print(f"W_E (engagement): "
              f"{first_snapshot['W_E']:.3f} -> "
              f"{last_snapshot['W_E']:.3f}")
        print(f"W_N (novelty):    "
              f"{first_snapshot['W_N']:.3f} -> "
              f"{last_snapshot['W_N']:.3f}")
        print(f"W_D (distance):   "
              f"{first_snapshot['W_D']:.3f} -> "
              f"{last_snapshot['W_D']:.3f} "
              f"({wd_change:+.3f})")
    
    print(f"\n-- INTERVENTION --")
    print(f"Type: {config.intervention_type}")
    print(f"Agents tracked: "
          f"{len(state.intervention_manager.agent_response_tracker)}")
    print(f"Adapted fraction: {adapted_fraction:.1%} "
          f"of agents learned to skip intervention content")
    
    print(f"\n-- ANOMALIES AND EVENTS --")
    print(f"Total anomalies detected: "
          f"{len(store.anomaly_log)}")
    anomaly_types = {}
    for a in store.anomaly_log:
        t = a.anomaly_type
        anomaly_types[t] = anomaly_types.get(t, 0) + 1
    for atype, count in anomaly_types.items():
        print(f"  {atype}: {count}")
    print(f"Total skip events: {len(store.skip_log)}")
    
    print(f"\n-- REDDIT COMMUNITIES --")
    print(f"Communities active: {len(community_counts)}")
    if community_counts:
        sizes = list(community_counts.values())
        print(f"Largest community: {max(sizes)} agents")
        print(f"Smallest community: {min(sizes)} agents")
    
    print(f"\n-- SNAPSHOTS SAVED --")
    for t, snap in store.snapshots.items():
        print(f"  t={t}: mean={snap['mean_belief']:.3f}, "
              f"std={snap['std_belief']:.3f}, "
              f"n={snap['n_agents']}")
    
    print("\n===========================")
    print("Run complete. Data ready for dashboard.")
    
    # Save results summary to JSON for dashboard use
    results_summary = {
        "run_id": config.run_id,
        "elapsed_seconds": elapsed,
        "final_n_agents": len(agents),
        "final_mean_belief": mean_final,
        "final_std_belief": std_final,
        "checkpoint_means": checkpoint_means,
        "youtube_wv_start": (
            first_snapshot["W_V"] if weight_snapshots else 0.25
        ),
        "youtube_wv_end": (
            last_snapshot["W_V"] if weight_snapshots else 0.25
        ),
        "early_exposure_bias": early_bias,
        "late_exposure_bias": late_bias,
        "post_intervention_bias": intervention_bias,
        "adapted_fraction": adapted_fraction,
        "total_anomalies": len(store.anomaly_log),
        "anomaly_types": anomaly_types,
        "total_skips": len(store.skip_log),
        "n_communities": len(community_counts),
        "intervention_type": config.intervention_type,
        "intervention_start": config.intervention_start,
    }
    
    with open("experiments/run_002_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print("Summary saved to experiments/run_002_summary.json")
    
    full_results = save_full_results(state, config, elapsed)
    return state


def save_full_results(state, config, elapsed):
    import json
    import numpy as np
    
    store = state.measurement_store
    agents = state.agents
    youtube = state.platform_factory.get_platform("youtube")
    
    # Belief trajectories — population mean per timestep
    # Shape: (n_timesteps,)
    mean_trajectory = []
    std_trajectory = []
    for t in range(config.total_timesteps):
        positions = store.belief_positions[:len(agents), t]
        positions = positions[positions != 0]
        if len(positions) > 0:
            mean_trajectory.append(float(np.mean(positions)))
            std_trajectory.append(float(np.std(positions)))
        else:
            mean_trajectory.append(5.0)
            std_trajectory.append(0.0)
    
    # Per-agent final state
    agent_finals = []
    for agent in agents:
        agent_finals.append({
            "id": agent.id,
            "archetype": agent.archetype,
            "final_belief": agent.belief_position,
            "interaction_count": agent.interaction_count,
            "primary_community": agent.primary_community,
        })
    
    # YouTube weight history
    weight_history = youtube.weight_history
    
    # Metric trajectories — population mean per timestep
    diversity_trajectory = []
    echo_trajectory = []
    exposure_trajectory = []
    for t in range(config.total_timesteps):
        div = store.diversity_scores[:len(agents), t]
        echo = store.echo_chamber_index[:len(agents), t]
        exp = store.exposure_bias[:len(agents), t]
        diversity_trajectory.append(float(np.mean(div)))
        echo_trajectory.append(float(np.mean(echo)))
        exposure_trajectory.append(float(np.mean(exp)))
    
    # Anomaly log
    anomaly_data = []
    for a in store.anomaly_log:
        anomaly_data.append({
            "agent_id": a.agent_id,
            "type": a.anomaly_type,
            "timestep": a.timestep,
            "belief_position": a.agent_belief_position,
            "arousal": a.agent_arousal_level,
        })
    
    # Archetype breakdown
    archetype_finals = {}
    for agent in agents:
        arch = agent.archetype
        if arch not in archetype_finals:
            archetype_finals[arch] = []
        archetype_finals[arch].append(agent.belief_position)
    archetype_means = {
        k: float(np.mean(v)) 
        for k, v in archetype_finals.items()
    }
    
    full_results = {
        "run_id": config.run_id,
        "elapsed_seconds": elapsed,
        "config": {
            "n_agents": config.n_agents,
            "total_timesteps": config.total_timesteps,
            "intervention_type": config.intervention_type,
            "intervention_start": config.intervention_start,
            "random_seed": config.random_seed,
        },
        "mean_trajectory": mean_trajectory,
        "std_trajectory": std_trajectory,
        "diversity_trajectory": diversity_trajectory,
        "echo_trajectory": echo_trajectory,
        "exposure_trajectory": exposure_trajectory,
        "weight_history": weight_history,
        "agent_finals": agent_finals,
        "archetype_means": archetype_means,
        "anomaly_log": anomaly_data,
        "snapshots": {
            str(k): v 
            for k, v in store.snapshots.items()
        },
        "intervention_adaptation": (
            state.intervention_manager
            .calculate_adaptation_rate()
        ),
        "skip_count": len(store.skip_log),
        "total_anomalies": len(store.anomaly_log),
    }
    
    path = f"experiments/{config.run_id}_full_results.json"
    with open(path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"Full results saved to {path}")
    
    return full_results

if __name__ == "__main__":
    run_full_research_simulation()
