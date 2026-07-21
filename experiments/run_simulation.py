import random
import numpy as np
from dataclasses import dataclass, field

from config.simulation_config import SimulationConfig, load_default_config
from core.agent import Agent
from data.loader import load_all_pools
from network.multiplex import (
    build_all_networks, calculate_social_influence,
    update_node_belief_positions, rebuild_imdb_network
)
from platforms.imdb import IMDBPlatform
from platforms.reddit import RedditPlatform
from platforms.youtube import YouTubePlatform
from core.platform import PlatformFactory
from measurement.store import MeasurementStore
from measurement.anomaly import detect_anomalies
from interventions.manager import InterventionManager


@dataclass
class SimulationState:
    """
    Holds all live objects that comprise the running simulation.
    """
    agents: list
    content_pools: dict
    networks: dict
    platform_factory: PlatformFactory
    measurement_store: MeasurementStore
    intervention_manager: InterventionManager
    config: SimulationConfig
    current_timestep: int = 0
    research_log: list = field(default_factory=list)
    llm_coordinator: any = None


def create_agents(config: SimulationConfig) -> list:
    """
    Creates all agents according to config.agent_distribution.

    Args:
        config: The simulation configuration.

    Returns:
        List of Agent objects, total length = config.n_agents.
    """
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    agents = []
    agent_id = 0

    for archetype, count in config.agent_distribution.items():
        for _ in range(count):
            agent = Agent(agent_id, archetype)
            agent.belief_position = float(
                np.clip(np.random.normal(5.0, 2.0), 0.0, 10.0)
            )
            agents.append(agent)
            agent_id += 1

    return agents


def initialize_simulation(config: SimulationConfig) -> SimulationState:
    """
    Runs all initialization steps in order and returns a SimulationState.

    Args:
        config: The simulation configuration.

    Returns:
        SimulationState with all components initialized.
    """
    # Step 1: Set random seeds
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Step 2: Create agents
    agents = create_agents(config)

    # Step 3: Load content pools
    content_pools = load_all_pools(config.data_dir, config.latent_dims)

    # Step 4: Build networks
    networks = build_all_networks(agents, config)

    # Step 5: Instantiate platforms
    imdb_platform = IMDBPlatform(config)
    reddit_platform = RedditPlatform(config, networks["reddit"])
    youtube_platform = YouTubePlatform(config)
    platform_factory = PlatformFactory({
        "imdb": imdb_platform,
        "reddit": reddit_platform,
        "youtube": youtube_platform
    })

    # Step 6: Initialize measurement store
    measurement_store = MeasurementStore(
        n_agents=config.n_agents,
        n_timesteps=config.total_timesteps,
        checkpoints=config.checkpoints
    )

    # Step 7: Initialize intervention manager
    intervention_manager = InterventionManager(
        config,
        {"imdb": imdb_platform,
         "reddit": reddit_platform,
         "youtube": youtube_platform},
        networks
    )

    # Step 8: Initialize LLM coordinator and run static mode
    from llm.coordinator import LLMPipelineCoordinator
    llm_coordinator = LLMPipelineCoordinator(config)
    llm_coordinator.run_static_mode(agents)

    state = SimulationState(
        agents=agents,
        content_pools=content_pools,
        networks=networks,
        platform_factory=platform_factory,
        measurement_store=measurement_store,
        intervention_manager=intervention_manager,
        config=config,
    )
    state.llm_coordinator = llm_coordinator
    return state


def run_core_timestep(state: SimulationState, timestep: int) -> None:
    """
    Runs the nine-step interaction loop for every agent in random order.

    Args:
        state: The current simulation state.
        timestep: The current timestep index.
    """
    agent_order = state.agents.copy()
    random.shuffle(agent_order)

    for agent in agent_order:
        # STEP 0: Draw effective parameters
        params = agent.draw_effective_parameters()

        # STEP 1: Select platform and content
        platform = state.platform_factory.get_platform_for_agent(agent, timestep)
        active_intervention = state.intervention_manager.get_active_intervention()
        content = platform.serve_content(
            agent,
            state.content_pools[platform.platform_name],
            timestep,
            intervention=active_intervention
        )
        if content is None:
            continue

        # STEP 2: Engagement decision
        distance = abs(content.position_in_belief_space - agent.belief_position)
        valence_stretch = content.emotional_valence * params["susceptibility"]
        effective_threshold = params["confidence_threshold"] + valence_stretch
        engaged = distance <= effective_threshold

        # STEP 3: Engagement strength
        if engaged:
            engagement_strength = float(np.clip(
                content.engagement_potential
                * (1 - distance / max(effective_threshold, 0.01))
                * params["open_mind"]
                * (content.quality_signal * params["critical_thinking"]
                   + (1 - params["critical_thinking"]))
                * content.novelty,
                0.0, 1.0
            ))
        else:
            engagement_strength = 0.0

        # STEP 4: Belief update from content
        if engaged:
            content_influence = (
                params["update_rate"]
                * engagement_strength
                * (content.position_in_belief_space - agent.belief_position)
            )
        else:
            content_influence = 0.0

        # STEP 5: Emotional state update
        agent.update_emotional_state(content.emotional_valence)

        # Dynamic LLM trigger check
        if (state.config.use_llm and
                state.llm_coordinator is not None):
            state.llm_coordinator.run_dynamic_mode(
                agent, state, timestep
            )

        # STEP 6: Social influence
        social_pull = calculate_social_influence(
            agent, state.networks, state.config,
            confidence_threshold=params["confidence_threshold"]
        )
        social_influence = social_pull - agent.belief_position

        # STEP 7: Final position update
        new_position = (
            agent.belief_position
            + content_influence * platform.algo_weight
            + social_influence * platform.social_weight * 0.1
        )
        agent.belief_position = float(np.clip(new_position, 0.0, 10.0))

        # STEP 8: Algorithm feedback
        platform.learn_from_feedback(agent, content, engaged, engagement_strength)
        agent.interaction_count += 1
        agent.seen_content_ids.add(content.content_id)
        agent.update_history(content.position_in_belief_space)

        if getattr(content, "is_intervention_content", False):
            agent.intervention_content_served += 1
            if engaged:
                agent.intervention_content_engaged += 1

        state.intervention_manager.track_response(agent, content, engaged, timestep)

        # STEP 9: Record measurements
        state.measurement_store.record(agent, state.networks, timestep)
        if not engaged:
            state.measurement_store.record_skip(agent, content, timestep)

        agent.reset_migration_flag()


def run_checkpoint(state: SimulationState, timestep: int) -> None:
    """
    Saves a population snapshot and detects anomalies at a checkpoint.

    Args:
        state: The current simulation state.
        timestep: The checkpoint timestep.
    """
    snapshot = {
        "timestep": timestep,
        "agent_positions": [a.belief_position for a in state.agents],
        "mean_belief": float(np.mean([a.belief_position for a in state.agents])),
        "std_belief": float(np.std([a.belief_position for a in state.agents])),
        "n_agents": len(state.agents),
    }
    state.measurement_store.save_snapshot(snapshot)

    anomalies = detect_anomalies(state.agents, state.measurement_store, timestep)
    if anomalies:
        state.measurement_store.flag_anomalies(anomalies, timestep)

    for platform_name in ["imdb", "reddit", "youtube"]:
        platform = state.platform_factory.get_platform(platform_name)
        platform.record_weight_snapshot(timestep)

    # Sequential LLM checkpoint analysis
    if (state.config.use_llm and
            timestep in state.config.sequential_llm_checkpoints and
            state.llm_coordinator is not None):
        state.llm_coordinator.run_sequential_mode(state, timestep)


def run_simulation(config: SimulationConfig = None) -> SimulationState:
    """
    Runs the complete simulation from initialization to final timestep.

    Args:
        config: The simulation configuration. Uses defaults if None.

    Returns:
        The final SimulationState after all timesteps.
    """
    if config is None:
        config = load_default_config()

    state = initialize_simulation(config)

    for t in range(1, config.total_timesteps + 1):
        state.current_timestep = t

        # Phase routing
        if t <= config.burnin_end:
            run_core_timestep(state, t)
            state.measurement_store.mark_burnin(t)

        elif t <= config.intervention_start:
            run_core_timestep(state, t)

            # New agent entry — 1 agent per 10 timesteps
            if t % 10 == 0 and config.new_agent_entry_rate > 0:
                new_id = len(state.agents)
                new_agent = Agent(new_id, "balanced")
                new_agent.belief_position = float(
                    np.clip(np.random.normal(5.0, 2.0), 0.0, 10.0)
                )
                state.agents.append(new_agent)

            # Reddit migration check
            if t % config.reddit_migration_check_interval == 0:
                reddit_platform = state.platform_factory.get_platform("reddit")
                for agent in state.agents:
                    reddit_platform.check_community_migration(agent, t)

            # IMDB network rebuild
            if t % config.imdb_network_recalc_interval == 0:
                state.networks["imdb"] = rebuild_imdb_network(
                    state.agents, config.imdb_knn_k
                )

        else:  # intervention phase
            if t == config.intervention_start + 1:
                state.intervention_manager.activate(t)
            run_core_timestep(state, t)

        # Checkpoints
        if t in config.checkpoints:
            run_checkpoint(state, t)

        # Sync network node attributes
        update_node_belief_positions(state.agents, state.networks)

        # Progress logging every 100 timesteps
        if t % 100 == 0:
            mean_pos = np.mean([a.belief_position for a in state.agents])
            print(f"Timestep {t} | Mean belief: {mean_pos:.2f} | Agents: {len(state.agents)}")

    print("Simulation complete.")

    # Sync research log from LLM coordinator
    if state.llm_coordinator is not None:
        state.research_log = state.llm_coordinator.research_log

    return state
