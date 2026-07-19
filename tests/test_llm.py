import pytest
import numpy as np
import networkx as nx

from config.simulation_config import SimulationConfig
from llm.static_llm import run_static_llm
from llm.dynamic_llm import run_dynamic_llm, build_dynamic_context
from llm.sequential_llm import run_sequential_llm, build_trajectory_summary
from llm.coordinator import LLMPipelineCoordinator
from measurement.store import MeasurementStore
from core.platform import PlatformFactory


# ── Shared mock helpers ────────────────────────────────────────────────────────

class MockAgent:
    def __init__(self, agent_id=0, archetype="balanced"):
        self.id = agent_id
        self.archetype = archetype
        self.belief_position = 5.0
        self.arousal_level = 0.2          # below trigger threshold (0.8)
        self.interaction_count = 10
        self.belief_position_history = [5.0] * 5
        self.content_position_history = [5.0] * 5
        self.platform_weights = {"imdb": 0.33, "reddit": 0.33, "youtube": 0.34}
        self.primary_community = 1
        self.community_just_migrated = False

    def draw_effective_parameters(self):
        return {"confidence_threshold": 0.5}


class MockPlatform:
    """Minimal stub for a platform inside PlatformFactory."""
    def __init__(self, name):
        self.platform_name = name
        self.algo_weight = 0.6
        self.social_weight = 0.4
        self.weight_history = []

    def record_weight_snapshot(self, t): pass
    def serve_content(self, *a, **kw): return None
    def learn_from_feedback(self, *a, **kw): pass
    def generate_candidates(self, *a, **kw): return []


class MockState:
    """Minimal SimulationState stub for LLM tests."""
    def __init__(self):
        self.agents = [MockAgent(i) for i in range(10)]
        self.measurement_store = MeasurementStore(
            n_agents=10, n_timesteps=50, checkpoints=[]
        )
        # Seed some non-zero belief data so trajectory summary has real numbers
        for i in range(10):
            self.measurement_store.belief_positions[i, 0] = 5.0 + i * 0.1

        platforms = {n: MockPlatform(n) for n in ("imdb", "reddit", "youtube")}
        self.platform_factory = PlatformFactory(platforms)

        class _IvMgr:
            def get_active_intervention(self): return None
        self.intervention_manager = _IvMgr()


def _no_llm_config(**kw):
    cfg = SimulationConfig()
    cfg.use_llm = False
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_static_llm_skipped():
    # 1. run_static_llm() with use_llm=False returns dict with status="skipped"
    config = _no_llm_config()
    agents = [MockAgent(i) for i in range(5)]
    result = run_static_llm(agents, config)
    assert result["status"] == "skipped"
    assert "adjustments" in result


def test_dynamic_llm_skipped():
    # 2. run_dynamic_llm() with use_llm=False returns dict with status="skipped"
    config = _no_llm_config()
    agent = MockAgent(0)
    state = MockState()
    result = run_dynamic_llm(agent, state, timestep=10,
                              trigger="arousal_threshold", config=config)
    assert result["status"] == "skipped"
    assert "parameter_adjustments" in result


def test_sequential_llm_skipped():
    # 3. run_sequential_llm() with use_llm=False returns dict with status="skipped"
    config = _no_llm_config()
    state = MockState()
    result = run_sequential_llm(state, checkpoint_t=1,
                                prior_findings=[], config=config)
    assert result["status"] == "skipped"
    assert "findings" in result
    assert "hypotheses" in result


def test_coordinator_instantiation():
    # 4. LLMPipelineCoordinator instantiates correctly
    config = _no_llm_config()
    coordinator = LLMPipelineCoordinator(config)
    assert coordinator.research_log == []
    assert coordinator.hypothesis_store == []
    assert coordinator.dynamic_call_log == []


def test_run_static_mode_appends_log():
    # 5. run_static_mode() appends to research_log
    config = _no_llm_config()
    coordinator = LLMPipelineCoordinator(config)
    agents = [MockAgent(i) for i in range(5)]
    coordinator.run_static_mode(agents)
    assert len(coordinator.research_log) == 1
    assert coordinator.research_log[0]["mode"] == "static"


def test_dynamic_mode_returns_none_low_arousal():
    # 6. run_dynamic_mode() returns None when agent arousal is below trigger threshold
    config = _no_llm_config()
    coordinator = LLMPipelineCoordinator(config)
    agent = MockAgent(0)
    agent.arousal_level = 0.1        # well below 0.8 threshold
    agent.community_just_migrated = False
    state = MockState()
    result = coordinator.run_dynamic_mode(agent, state, timestep=100)
    assert result is None


def test_dynamic_mode_returns_none_in_cooldown():
    # 7. run_dynamic_mode() returns None when agent is in cooldown period
    config = _no_llm_config()
    coordinator = LLMPipelineCoordinator(config)
    agent = MockAgent(0)
    agent.arousal_level = 0.95       # above threshold
    state = MockState()

    # First call at t=100 — should fire (returns skipped dict, not None)
    result1 = coordinator.run_dynamic_mode(agent, state, timestep=100)
    assert result1 is not None

    # Second call only 5 timesteps later — should be blocked by cooldown (50 steps)
    result2 = coordinator.run_dynamic_mode(agent, state, timestep=105)
    assert result2 is None


def test_run_sequential_mode_appends_log():
    # 8. run_sequential_mode() appends to research_log
    config = _no_llm_config()
    coordinator = LLMPipelineCoordinator(config)
    state = MockState()
    coordinator.run_sequential_mode(state, checkpoint_t=1)
    assert len(coordinator.research_log) == 1
    assert coordinator.research_log[0]["mode"] == "sequential"


def test_build_trajectory_summary_keys():
    # 9. build_trajectory_summary() returns dict with mean_belief_position key
    state = MockState()
    summary = build_trajectory_summary(state, checkpoint_t=1)
    assert "mean_belief_position" in summary
    assert isinstance(summary["mean_belief_position"], float)
    assert "n_agents" in summary


def test_get_next_batch_configs_empty():
    # 10. get_next_batch_configs() returns list (even if hypothesis_store is empty)
    config = _no_llm_config()
    coordinator = LLMPipelineCoordinator(config)
    result = coordinator.get_next_batch_configs(config)
    assert isinstance(result, list)
    assert len(result) == 0   # nothing to test yet
