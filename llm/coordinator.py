from typing import Any, Optional, List

from llm.static_llm import run_static_llm
from llm.dynamic_llm import run_dynamic_llm, build_dynamic_context
from llm.sequential_llm import run_sequential_llm, build_trajectory_summary


class LLMPipelineCoordinator:
    """
    Coordinates all three LLM pipeline modes (static, dynamic, sequential)
    and accumulates research findings across the simulation run.
    """

    def __init__(self, config: Any):
        """
        Initializes the coordinator with simulation config.

        Args:
            config: SimulationConfig object.
        """
        self.config = config
        self.research_log: List[dict] = []
        self.hypothesis_store: List[dict] = []
        self.dynamic_call_log: List[dict] = []
        self.cooldown_per_agent: dict = {}

    def run_static_mode(self, agents: list) -> dict:
        """
        Runs the static LLM pass at simulation initialization.

        Args:
            agents: The full list of agent objects.

        Returns:
            The static LLM result dict.
        """
        result = run_static_llm(agents, self.config)
        self.research_log.append({
            "mode": "static", "timestep": 0, "result": result
        })
        return result

    def run_dynamic_mode(self, agent: Any, state: Any,
                         timestep: int) -> Optional[dict]:
        """
        Checks trigger conditions and fires a dynamic LLM call if warranted.

        Args:
            agent: The agent to evaluate.
            state: The current SimulationState.
            timestep: The current simulation timestep.

        Returns:
            The dynamic LLM result dict, or None if conditions not met.
        """
        # Cooldown check
        last_call = self.cooldown_per_agent.get(agent.id, 0)
        if timestep - last_call < self.config.dynamic_llm_cooldown_timesteps:
            return None

        # Identify trigger
        trigger = None
        if agent.arousal_level > self.config.dynamic_llm_trigger_threshold:
            trigger = "arousal_threshold"
        elif (hasattr(state, "measurement_store") and
              state.measurement_store.get_latest_metrics(
                  agent.id)["drift_rate"] > 0.8):
            trigger = "rapid_radicalization"
        elif agent.community_just_migrated:
            trigger = "community_migration"

        if trigger is None:
            return None

        result = run_dynamic_llm(agent, state, timestep, trigger, self.config)
        self.cooldown_per_agent[agent.id] = timestep

        entry = {
            "mode": "dynamic",
            "timestep": timestep,
            "agent_id": agent.id,
            "trigger": trigger,
            "result": result
        }
        self.dynamic_call_log.append(entry)
        self.research_log.append(entry)

        return result

    def run_sequential_mode(self, state: Any, checkpoint_t: int) -> dict:
        """
        Runs a sequential checkpoint LLM analysis.

        Args:
            state: The current SimulationState.
            checkpoint_t: The checkpoint timestep to analyze.

        Returns:
            The sequential LLM result dict.
        """
        prior = [e["result"] for e in self.research_log
                 if e.get("mode") == "sequential"]
        result = run_sequential_llm(state, checkpoint_t, prior, self.config)
        self.research_log.append({
            "mode": "sequential",
            "timestep": checkpoint_t,
            "result": result
        })
        if "hypotheses" in result:
            self.hypothesis_store.extend(result["hypotheses"])
        return result

    def get_next_batch_configs(self, current_config: Any) -> list:
        """
        Generates modified SimulationConfig objects to test top hypotheses.

        Args:
            current_config: The current SimulationConfig to clone and modify.

        Returns:
            List of SimulationConfig objects, one per testable hypothesis.
        """
        from config.simulation_config import SimulationConfig

        configs = []
        for hypothesis in self.hypothesis_store[:5]:
            param = hypothesis.get("parameter_to_vary")
            value = hypothesis.get("test_value")
            if not param or value is None:
                continue

            try:
                new_config = SimulationConfig(**current_config.to_dict())
                if hasattr(new_config, param):
                    setattr(new_config, param, value)
                    new_config.run_id = f"hypothesis_{param}_{value}"
                    configs.append(new_config)
            except (ValueError, TypeError):
                # Skip hypotheses that result in invalid configs
                continue

        return configs
