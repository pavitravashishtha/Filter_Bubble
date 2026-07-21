import random
from typing import List, Any, Optional

from core.platform import Platform
from config.simulation_config import SimulationConfig

class YouTubePlatform(Platform):
    """
    YouTube Reinforcement Learning Contextual Bandit recommendation platform.
    """
    platform_name = "youtube"

    def __init__(self, config: SimulationConfig):
        """
        Initializes the YouTube platform using SimulationConfig.
        
        Args:
            config: The simulation configuration.
        """
        super().__init__(
            algo_weight=config.youtube_algo_weight,
            social_weight=config.youtube_social_weight
        )
        self.weights = {
            "W_V": 0.25,
            "W_E": 0.25,
            "W_N": 0.25,
            "W_D": 0.25,
        }
        self.learning_rate = config.youtube_learning_rate
        self.exploration_rate = config.youtube_exploration_rate

    def serve_content(self, agent: Any, content_pool: List[Any], timestep: int,
                      intervention: Optional[str] = None) -> Any:
        """
        Serves content using an RL contextual bandit approach.
        
        Args:
            agent: The agent to serve content to.
            content_pool: List of available content.
            timestep: The current simulation timestep.
            intervention: Optional intervention type.
            
        Returns:
            The selected content item.
        """
        if agent.interaction_count < 10:
            return self._cold_start_serve(content_pool)
        if random.random() < self.exploration_rate:
            return random.choice(content_pool)
        candidates = self.generate_candidates(
            agent, content_pool, intervention
        )
        if not candidates:
            return random.choice(content_pool)
        return max(candidates, key=lambda c: self._score(c, agent))

    def generate_candidates(self, agent: Any, content_pool: List[Any],
                            intervention: Optional[str] = None) -> List[Any]:
        """
        Generates candidate items.
        
        Args:
            agent: The agent.
            content_pool: List of available content.
            intervention: Optional intervention type.
            
        Returns:
            List of candidate content items.
        """
        candidates = random.sample(
            content_pool, min(100, len(content_pool))
        )
        if intervention == "diversity_injection":
            distant = [c for c in content_pool
                       if abs(c.position_in_belief_space
                              - agent.belief_position) > 3.0]
            diverse = random.sample(distant, min(20, len(distant)))
            for c in diverse:
                c.is_intervention_content = True
            candidates = candidates[:80] + diverse
        return candidates

    def _score(self, content: Any, agent: Any) -> float:
        """
        Scores a candidate content item.
        
        Args:
            content: The candidate content item.
            agent: The agent.
            
        Returns:
            The score of the content item.
        """
        distance = abs(content.position_in_belief_space
                       - agent.belief_position)
        return (
            self.weights["W_V"] * content.emotional_valence
            + self.weights["W_E"] * content.engagement_potential
            + self.weights["W_N"] * content.novelty
            - self.weights["W_D"] * distance
        )

    def _cold_start_serve(self, content_pool: List[Any]) -> Any:
        """
        Serves content to agents with too few interactions.
        
        Args:
            content_pool: List of available content.
            
        Returns:
            A selected content item using a default scoring function.
        """
        candidates = random.sample(
            content_pool, min(100, len(content_pool))
        )
        return max(candidates,
                   key=lambda c: (
                       0.25 * c.emotional_valence
                       + 0.25 * c.engagement_potential
                       + 0.25 * c.novelty
                   ))

    def learn_from_feedback(self, agent: Any, content: Any, engaged: bool,
                            engagement_strength: float) -> None:
        """
        Learns from agent feedback, updating bandit weights.
        
        Args:
            agent: The agent.
            content: The content interacted with.
            engaged: Whether the agent engaged.
            engagement_strength: The strength of the engagement.
        """
        if engaged:
            completion = min(1.0, engagement_strength)
            reward = completion * engagement_strength
        else:
            reward = -0.5

        distance = abs(content.position_in_belief_space
                       - agent.belief_position)
        predicted = self._score(content, agent)
        error = reward - predicted

        self.weights["W_V"] += (self.learning_rate * error
                                * content.emotional_valence)
        self.weights["W_E"] += (self.learning_rate * error
                                * content.engagement_potential)
        self.weights["W_N"] += (self.learning_rate * error
                                * content.novelty)
        self.weights["W_D"] += (self.learning_rate * (-error)
                                * distance)

        total = sum(abs(v) for v in self.weights.values())
        if total > 0:
            self.weights = {k: v / total
                            for k, v in self.weights.items()}
        self.weights["W_D"] = max(0.0, self.weights["W_D"])
        self.weights["W_N"] = max(0.0, self.weights["W_N"])

    def apply_friction(self, friction_factor: float = 0.5) -> None:
        """
        Applies an algorithmic friction intervention by lowering the weight of valence.
        
        Args:
            friction_factor: The multiplier to apply to the valence weight.
        """
        self.weights["W_V"] *= friction_factor

    def record_weight_snapshot(self, timestep: int) -> None:
        """
        Records the current bandit weights to history.
        
        Args:
            timestep: The current simulation timestep.
        """
        self.weight_history.append({
            "timestep": timestep,
            "W_V": self.weights["W_V"],
            "W_E": self.weights["W_E"],
            "W_N": self.weights["W_N"],
            "W_D": self.weights["W_D"],
        })
