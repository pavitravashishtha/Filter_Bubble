import random
import numpy as np
from typing import List, Any, Optional

from core.platform import Platform
from config.simulation_config import SimulationConfig


class IMDBPlatform(Platform):
    """
    IMDB Matrix Factorization recommendation platform.
    """
    platform_name = "imdb"

    def __init__(self, config: SimulationConfig):
        """
        Initializes the IMDB platform using SimulationConfig.
        
        Args:
            config (SimulationConfig): The simulation configuration.
        """
        super().__init__(
            algo_weight=config.imdb_algo_weight,
            social_weight=config.imdb_social_weight
        )
        self.global_mean_rating = config.imdb_global_mean_rating
        self.learning_rate = config.imdb_learning_rate
        self.regularization = config.imdb_regularization
        self.latent_dims = config.latent_dims

    def serve_content(self, agent: Any, content_pool: List[Any], timestep: int,
                      intervention: Optional[str] = None) -> Any:
        """
        Serves content to the agent based on matrix factorization predictions.
        
        Args:
            agent: The agent to serve content to.
            content_pool: List of available content.
            timestep: The current simulation timestep.
            intervention: Optional intervention type.
            
        Returns:
            The selected content item.
        """
        candidates = self.generate_candidates(
            agent, content_pool, intervention
        )
        if not candidates:
            return random.choice(content_pool)
        return max(candidates,
                   key=lambda c: self._predict_rating(agent, c))

    def generate_candidates(self, agent: Any, content_pool: List[Any],
                            intervention: Optional[str] = None) -> List[Any]:
        """
        Generates candidate content items for the agent.
        
        Args:
            agent: The agent.
            content_pool: List of available content.
            intervention: Optional intervention type.
            
        Returns:
            List of candidate content items.
        """
        if agent.interaction_count < 10:
            return random.sample(content_pool,
                                 min(200, len(content_pool)))
        unseen = [c for c in content_pool
                  if c.content_id not in agent.seen_content_ids]
        candidates = random.sample(unseen, min(200, len(unseen)))
        if intervention == "diversity_injection":
            distant = [c for c in unseen
                       if abs(c.position_in_belief_space
                              - agent.belief_position) > 3.0]
            diverse = random.sample(distant, min(40, len(distant)))
            candidates = candidates[:160] + diverse
            for c in diverse:
                c.is_intervention_content = True
        return candidates

    def _predict_rating(self, agent: Any, content: Any) -> float:
        """
        Predicts the rating an agent would give to a content item.
        
        Args:
            agent: The agent.
            content: The content item.
            
        Returns:
            Predicted rating.
        """
        dot_product = float(np.dot(
            agent.preference_vector, content.latent_features
        ))
        return (self.global_mean_rating
                + agent.user_bias
                + content.item_bias
                + dot_product)

    def learn_from_feedback(self, agent: Any, content: Any, engaged: bool,
                            engagement_strength: float) -> None:
        """
        Learns from user feedback by updating user and item latent features.
        
        Args:
            agent: The agent.
            content: The content interacted with.
            engaged: Whether the agent engaged.
            engagement_strength: The strength of the engagement.
        """
        actual_rating = engagement_strength if engaged else 0.0
        predicted = self._predict_rating(agent, content)
        error = actual_rating - predicted

        agent.user_bias += self.learning_rate * (
            error - self.regularization * agent.user_bias
        )
        content.item_bias += self.learning_rate * (
            error - self.regularization * content.item_bias
        )

        p_update = self.learning_rate * (
            error * content.latent_features
            - self.regularization * agent.preference_vector
        )
        q_update = self.learning_rate * (
            error * agent.preference_vector
            - self.regularization * content.latent_features
        )
        agent.preference_vector += p_update
        content.latent_features += q_update

        ideology_bleed = float(np.dot(
            agent.preference_vector,
            content.ideological_loading_vector
        ))
        agent.belief_position += (
            ideology_bleed
            * agent.draw_effective_parameters()["update_rate"]
            * 0.1
        )
        agent.belief_position = float(
            np.clip(agent.belief_position, 0.0, 10.0)
        )

    def record_weight_snapshot(self, timestep: int) -> None:
        """
        Records a snapshot of the current platform weights.
        
        Args:
            timestep: The current simulation timestep.
        """
        self.weight_history.append({
            "timestep": timestep,
            "global_mean_rating": self.global_mean_rating,
            "learning_rate": self.learning_rate,
        })
