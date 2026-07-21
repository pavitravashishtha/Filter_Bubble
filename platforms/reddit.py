import math
import random
import numpy as np
import networkx as nx
from typing import List, Any, Optional

from core.platform import Platform
from config.simulation_config import SimulationConfig

class RedditPlatform(Platform):
    """
    Reddit Graph Theory and Statistical Confidence recommendation platform.
    """
    platform_name = "reddit"

    def __init__(self, config: SimulationConfig, community_graph: nx.Graph):
        """
        Initializes the Reddit platform.
        
        Args:
            config: The simulation configuration.
            community_graph: A NetworkX graph containing communities.
        """
        super().__init__(
            algo_weight=config.reddit_algo_weight,
            social_weight=config.reddit_social_weight
        )
        self.graph = community_graph
        self.decay_constant = config.reddit_decay_constant
        self._community_mean_cache: dict = {}   # {community_id: float}
        self._cache_timestep: int = -1

    def serve_content(self, agent: Any, content_pool: List[Any], timestep: int,
                      intervention: Optional[str] = None) -> Any:
        """
        Serves content using hybrid hot-scoring and ideological alignment.
        
        Args:
            agent: The agent to serve content to.
            content_pool: List of available content.
            timestep: The current simulation timestep.
            intervention: Optional intervention type.
            
        Returns:
            The selected content item.
        """
        self._refresh_community_means(timestep)
        candidates = self.generate_candidates(
            agent, content_pool, intervention
        )
        if not candidates:
            return random.choice(content_pool)
        return max(candidates,
                   key=lambda c: self._hybrid_score(
                       c, agent, timestep
                   ))

    def generate_candidates(self, agent: Any, content_pool: List[Any],
                            intervention: Optional[str] = None) -> List[Any]:
        """
        Generates candidate items from the agent's communities.
        
        Args:
            agent: The agent.
            content_pool: List of available content.
            intervention: Optional intervention type.
            
        Returns:
            List of candidate content items.
        """
        community_content = self._get_community_content(
            agent, content_pool
        )
        if not community_content:
            community_content = random.sample(
                content_pool, min(100, len(content_pool))
            )
        if intervention == "diversity_injection":
            distant = [c for c in content_pool
                       if abs(c.position_in_belief_space
                              - agent.belief_position) > 3.0
                       and c not in community_content]
            diverse = random.sample(distant, min(20, len(distant)))
            for c in diverse:
                c.is_intervention_content = True
            community_content = community_content[:80] + diverse
        return community_content

    def _get_community_content(self, agent: Any, content_pool: List[Any]) -> List[Any]:
        """
        Filters the content pool down to the agent's subscribed communities.
        """
        if not agent.subscribed_communities:
            return []
        agent_community_ids = set(agent.subscribed_communities)
        return [c for c in content_pool
                if hasattr(c, "community_id")
                and c.community_id in agent_community_ids]

    def _hybrid_score(self, content: Any, agent: Any, timestep: int) -> float:
        """
        Computes a composite score based on 'hotness' and 'ideological alignment'.
        """
        hot = self._hot_score(content, timestep)
        ideological = self._ideological_alignment(content, agent, timestep)
        return hot * 0.7 + ideological * 0.3

    def _hot_score(self, content: Any, timestep: int) -> float:
        """
        Computes the Reddit hotness score based on upvotes, downvotes, and age.
        """
        net_votes = content.upvotes - content.downvotes
        order = math.log10(max(abs(net_votes), 1))
        sign = 1 if net_votes > 0 else (-1 if net_votes < 0 else 0)
        age = timestep - content.timestamp
        time_penalty = age / self.decay_constant
        return (sign * order) - time_penalty

    def _wilson_score(self, content: Any, confidence: float = 1.96) -> float:
        """
        Computes the Wilson score interval lower bound.
        """
        n = content.upvotes + content.downvotes
        if n == 0:
            return 0.0
        p_hat = content.upvotes / n
        z = confidence
        numerator = (p_hat + z**2 / (2 * n)
                     - z * math.sqrt(
                         (p_hat * (1 - p_hat) / n)
                         + z**2 / (4 * n**2)
                     ))
        denominator = 1 + z**2 / n
        return numerator / denominator

    def _ideological_alignment(self, content: Any, agent: Any, timestep: int = 0) -> float:
        """
        Computes how well the content aligns with the community's mean belief.
        """
        community_mean = self._get_community_mean(agent, timestep)
        distance = abs(content.position_in_belief_space - community_mean)
        return max(0.0, 1.0 - distance / 10.0)

    def _refresh_community_means(self, timestep: int) -> None:
        """
        Recomputes and caches community mean belief positions once per timestep.
        """
        if timestep == self._cache_timestep:
            return
        self._cache_timestep = timestep
        self._community_mean_cache = {}
        # Group node belief_positions by community_id in a single pass
        buckets: dict = {}
        for n, data in self.graph.nodes(data=True):
            cid = data.get("community_id")
            if cid is not None:
                buckets.setdefault(cid, []).append(data.get("belief_position", 5.0))
        for cid, positions in buckets.items():
            self._community_mean_cache[cid] = float(np.mean(positions))

    def _get_community_mean(self, agent: Any, timestep: int = 0) -> float:
        """
        Retrieves the cached average belief position of the agent's primary community.
        """
        if not agent.subscribed_communities:
            return 5.0
        community_id = agent.subscribed_communities[0]
        return self._community_mean_cache.get(community_id, 5.0)

    def score_content(self, content: Any, agent: Any, timestep: int) -> float:
        """
        Unified scoring pipeline that dynamically routes based on content type.
        
        Args:
            content: The content to score.
            agent: The agent doing the scoring.
            timestep: The current simulation timestep.
            
        Returns:
            The final score.
        """
        if content.content_type == "post":
            return self._hot_score(content, timestep)
        elif content.content_type == "comment":
            return self._wilson_score(content)
        return self._hot_score(content, timestep)

    def learn_from_feedback(self, agent: Any, content: Any, engaged: bool,
                            engagement_strength: float) -> None:
        """
        Learns from agent feedback by incrementing votes.
        
        Args:
            agent: The agent.
            content: The content interacted with.
            engaged: Whether the agent engaged.
            engagement_strength: The strength of the engagement.
        """
        if engaged and hasattr(content, "upvotes"):
            content.upvotes += 1
        elif not engaged and hasattr(content, "downvotes"):
            content.downvotes += 1

    def check_community_migration(self, agent: Any, timestep: int) -> None:
        """
        Checks if the agent has drifted too far from its community and should migrate.
        
        Args:
            agent: The agent to check.
            timestep: The current simulation timestep.
        """
        community_mean = self._get_community_mean(agent)
        drift = abs(agent.belief_position - community_mean)
        effective_threshold = (
            agent.draw_effective_parameters()["confidence_threshold"]
        )
        if drift > effective_threshold:
            new_community = self._find_aligned_community(agent)
            if new_community is not None:
                if agent.primary_community in agent.subscribed_communities:
                    agent.subscribed_communities.remove(
                        agent.primary_community
                    )
                agent.subscribed_communities.append(new_community)
                agent.primary_community = new_community
                agent.community_just_migrated = True

    def _find_aligned_community(self, agent: Any) -> Optional[Any]:
        """
        Finds the closest community to the agent's current belief position.
        """
        best_community = None
        best_distance = float("inf")
        all_communities = set(
            self.graph.nodes[n].get("community_id")
            for n in self.graph.nodes()
            if self.graph.nodes[n].get("community_id") is not None
        )
        for community_id in all_communities:
            if community_id in agent.subscribed_communities:
                continue
            members = [n for n in self.graph.nodes()
                       if self.graph.nodes[n].get(
                           "community_id"
                       ) == community_id]
            if not members:
                continue
            positions = [self.graph.nodes[m].get(
                "belief_position", 5.0
            ) for m in members]
            community_mean = float(np.mean(positions))
            distance = abs(agent.belief_position - community_mean)
            if distance < best_distance:
                best_distance = distance
                best_community = community_id
        return best_community

    def record_weight_snapshot(self, timestep: int) -> None:
        """
        Records the current decay constant to weight history.
        
        Args:
            timestep: The current simulation timestep.
        """
        self.weight_history.append({
            "timestep": timestep,
            "decay_constant": self.decay_constant,
        })
