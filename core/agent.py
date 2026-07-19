import numpy as np
from typing import Dict, Set, List, Optional

class Agent:
    """
    Represents a user agent in the filter bubble simulation, engaging across
    multiple platforms (IMDB, Reddit, YouTube) with a stable identity,
    stochastic personality traits, and an emotional state machine.
    """

    def __init__(self, agent_id: int, archetype: str):
        """
        Initializes an agent with a specific ID and archetype.

        Args:
            agent_id (int): Unique identifier for the agent.
            archetype (str): The psychological archetype of the agent.
                             Must be one of: 'heavy_imdb', 'heavy_reddit',
                             'heavy_youtube', 'balanced', 'cross_platform'.
        """
        self.agent_id: int = agent_id
        
        valid_archetypes = ['heavy_imdb', 'heavy_reddit', 'heavy_youtube', 'balanced', 'cross_platform']
        if archetype not in valid_archetypes:
            raise ValueError(f"Invalid archetype: {archetype}. Must be one of {valid_archetypes}")
        
        self.archetype: str = archetype
        
        # LEVEL 1 - STABLE IDENTITY
        self.base_belief_position: float = float(np.clip(np.random.normal(5.0, 2.0), 0.0, 10.0))
        self.belief_position: float = self.base_belief_position
        
        self.platform_weights: Dict[str, float] = self._init_platform_weights()
        
        # LEVEL 2 - STOCHASTIC PERSONALITY TRAITS (means and stds)
        self._trait_params: Dict[str, tuple] = self._init_trait_params()
        
        # LEVEL 3 - EMOTIONAL STATE MACHINE
        self.arousal_level: float = 0.0
        self.consecutive_high_valence_count: int = 0
        self.state_decay_rate: float = 0.05
        self.current_cognitive_load: float = 0.0
        
        # TRACKING PROPERTIES
        self.interaction_count: int = 0
        self.seen_content_ids: Set[int] = set()
        self.community_id: int = -1  # Default invalid ID, assigned later
        self.subscribed_communities: List[int] = []
        self.primary_community: Optional[int] = None
        self.community_just_migrated: bool = False
        self.watch_history: List[int] = []
        self.preference_vector: np.ndarray = np.zeros(25)
        self.user_bias: float = 0.0

    def _init_platform_weights(self) -> Dict[str, float]:
        if self.archetype == 'heavy_imdb':
            return {'imdb_weight': 0.7, 'reddit_weight': 0.2, 'youtube_weight': 0.1}
        elif self.archetype == 'heavy_reddit':
            return {'imdb_weight': 0.2, 'reddit_weight': 0.7, 'youtube_weight': 0.1}
        elif self.archetype == 'heavy_youtube':
            return {'imdb_weight': 0.1, 'reddit_weight': 0.2, 'youtube_weight': 0.7}
        elif self.archetype == 'balanced':
            return {'imdb_weight': 0.33, 'reddit_weight': 0.33, 'youtube_weight': 0.34}
        elif self.archetype == 'cross_platform':
            weights = np.random.dirichlet([1.0, 1.0, 1.0])
            return {'imdb_weight': float(weights[0]), 'reddit_weight': float(weights[1]), 'youtube_weight': float(weights[2])}
        return {}

    def _init_trait_params(self) -> Dict[str, tuple]:
        params = {}
        if self.archetype == 'heavy_imdb':
            means = {'conf_thresh': 0.5, 'update_rate': 0.3, 'susceptibility': 0.4, 'open_mind': 0.6, 'crit_think': 0.6}
        elif self.archetype == 'heavy_reddit':
            means = {'conf_thresh': 0.4, 'update_rate': 0.4, 'susceptibility': 0.5, 'open_mind': 0.5, 'crit_think': 0.5}
        elif self.archetype == 'heavy_youtube':
            means = {'conf_thresh': 0.6, 'update_rate': 0.5, 'susceptibility': 0.6, 'open_mind': 0.4, 'crit_think': 0.4}
        elif self.archetype == 'balanced':
            means = {'conf_thresh': 0.5, 'update_rate': 0.3, 'susceptibility': 0.4, 'open_mind': 0.6, 'crit_think': 0.6}
        elif self.archetype == 'cross_platform':
            means = {'conf_thresh': 0.5, 'update_rate': 0.4, 'susceptibility': 0.5, 'open_mind': 0.7, 'crit_think': 0.5}
        
        for trait, mean in means.items():
            params[trait] = (mean, mean * 0.1)
        return params
        
    def draw_effective_parameters(self) -> Dict[str, float]:
        """
        Draws fresh stochastic parameters from the agent's trait distributions,
        applies modifiers based on the current emotional state, and clamps to [0.0, 1.0].
        
        Returns:
            Dict[str, float]: A dictionary containing the effective parameters.
        """
        base_draws = {}
        for trait, (mean, std) in self._trait_params.items():
            val = np.random.normal(mean, std)
            base_draws[trait] = np.clip(val, 0.0, 1.0)
            
        effective = {}
        effective['susceptibility'] = np.clip(base_draws['susceptibility'] * (1 + self.arousal_level), 0.0, 1.0)
        effective['crit_think'] = np.clip(base_draws['crit_think'] * (1 - 0.5 * self.arousal_level), 0.0, 1.0)
        effective['conf_thresh'] = np.clip(base_draws['conf_thresh'] * (1 - 0.3 * self.arousal_level), 0.0, 1.0)
        effective['update_rate'] = np.clip(base_draws['update_rate'] * (1 + 0.2 * self.arousal_level), 0.0, 1.0)
        effective['open_mind'] = base_draws['open_mind']
        
        return {
            'susceptibility': float(effective['susceptibility']),
            'critical_thinking': float(effective['crit_think']),
            'confidence_threshold': float(effective['conf_thresh']),
            'update_rate': float(effective['update_rate']),
            'open_mind': float(effective['open_mind'])
        }

    def update_emotional_state(self, content_valence: float) -> str:
        """
        Updates the emotional state and cognitive load of the agent based on
        the valence of the consumed content.

        Args:
            content_valence (float): The valence of the content, typically [0.0, 1.0].

        Returns:
            str: The name of the resulting emotional state.
        """
        if content_valence > 0.7:
            self.arousal_level += 0.15
            self.consecutive_high_valence_count += 1
            self.current_cognitive_load += 0.1
        else:
            self.arousal_level -= (self.state_decay_rate * self.arousal_level)
            self.consecutive_high_valence_count = 0
            self.current_cognitive_load -= (self.state_decay_rate * self.current_cognitive_load)
            
        self.arousal_level = float(np.clip(self.arousal_level, 0.0, 1.0))
        self.current_cognitive_load = float(np.clip(self.current_cognitive_load, 0.0, 1.0))
        
        return self.get_state_name()

    def get_state_name(self) -> str:
        """
        Determines the current state name based on arousal level.

        Returns:
            str: State name (CALM, ENGAGED, AGITATED, or REACTIVE).
        """
        if self.arousal_level <= 0.3:
            return "CALM"
        elif self.arousal_level <= 0.6:
            return "ENGAGED"
        elif self.arousal_level <= 0.8:
            return "AGITATED"
        else:
            return "REACTIVE"
            
    def reset_migration_flag(self) -> None:
        """
        Resets the community migration flag to False.
        """
        self.community_just_migrated = False
