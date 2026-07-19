from typing import Dict, List, Any

class SimulationConfig:
    """
    Configuration system for the filter bubble simulation.
    Contains all parameters, settings, and constants used across the system.
    """

    def __init__(self, **kwargs):
        # RUN IDENTITY
        self.run_id: str = "run_001"
        self.random_seed: int = 42

        # AGENT SETTINGS
        self.n_agents: int = 300
        self.agent_distribution: dict = {
            "heavy_imdb": 60,
            "heavy_reddit": 60,
            "heavy_youtube": 60,
            "balanced": 80,
            "cross_platform": 40
        }

        # TIME SETTINGS
        self.total_timesteps: int = 1000
        self.burnin_end: int = 100
        self.intervention_start: int = 800
        self.checkpoints: list = [100, 200, 400, 600, 800, 900, 1000]
        self.sequential_llm_checkpoints: list = [400, 800, 1000]

        # PLATFORM SETTINGS
        self.latent_dims: int = 25
        self.n_reddit_communities: int = 20
        self.n_youtube_clusters: int = 15

        # DATA PATHS
        self.data_dir: str = "data"
        self.imdb_csv: str = "data/imdb/imdb_content_pool.csv"
        self.reddit_csv: str = "data/reddit/reddit_content_pool.csv"
        self.youtube_csv: str = "data/youtube/youtube_content_pool.csv"
        self.genre_index_json: str = "data/imdb/imdb_genre_index.json"

        # ALGORITHM SETTINGS — IMDB
        self.imdb_algo_weight: float = 0.6
        self.imdb_social_weight: float = 0.4
        self.imdb_learning_rate: float = 0.01
        self.imdb_regularization: float = 0.02
        self.imdb_global_mean_rating: float = 0.65

        # ALGORITHM SETTINGS — REDDIT
        self.reddit_algo_weight: float = 0.5
        self.reddit_social_weight: float = 0.5
        self.reddit_decay_constant: float = 45000.0
        self.reddit_p_within: float = 0.7
        self.reddit_p_across: float = 0.02

        # ALGORITHM SETTINGS — YOUTUBE
        self.youtube_algo_weight: float = 0.9
        self.youtube_social_weight: float = 0.1
        self.youtube_learning_rate: float = 0.05
        self.youtube_exploration_rate: float = 0.10

        # NETWORK SETTINGS
        self.imdb_knn_k: int = 10
        self.imdb_network_recalc_interval: int = 50
        self.reddit_migration_check_interval: int = 10
        self.youtube_subscription_probability: float = 0.05

        # LAYER WEIGHTS BY CONTEXT
        self.layer_weights: dict = {
            "imdb":          {"imdb": 0.6, "reddit": 0.2, "youtube": 0.1},
            "reddit":        {"imdb": 0.2, "reddit": 0.6, "youtube": 0.2},
            "youtube":       {"imdb": 0.1, "reddit": 0.2, "youtube": 0.7},
            "cross_platform":{"imdb": 0.33, "reddit": 0.33, "youtube": 0.34}
        }

        # INTERVENTION SETTINGS
        self.intervention_type: str = "diversity_injection"
        self.intervention_strength: float = 0.2
        self.intervention_diversity_distance_threshold: float = 3.0
        self.active_intervention: str | None = None

        # LLM SETTINGS
        self.dynamic_llm_trigger_threshold: float = 0.8
        self.dynamic_llm_cooldown_timesteps: int = 50
        self.use_llm: bool = False

        # NEW AGENT ENTRY (main phase)
        self.new_agent_entry_rate: int = 1

        # CONTENT REFRESH RATES
        self.imdb_injection_rate: int = 3
        self.reddit_injection_rate: int = 30
        self.youtube_injection_rate: int = 15
        
        # Override any defaults with provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
                
        self.validate()

    def validate(self) -> None:
        """
        Validates the configuration parameters.
        Raises ValueError if any constraints are violated.
        """
        # Checks agent_distribution values sum to n_agents
        if sum(self.agent_distribution.values()) != self.n_agents:
            raise ValueError(
                f"Agent distribution values sum to {sum(self.agent_distribution.values())}, "
                f"but n_agents is {self.n_agents}."
            )
            
        # Checks intervention_type is valid
        valid_interventions = ["diversity_injection", "algorithm_friction", "social_bridge", "null"]
        if self.intervention_type not in valid_interventions:
            raise ValueError(
                f"Invalid intervention_type: {self.intervention_type}. "
                f"Must be one of {valid_interventions}."
            )
            
        # Checks all weight pairs (algo+social) sum to 1.0 for each platform
        if not abs((self.imdb_algo_weight + self.imdb_social_weight) - 1.0) < 1e-6:
            raise ValueError("IMDB algo and social weights must sum to 1.0.")
            
        if not abs((self.reddit_algo_weight + self.reddit_social_weight) - 1.0) < 1e-6:
            raise ValueError("Reddit algo and social weights must sum to 1.0.")
            
        if not abs((self.youtube_algo_weight + self.youtube_social_weight) - 1.0) < 1e-6:
            raise ValueError("YouTube algo and social weights must sum to 1.0.")

    def to_dict(self) -> dict:
        """
        Returns all configuration properties as a flat dictionary.
        
        Returns:
            dict: The configuration as a dictionary.
        """
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> 'SimulationConfig':
        """
        Creates a configuration instance from a dictionary of parameters.
        
        Args:
            d (dict): Dictionary of configuration parameters.
            
        Returns:
            SimulationConfig: A new configuration object.
        """
        return cls(**d)


def load_default_config() -> SimulationConfig:
    """
    Returns a SimulationConfig object populated with all default settings.
    
    Returns:
        SimulationConfig: The default configuration.
    """
    return SimulationConfig()
