import numpy as np
from abc import ABC, abstractmethod

class BaseContent(ABC):
    """
    Abstract base class representing generic content in the filter bubble simulation.
    Contains universal properties and methods shared across all platforms.
    """

    @abstractmethod
    def __init__(self, content_id: int, birth_timestep: int):
        """
        Initializes the base content properties.
        Must be called by subclasses.

        Args:
            content_id (int): Unique identifier for the content.
            birth_timestep (int): The timestep when the content entered the pool.
        """
        # Ensure it cannot be instantiated directly without subclassing
        if type(self) is BaseContent:
            raise TypeError("BaseContent cannot be instantiated directly.")
            
        self.content_id: int = content_id
        self.birth_timestep: int = birth_timestep
        
        self.position_in_belief_space: float = float(
            np.clip(np.random.normal(5.0, 2.0), 0.0, 10.0)
        )
        self.emotional_valence: float = float(np.random.uniform(0.0, 1.0))
        self.engagement_potential: float = float(np.random.uniform(0.0, 1.0))
        self.quality_signal: float = float(np.random.uniform(0.0, 1.0))
        self.novelty: float = 1.0
        self.is_intervention_content: bool = False
        self.recent_engagement: float = 0.0
        self.consecutive_zero_engagement_timesteps: int = 0
        
        # Subclasses will define these
        self.decay_rate: float = 0.0
        self.retirement_threshold: int = 0

    def decay_novelty(self, current_timestep: int, decay_rate: float) -> None:
        """
        Decays the novelty of the content based on its age and a decay rate.

        Args:
            current_timestep (int): The current simulation timestep.
            decay_rate (float): The rate at which novelty decays.
        """
        self.novelty = max(0.0, 1.0 - (current_timestep - self.birth_timestep) * decay_rate)

    def should_retire(self, current_timestep: int) -> bool:
        """
        Checks if the content should be retired from the active pool.

        Args:
            current_timestep (int): The current simulation timestep.

        Returns:
            bool: True if the content should be retired, False otherwise.
        """
        if self.novelty <= 0 and self.recent_engagement <= 0:
            self.consecutive_zero_engagement_timesteps += 1
        else:
            self.consecutive_zero_engagement_timesteps = 0
            
        return self.consecutive_zero_engagement_timesteps >= self.retirement_threshold


class IMDBContent(BaseContent):
    """
    Represents a movie or show from IMDB with latent features and critical consensus.
    """
    
    def __init__(self, content_id: int, birth_timestep: int):
        super().__init__(content_id, birth_timestep)
        
        self.latent_features: np.ndarray = np.random.normal(0, 0.1, 25)
        self.ideological_loading_vector: np.ndarray = np.random.normal(0, 0.05, 25)
        self.item_bias: float = float(np.random.normal(0, 0.1))
        
        eras = [1970, 1980, 1990, 2000, 2010, 2020]
        self.release_era: int = int(np.random.choice(eras))
        
        self.critical_consensus: float = float(np.random.uniform(0.0, 1.0))
        self.mainstream_appeal: float = float(np.random.uniform(0.0, 1.0))
        
        self.decay_rate: float = 0.002
        self.retirement_threshold: int = 200


class RedditContent(BaseContent):
    """
    Represents a post or comment from Reddit with upvotes and downvotes.
    """
    
    def __init__(self, content_id: int, birth_timestep: int, community_id: int, content_type: str = "post"):
        super().__init__(content_id, birth_timestep)
        
        self.community_id: int = community_id
        self.upvotes: int = int(np.random.randint(1, 11))
        self.downvotes: int = int(np.random.randint(0, 4))
        self.timestamp: int = birth_timestep
        self.content_type: str = content_type
        self.cross_post_origin: int | None = None
        
        self.decay_rate: float = 0.05
        self.retirement_threshold: int = 10


class YouTubeContent(BaseContent):
    """
    Represents a video from YouTube with predicted watch time and thumbnail appeal.
    """
    
    def __init__(self, content_id: int, birth_timestep: int, topic_cluster: int):
        super().__init__(content_id, birth_timestep)
        
        self.predicted_watch_time: float = float(np.random.uniform(0.0, 1.0))
        self.completion_rate_baseline: float = 0.5
        self.thumbnail_appeal: float = float(np.random.uniform(0.0, 1.0))
        self.topic_cluster: int = topic_cluster
        self.creator_authority: float = float(np.random.uniform(0.0, 1.0))
        
        self.decay_rate: float = 0.02
        self.retirement_threshold: int = 100
