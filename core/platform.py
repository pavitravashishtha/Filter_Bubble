from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class Platform(ABC):
    """
    Abstract base class for all platform algorithms.
    
    This class defines the interface every platform must follow.
    It implements no recommendation logic itself.
    """

    def __init__(self, algo_weight: float, social_weight: float):
        """
        Initialize the platform with algorithm and social weights.
        
        Args:
            algo_weight (float): How much the algorithm drives content vs social.
            social_weight (float): How much the social network drives content.
            
        Raises:
            ValueError: If algo_weight + social_weight does not equal 1.0.
        """
        if not np.isclose(algo_weight + social_weight, 1.0):
            raise ValueError("algo_weight and social_weight must sum to 1.0")
            
        self.algo_weight: float = algo_weight
        self.social_weight: float = social_weight
        self.weight_history: List[Dict[str, Any]] = []

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """
        The name of the platform.
        """
        pass

    @abstractmethod
    def serve_content(self, agent: Any, content_pool: List[Any], timestep: int, intervention: Optional[Dict[str, Any]] = None) -> Any:
        """
        Selects and returns one content item for this agent.
        
        Args:
            agent (Agent): The agent to serve content to.
            content_pool (List[Content]): The pool of available content.
            timestep (int): The current timestep in the simulation.
            intervention (Optional[Dict]): Any active intervention to apply during content selection.
            
        Returns:
            Content: The selected content item.
        """
        pass

    @abstractmethod
    def learn_from_feedback(self, agent: Any, content: Any, engaged: bool, engagement_strength: float) -> None:
        """
        Updates platform internal model based on agent response.
        
        Args:
            agent (Agent): The agent who interacted with the content.
            content (Content): The content item.
            engaged (bool): Whether the agent engaged with the content.
            engagement_strength (float): The strength of the engagement.
        """
        pass

    @abstractmethod
    def generate_candidates(self, agent: Any, content_pool: List[Any], intervention: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Returns a list of candidate content items before final scoring.
        
        Args:
            agent (Agent): The agent to generate candidates for.
            content_pool (List[Content]): The pool of available content.
            intervention (Optional[Dict]): Any interventions to apply.
            
        Returns:
            List[Content]: A list of candidate content items.
        """
        pass

    def record_weight_snapshot(self, timestep: int) -> None:
        """
        Appends current internal state to weight_history.
        Subclasses can override to specify what state to record.
        
        Args:
            timestep (int): The current timestep in the simulation.
        """
        self.weight_history.append({
            "timestep": timestep,
            "algo_weight": self.algo_weight,
            "social_weight": self.social_weight
        })

    def get_weight_history(self) -> List[Dict[str, Any]]:
        """
        Returns the weight_history list.
        
        Returns:
            List[Dict]: The history of weight snapshots.
        """
        return self.weight_history


class PlatformFactory:
    """
    Manages all three platform instances and routes agents to the correct platform.
    """

    def __init__(self, platforms: Dict[str, Platform]):
        """
        Initialize the PlatformFactory with a dictionary of platforms.
        
        Args:
            platforms (Dict[str, Platform]): A dictionary mapping platform names to instances.
            
        Raises:
            ValueError: If "imdb", "reddit", or "youtube" key is missing.
        """
        required_keys = {"imdb", "reddit", "youtube"}
        if not required_keys.issubset(platforms.keys()):
            raise ValueError(f"Platforms dictionary must contain keys: {required_keys}")
            
        self.platforms: Dict[str, Platform] = platforms

    def get_platform(self, platform_name: str) -> Platform:
        """
        Returns a platform by name.
        
        Args:
            platform_name (str): The name of the platform.
            
        Returns:
            Platform: The platform instance.
            
        Raises:
            KeyError: If the platform name is not found.
        """
        if platform_name not in self.platforms:
            raise KeyError(f"Platform '{platform_name}' not found.")
        return self.platforms[platform_name]

    def get_platform_for_agent(self, agent: Any, timestep: int) -> Platform:
        """
        Returns a platform weighted randomly by agent.platform_weights.
        
        Args:
            agent (Agent): The agent to select a platform for.
            timestep (int): The current timestep.
            
        Returns:
            Platform: The selected platform instance.
        """
        platforms_list = list(agent.platform_weights.keys())
        weights = list(agent.platform_weights.values())
        
        # Normalize weights to sum to 1, as required by numpy.random.choice
        weights = np.array(weights) / np.sum(weights)
        
        selected_platform_name = np.random.choice(platforms_list, p=weights)
        return self.get_platform(selected_platform_name)
