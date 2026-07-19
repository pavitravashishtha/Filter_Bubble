import pytest
from core.platform import Platform, PlatformFactory
from typing import List, Dict, Any, Optional

class ValidPlatform(Platform):
    @property
    def platform_name(self) -> str:
        return "valid_platform"

    def serve_content(self, agent: Any, content_pool: List[Any], timestep: int) -> Any:
        return None

    def learn_from_feedback(self, agent: Any, content: Any, engaged: bool, engagement_strength: float) -> None:
        pass

    def generate_candidates(self, agent: Any, content_pool: List[Any], intervention: Optional[Dict[str, Any]] = None) -> List[Any]:
        return []

class DummyAgent:
    def __init__(self, platform_weights: Dict[str, float]):
        self.platform_weights = platform_weights


def test_platform_cannot_be_instantiated():
    """1. Platform abstract class cannot be instantiated directly"""
    with pytest.raises(TypeError):
        Platform(algo_weight=0.5, social_weight=0.5)

def test_missing_serve_content():
    """2. A concrete subclass without serve_content raises TypeError"""
    class InvalidPlatform(Platform):
        @property
        def platform_name(self) -> str:
            return "invalid"
        def learn_from_feedback(self, agent, content, engaged, engagement_strength):
            pass
        def generate_candidates(self, agent, content_pool, intervention=None):
            return []
            
    with pytest.raises(TypeError):
        InvalidPlatform(algo_weight=0.5, social_weight=0.5)

def test_missing_learn_from_feedback():
    """3. A concrete subclass without learn_from_feedback raises TypeError"""
    class InvalidPlatform(Platform):
        @property
        def platform_name(self) -> str:
            return "invalid"
        def serve_content(self, agent, content_pool, timestep):
            pass
        def generate_candidates(self, agent, content_pool, intervention=None):
            return []
            
    with pytest.raises(TypeError):
        InvalidPlatform(algo_weight=0.5, social_weight=0.5)

def test_missing_generate_candidates():
    """4. A concrete subclass without generate_candidates raises TypeError"""
    class InvalidPlatform(Platform):
        @property
        def platform_name(self) -> str:
            return "invalid"
        def serve_content(self, agent, content_pool, timestep):
            pass
        def learn_from_feedback(self, agent, content, engaged, engagement_strength):
            pass
            
    with pytest.raises(TypeError):
        InvalidPlatform(algo_weight=0.5, social_weight=0.5)

def test_valid_platform_instantiates():
    """5. A valid concrete subclass with all methods instantiates correctly"""
    platform = ValidPlatform(algo_weight=0.6, social_weight=0.4)
    assert platform.algo_weight == 0.6
    assert platform.social_weight == 0.4
    assert platform.platform_name == "valid_platform"

def test_invalid_weights():
    """6. algo_weight + social_weight not equal to 1.0 raises ValueError"""
    with pytest.raises(ValueError, match="algo_weight and social_weight must sum to 1.0"):
        ValidPlatform(algo_weight=0.5, social_weight=0.6)

def test_platform_factory_raises_key_error():
    """7. PlatformFactory raises KeyError for unknown platform name"""
    platforms = {
        "imdb": ValidPlatform(0.5, 0.5),
        "reddit": ValidPlatform(0.5, 0.5),
        "youtube": ValidPlatform(0.5, 0.5)
    }
    factory = PlatformFactory(platforms)
    with pytest.raises(KeyError, match="Platform 'unknown' not found."):
        factory.get_platform("unknown")

def test_platform_factory_raises_value_error_missing_keys():
    """8. PlatformFactory raises ValueError if missing any of the three platform keys"""
    platforms = {
        "imdb": ValidPlatform(0.5, 0.5),
        "reddit": ValidPlatform(0.5, 0.5)
    }
    with pytest.raises(ValueError, match="Platforms dictionary must contain keys"):
        PlatformFactory(platforms)

def test_get_platform_for_agent():
    """9. get_platform_for_agent() returns a platform — run 1000 times and verify only valid platform names returned"""
    platforms = {
        "imdb": ValidPlatform(0.1, 0.9),
        "reddit": ValidPlatform(0.2, 0.8),
        "youtube": ValidPlatform(0.3, 0.7)
    }
    factory = PlatformFactory(platforms)
    
    agent = DummyAgent(platform_weights={
        "imdb": 0.2,
        "reddit": 0.5,
        "youtube": 0.3
    })
    
    counts = {"imdb": 0, "reddit": 0, "youtube": 0}
    for _ in range(1000):
        platform = factory.get_platform_for_agent(agent, timestep=1)
        assert platform in platforms.values()
        for k, v in platforms.items():
            if platform is v:
                counts[k] += 1
                
    assert counts["imdb"] > 0
    assert counts["reddit"] > 0
    assert counts["youtube"] > 0
    
    # We also check that only valid platform instances were returned
    assert sum(counts.values()) == 1000

def test_weight_history_starts_empty():
    """10. weight_history starts as empty list where these task completed"""
    platform = ValidPlatform(algo_weight=0.5, social_weight=0.5)
    assert platform.get_weight_history() == []
    assert platform.weight_history == []
