import pytest
import numpy as np
from core.content import BaseContent, IMDBContent, RedditContent, YouTubeContent

def test_base_content_cannot_be_instantiated():
    with pytest.raises(TypeError):
        # Even though we subclassed ABC, we added an explicit check just in case ABC abstract methods weren't fully utilized correctly for this requirement
        class DummyContent(BaseContent):
            pass
        # Wait, ABC will catch it if we try to instantiate BaseContent directly.
        BaseContent(content_id=1, birth_timestep=0)  # type: ignore

def test_imdb_content_instantiation():
    content = IMDBContent(content_id=1, birth_timestep=0)
    assert content.content_id == 1
    assert content.birth_timestep == 0
    assert content.decay_rate == 0.002
    assert content.retirement_threshold == 200
    assert hasattr(content, 'item_bias')
    assert hasattr(content, 'release_era')
    assert hasattr(content, 'critical_consensus')
    assert hasattr(content, 'mainstream_appeal')

def test_reddit_content_instantiation():
    content = RedditContent(content_id=1, birth_timestep=0, community_id=42, content_type="post")
    assert content.content_id == 1
    assert content.birth_timestep == 0
    assert content.community_id == 42
    assert content.decay_rate == 0.05
    assert content.retirement_threshold == 10
    assert 1 <= content.upvotes <= 10
    assert 0 <= content.downvotes <= 3
    assert content.content_type == "post"
    assert content.cross_post_origin is None

def test_youtube_content_instantiation():
    content = YouTubeContent(content_id=1, birth_timestep=0, topic_cluster=7)
    assert content.content_id == 1
    assert content.birth_timestep == 0
    assert content.topic_cluster == 7
    assert content.decay_rate == 0.02
    assert content.retirement_threshold == 100
    assert content.completion_rate_baseline == 0.5
    assert 0.0 <= content.predicted_watch_time <= 1.0
    assert 0.0 <= content.thumbnail_appeal <= 1.0
    assert 0.0 <= content.creator_authority <= 1.0

def test_imdb_latent_features_length():
    content = IMDBContent(content_id=1, birth_timestep=0)
    assert len(content.latent_features) == 25
    assert len(content.ideological_loading_vector) == 25

def test_initial_novelty():
    content = IMDBContent(content_id=1, birth_timestep=0)
    assert content.novelty == 1.0

def test_decay_novelty():
    content = IMDBContent(content_id=1, birth_timestep=0)
    content.decay_novelty(current_timestep=100, decay_rate=0.002)
    # 1.0 - (100 * 0.002) = 0.8
    assert np.isclose(content.novelty, 0.8)

def test_novelty_never_below_zero():
    content = IMDBContent(content_id=1, birth_timestep=0)
    content.decay_novelty(current_timestep=1000, decay_rate=0.002)
    # 1.0 - (1000 * 0.002) = -1.0 -> should be 0.0
    assert content.novelty == 0.0

def test_is_intervention_content_initialization():
    content = IMDBContent(content_id=1, birth_timestep=0)
    assert content.is_intervention_content is False

def test_should_retire():
    content = YouTubeContent(content_id=1, birth_timestep=0, topic_cluster=1)
    
    # Fresh content shouldn't retire
    assert content.should_retire(current_timestep=1) is False
    assert content.consecutive_zero_engagement_timesteps == 0
    
    # Force novelty and engagement to 0
    content.novelty = 0.0
    content.recent_engagement = 0.0
    
    # Advance time to right below threshold
    for t in range(2, 101): # Should be retired when threshold (100) is reached.
        retired = content.should_retire(current_timestep=t)
        if content.consecutive_zero_engagement_timesteps < 100:
            assert retired is False
        else:
            assert retired is True
            
    assert content.consecutive_zero_engagement_timesteps == 99
    assert content.should_retire(101) is True
    assert content.consecutive_zero_engagement_timesteps == 100

def test_belief_position_distribution():
    positions = []
    for i in range(1000):
        content = IMDBContent(content_id=i, birth_timestep=0)
        positions.append(content.position_in_belief_space)
        
    positions = np.array(positions)
    assert np.all(positions >= 0.0)
    assert np.all(positions <= 10.0)
