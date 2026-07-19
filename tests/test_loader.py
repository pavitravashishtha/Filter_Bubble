import os
import pytest
from data.loader import load_imdb_pool, load_reddit_pool, load_youtube_pool, load_all_pools

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def test_imdb_pool_loader():
    csv_path = os.path.join(DATA_DIR, 'imdb', 'imdb_content_pool.csv')
    imdb_items = load_imdb_pool(csv_path)
    
    # 1. load_imdb_pool() returns exactly 1000 IMDBContent objects
    assert len(imdb_items) == 1000
    
    for item in imdb_items:
        # 4. Every IMDBContent object has latent_features of length 25
        assert len(item.latent_features) == 25
        # 5. Every IMDBContent object has ideological_loading_vector of length 25
        assert len(item.ideological_loading_vector) == 25
        # 6. No IMDBContent object has latent_features as None
        assert item.latent_features is not None
        # 9. All position_in_belief_space values across all three pools are within [0.0, 10.0]
        assert 0.0 <= item.position_in_belief_space <= 10.0
        # 10. All novelty values start at 1.0
        assert item.novelty == 1.0

def test_reddit_pool_loader():
    csv_path = os.path.join(DATA_DIR, 'reddit', 'reddit_content_pool.csv')
    reddit_items = load_reddit_pool(csv_path)
    
    # 2. load_reddit_pool() returns exactly 2500 RedditContent objects
    assert len(reddit_items) == 2500
    
    for item in reddit_items:
        # 7 & 8: cross_post_origin is None if -1 in CSV, else integer
        if item.cross_post_origin is not None:
            assert isinstance(item.cross_post_origin, int)
            
        # 9. All position_in_belief_space values across all three pools are within [0.0, 10.0]
        assert 0.0 <= item.position_in_belief_space <= 10.0
        # 10. All novelty values start at 1.0
        assert item.novelty == 1.0

def test_youtube_pool_loader():
    csv_path = os.path.join(DATA_DIR, 'youtube', 'youtube_content_pool.csv')
    youtube_items = load_youtube_pool(csv_path)
    
    # 3. load_youtube_pool() returns exactly 4000 YouTubeContent objects
    assert len(youtube_items) == 4000
    
    for item in youtube_items:
        # 9. All position_in_belief_space values across all three pools are within [0.0, 10.0]
        assert 0.0 <= item.position_in_belief_space <= 10.0
        # 10. All novelty values start at 1.0
        assert item.novelty == 1.0
        # 12. YouTube completion_rate_baseline is 0.5 for all items
        assert item.completion_rate_baseline == 0.5

def test_load_all_pools():
    # 11. load_all_pools() returns dict with exactly three keys: imdb, reddit, youtube
    pools = load_all_pools(DATA_DIR)
    
    assert isinstance(pools, dict)
    assert set(pools.keys()) == {'imdb', 'reddit', 'youtube'}
    
    assert len(pools['imdb']) == 1000
    assert len(pools['reddit']) == 2500
    assert len(pools['youtube']) == 4000
