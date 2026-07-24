import os
import json
import pandas as pd
import numpy as np
from core.content import IMDBContent, RedditContent, YouTubeContent

def load_imdb_pool(csv_path: str, latent_dims: int = 25) -> list:
    """
    Reads imdb_content_pool.csv and converts each row into an IMDBContent object.
    
    Args:
        csv_path (str): Path to the IMDB CSV file.
        latent_dims (int): Dimension of latent features to generate.
        
    Returns:
        list: A list of IMDBContent objects.
    """
    df = pd.read_csv(csv_path, engine='python')
    df = df.fillna(0.5)
    
    np.random.seed(42)
    items = []
    for _, row in df.iterrows():
        content = IMDBContent(
            content_id=int(row['content_id'].split('_')[-1]),
            birth_timestep=int(row['birth_timestep'])
        )
        
        # Override properties that are parsed from JSON
        content.genre_list = json.loads(row['genre_list'])
        content.genre_vector = json.loads(row['genre_vector'])
        
        # Set explicitly required properties
        content.release_era = int(row['release_era'])
        content.critical_consensus = float(row['critical_consensus'])
        content.mainstream_appeal = float(row['mainstream_appeal'])
        
        content.position_in_belief_space = float(row['position_in_belief_space'])
        content.emotional_valence = float(row['emotional_valence'])
        content.engagement_potential = float(row['engagement_potential'])
        content.quality_signal = float(row['quality_signal'])
        content.novelty = float(row['novelty'])
        
        # Attach other columns from CSV
        for col in df.columns:
            if col not in ['content_id', 'birth_timestep', 'genre_list', 'genre_vector', 
                           'release_era', 'critical_consensus', 'mainstream_appeal', 
                           'position_in_belief_space', 'emotional_valence', 
                           'engagement_potential', 'quality_signal', 'novelty']:
                setattr(content, col, row[col])
        
        # Generate dimensions
        content.latent_features = np.random.normal(0, 0.1, latent_dims)
        content.ideological_loading_vector = np.random.normal(0, 0.05, latent_dims)
        content.item_bias = float(np.random.normal(0, 0.1))
        
        items.append(content)
        
    return items

def load_reddit_pool(csv_path: str) -> list:
    """
    Reads reddit_content_pool.csv and converts each row into a RedditContent object.
    
    Args:
        csv_path (str): Path to the Reddit CSV file.
        
    Returns:
        list: A list of RedditContent objects.
    """
    df = pd.read_csv(csv_path, engine='python')
    df = df.fillna(0.5)
    
    items = []
    for _, row in df.iterrows():
        content = RedditContent(
            content_id=int(row['content_id'].split('_')[-1]),
            birth_timestep=int(row['birth_timestep']),
            community_id=int(row['community_id']),
            content_type=str(row['content_type'])
        )
        
        content.upvotes = int(row['upvotes'])
        content.downvotes = int(row['downvotes'])
        content.timestamp = int(row['timestamp'])
        
        cross_post_origin = None if int(row["cross_post_origin"]) == -1 else int(row["cross_post_origin"])
        content.cross_post_origin = cross_post_origin
        
        content.position_in_belief_space = float(row['position_in_belief_space'])
        content.emotional_valence = float(row['emotional_valence'])
        content.engagement_potential = float(row['engagement_potential'])
        content.quality_signal = float(row['quality_signal'])
        content.novelty = float(row['novelty'])
        
        items.append(content)
        
    return items

def load_youtube_pool(csv_path: str) -> list:
    """
    Reads youtube_content_pool.csv and converts each row into a YouTubeContent object.
    
    Args:
        csv_path (str): Path to the YouTube CSV file.
        
    Returns:
        list: A list of YouTubeContent objects.
    """
    df = pd.read_csv(csv_path, engine='python')
    df = df.fillna(0.5)
    
    items = []
    for _, row in df.iterrows():
        content = YouTubeContent(
            content_id=int(row['content_id'].split('_')[-1]),
            birth_timestep=int(row['birth_timestep']),
            topic_cluster=int(row['topic_cluster'])
        )
        
        content.predicted_watch_time = float(row['predicted_watch_time'])
        content.completion_rate_baseline = float(row['completion_rate_baseline'])
        content.thumbnail_appeal = float(row['thumbnail_appeal'])
        content.creator_authority = float(row['creator_authority'])
        
        content.position_in_belief_space = float(row['position_in_belief_space'])
        content.emotional_valence = float(row['emotional_valence'])
        content.engagement_potential = float(row['engagement_potential'])
        content.quality_signal = float(row['quality_signal'])
        content.novelty = float(row['novelty'])
        
        items.append(content)
        
    return items

def load_all_pools(data_dir: str, latent_dims: int = 25) -> dict:
    """
    Loads all content pools from their respective CSV files.
    
    Args:
        data_dir (str): Base directory containing the data folders.
        latent_dims (int): Dimension of latent features for IMDB content.
        
    Returns:
        dict: A dictionary containing lists of content objects for each platform.
    """
    imdb_path = os.path.join(data_dir, "imdb", "imdb_content_pool.csv")
    reddit_path = os.path.join(data_dir, "reddit", "reddit_content_pool.csv")
    youtube_path = os.path.join(data_dir, "youtube", "youtube_content_pool.csv")
    
    return {
        "imdb": load_imdb_pool(imdb_path, latent_dims),
        "reddit": load_reddit_pool(reddit_path),
        "youtube": load_youtube_pool(youtube_path)
    }
