from typing import Dict, Any

def apply_algorithm_friction(platforms: Dict[str, Any], friction_factor: float = 0.5) -> None:
    """
    Reduces emotional amplification across all three platforms.
    """
    # YouTube — reduce valence weight
    if "youtube" in platforms:
        platforms["youtube"].apply_friction(
            friction_factor
        )
    
    # IMDB — reduce learning rate to slow 
    # taste profile updates
    if "imdb" in platforms:
        platforms["imdb"].learning_rate *= friction_factor
    
    # Reddit — increase decay constant to 
    # make posts age faster, reducing echo chamber momentum
    if "reddit" in platforms:
        platforms["reddit"].decay_constant *= (
            1.0 / friction_factor
        )

def get_friction_description() -> str:
    return (
        "Algorithm friction reduces emotional amplification "
        "across all three platforms: suppresses YouTube "
        "valence weight, slows IMDB taste learning, and "
        "accelerates Reddit content decay."
    )
