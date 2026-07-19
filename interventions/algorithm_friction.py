from typing import Dict, Any

def apply_algorithm_friction(platforms: Dict[str, Any], friction_factor: float = 0.5) -> None:
    """
    Reduces valence weight on YouTube platform.
    """
    platforms["youtube"].apply_friction(friction_factor)

def get_friction_description() -> str:
    """
    Returns description string of what this intervention does.
    """
    return "Applies algorithmic friction by reducing the weight of emotional valence in recommendations."
