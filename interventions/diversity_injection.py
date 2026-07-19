from typing import Any

def activate_diversity_injection(config: Any) -> None:
    """
    Sets config active_intervention flag.
    """
    config.active_intervention = "diversity_injection"

def deactivate_diversity_injection(config: Any) -> None:
    """
    Clears the active_intervention flag.
    """
    config.active_intervention = None

def get_diversity_description() -> str:
    """
    Returns description string of what this intervention does.
    """
    return "Injects ideologically diverse content into the recommendation queues."
