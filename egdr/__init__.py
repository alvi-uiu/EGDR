"""EGDR: Entropy-Gated Dynamic Rectification for object hallucination in LVLMs."""

from .direction import SteeringDirection, load_direction
from .gate import effective_threshold, normalized_entropy, rescale_entropy, risk_score
from .hooks import EntropyTracker, GatedSteeringHook, GateSpec, Steering

__version__ = "1.0.0"

__all__ = [
    "EntropyTracker",
    "GatedSteeringHook",
    "GateSpec",
    "Steering",
    "SteeringDirection",
    "effective_threshold",
    "load_direction",
    "normalized_entropy",
    "rescale_entropy",
    "risk_score",
]
