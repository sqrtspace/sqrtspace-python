"""Memory-efficient collections using √n space-time tradeoffs."""

from sqrtspace_spacetime.collections.spacetime_array import SpaceTimeArray
from sqrtspace_spacetime.collections.spacetime_dict import SpaceTimeDict

__all__ = [
    "SpaceTimeArray",
    "SpaceTimeDict",
]