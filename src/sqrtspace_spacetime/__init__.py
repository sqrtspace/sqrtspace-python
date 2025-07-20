"""
Ubiquity SpaceTime: Memory-efficient algorithms using √n space-time tradeoffs.

This package implements Williams' theoretical computer science results showing
that many algorithms can achieve better memory usage by accepting slightly
slower runtime.
"""

from sqrtspace_spacetime.config import SpaceTimeConfig
from sqrtspace_spacetime.collections import SpaceTimeArray, SpaceTimeDict
from sqrtspace_spacetime.algorithms import external_sort, external_groupby
from sqrtspace_spacetime.streams import Stream
from sqrtspace_spacetime.memory import MemoryMonitor, MemoryPressureLevel

__version__ = "0.1.0"
__author__ = "Ubiquity SpaceTime Contributors"
__license__ = "Apache-2.0"

__all__ = [
    "SpaceTimeConfig",
    "SpaceTimeArray",
    "SpaceTimeDict",
    "external_sort",
    "external_groupby",
    "Stream",
    "MemoryMonitor",
    "MemoryPressureLevel",
]

# Configure default settings
SpaceTimeConfig.set_defaults()