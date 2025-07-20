"""
External group-by algorithm using √n memory.
"""

import os
import pickle
import tempfile
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar
from collections import defaultdict

from sqrtspace_spacetime.config import config
from sqrtspace_spacetime.collections import SpaceTimeDict

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class GroupByStrategy(Enum):
    """Group-by strategies."""
    HASH_BASED = "hash_based"
    SORT_BASED = "sort_based"
    ADAPTIVE = "adaptive"


def external_groupby(
    data: Iterable[T],
    key_func: Callable[[T], K],
    strategy: GroupByStrategy = GroupByStrategy.ADAPTIVE,
    storage_path: Optional[str] = None
) -> Dict[K, List[T]]:
    """
    Group data by key using external memory.
    
    Args:
        data: Iterable of items to group
        key_func: Function to extract group key
        strategy: Grouping strategy
        storage_path: Path for temporary storage
        
    Returns:
        Dictionary mapping keys to lists of items
    """
    storage_path = storage_path or config.external_storage_path
    
    # Convert to list to get size
    if not isinstance(data, list):
        data = list(data)
    
    n = len(data)
    
    # Small datasets can be grouped in memory
    if n <= 10000:
        result = defaultdict(list)
        for item in data:
            result[key_func(item)].append(item)
        return dict(result)
    
    # Choose strategy
    if strategy == GroupByStrategy.ADAPTIVE:
        strategy = _choose_groupby_strategy(data, key_func)
    
    if strategy == GroupByStrategy.HASH_BASED:
        return _hash_based_groupby(data, key_func, storage_path)
    else:
        return _sort_based_groupby(data, key_func, storage_path)


def external_groupby_aggregate(
    data: Iterable[T],
    key_func: Callable[[T], K],
    value_func: Callable[[T], V],
    agg_func: Callable[[V, V], V],
    initial: Optional[V] = None,
    storage_path: Optional[str] = None
) -> Dict[K, V]:
    """
    Group by with aggregation using external memory.
    
    Args:
        data: Iterable of items
        key_func: Function to extract group key
        value_func: Function to extract value for aggregation
        agg_func: Aggregation function (e.g., sum, max)
        initial: Initial value for aggregation
        storage_path: Path for temporary storage
        
    Returns:
        Dictionary mapping keys to aggregated values
    """
    # Use SpaceTimeDict for memory-efficient aggregation
    result = SpaceTimeDict(storage_path=storage_path)
    
    for item in data:
        key = key_func(item)
        value = value_func(item)
        
        if key in result:
            result[key] = agg_func(result[key], value)
        else:
            result[key] = value if initial is None else agg_func(initial, value)
    
    # Convert to regular dict by creating a list first to avoid mutation issues
    return {k: v for k, v in list(result.items())}


def _choose_groupby_strategy(data: List[T], key_func: Callable[[T], K]) -> GroupByStrategy:
    """Choose grouping strategy based on data characteristics."""
    # Sample keys to estimate cardinality
    sample_size = min(1000, len(data))
    sample_keys = set()
    
    for i in range(0, len(data), max(1, len(data) // sample_size)):
        sample_keys.add(key_func(data[i]))
    
    estimated_groups = len(sample_keys) * (len(data) / sample_size)
    
    # If few groups relative to data size, use hash-based
    if estimated_groups < len(data) / 10:
        return GroupByStrategy.HASH_BASED
    else:
        return GroupByStrategy.SORT_BASED


def _hash_based_groupby(
    data: List[T],
    key_func: Callable[[T], K],
    storage_path: str
) -> Dict[K, List[T]]:
    """
    Hash-based grouping with spillover to disk.
    """
    chunk_size = config.calculate_chunk_size(len(data))
    
    # Use SpaceTimeDict for groups
    groups = SpaceTimeDict(threshold=chunk_size // 10, storage_path=storage_path)
    
    for item in data:
        key = key_func(item)
        
        if key in groups:
            group = groups[key]
            group.append(item)
            groups[key] = group
        else:
            groups[key] = [item]
    
    # Convert to regular dict
    return dict(groups.items())


def _sort_based_groupby(
    data: List[T],
    key_func: Callable[[T], K],
    storage_path: str
) -> Dict[K, List[T]]:
    """
    Sort-based grouping.
    """
    from sqrtspace_spacetime.algorithms.external_sort import external_sort_key
    
    # Sort by group key
    sorted_data = external_sort_key(data, key=key_func, storage_path=storage_path)
    
    # Group consecutive items
    result = {}
    current_key = None
    current_group = []
    
    for item in sorted_data:
        item_key = key_func(item)
        
        if item_key != current_key:
            if current_key is not None:
                result[current_key] = current_group
            current_key = item_key
            current_group = [item]
        else:
            current_group.append(item)
    
    # Don't forget the last group
    if current_key is not None:
        result[current_key] = current_group
    
    return result


# Convenience functions for common aggregations

def groupby_count(
    data: Iterable[T],
    key_func: Callable[[T], K]
) -> Dict[K, int]:
    """Count items by group."""
    return external_groupby_aggregate(
        data,
        key_func,
        lambda x: 1,
        lambda a, b: a + b,
        initial=0
    )


def groupby_sum(
    data: Iterable[T],
    key_func: Callable[[T], K],
    value_func: Callable[[T], float]
) -> Dict[K, float]:
    """Sum values by group."""
    return external_groupby_aggregate(
        data,
        key_func,
        value_func,
        lambda a, b: a + b,
        initial=0.0
    )


def groupby_avg(
    data: Iterable[T],
    key_func: Callable[[T], K],
    value_func: Callable[[T], float]
) -> Dict[K, float]:
    """Average values by group."""
    # First get sums and counts
    sums = defaultdict(float)
    counts = defaultdict(int)
    
    for item in data:
        key = key_func(item)
        value = value_func(item)
        sums[key] += value
        counts[key] += 1
    
    # Calculate averages
    return {key: sums[key] / counts[key] for key in sums}


def groupby_max(
    data: Iterable[T],
    key_func: Callable[[T], K],
    value_func: Callable[[T], V]
) -> Dict[K, V]:
    """Get maximum value by group."""
    return external_groupby_aggregate(
        data,
        key_func,
        value_func,
        max
    )


def groupby_min(
    data: Iterable[T],
    key_func: Callable[[T], K],
    value_func: Callable[[T], V]
) -> Dict[K, V]:
    """Get minimum value by group."""
    return external_groupby_aggregate(
        data,
        key_func,
        value_func,
        min
    )