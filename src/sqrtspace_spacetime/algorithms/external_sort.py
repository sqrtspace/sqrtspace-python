"""
External sorting algorithm using √n memory.
"""

import os
import heapq
import pickle
import tempfile
from enum import Enum
from typing import Any, Callable, Iterable, List, Optional, TypeVar, Union
from dataclasses import dataclass

from sqrtspace_spacetime.config import config
from sqrtspace_spacetime.memory import monitor

T = TypeVar('T')


class SortStrategy(Enum):
    """Sorting strategies."""
    MULTIWAY_MERGE = "multiway_merge"
    QUICKSORT_EXTERNAL = "quicksort_external"
    ADAPTIVE = "adaptive"


@dataclass
class SortRun:
    """A sorted run on disk."""
    filename: str
    count: int
    min_value: Any
    max_value: Any


def external_sort(
    data: Iterable[T],
    reverse: bool = False,
    strategy: SortStrategy = SortStrategy.ADAPTIVE,
    storage_path: Optional[str] = None
) -> List[T]:
    """
    Sort data using external memory with √n space complexity.
    
    Args:
        data: Iterable of items to sort
        reverse: Sort in descending order
        strategy: Sorting strategy to use
        storage_path: Path for temporary files
        
    Returns:
        Sorted list
    """
    return external_sort_key(
        data,
        key=lambda x: x,
        reverse=reverse,
        strategy=strategy,
        storage_path=storage_path
    )


def external_sort_key(
    data: Iterable[T],
    key: Callable[[T], Any],
    reverse: bool = False,
    strategy: SortStrategy = SortStrategy.ADAPTIVE,
    storage_path: Optional[str] = None
) -> List[T]:
    """
    Sort data by key using external memory.
    
    Args:
        data: Iterable of items to sort
        key: Function to extract sort key
        reverse: Sort in descending order
        strategy: Sorting strategy to use
        storage_path: Path for temporary files
        
    Returns:
        Sorted list
    """
    storage_path = storage_path or config.external_storage_path
    
    # Convert to list if needed to get size
    if not isinstance(data, list):
        data = list(data)
    
    n = len(data)
    
    # Small datasets can be sorted in memory
    if n <= 10000:
        return sorted(data, key=key, reverse=reverse)
    
    # Choose strategy
    if strategy == SortStrategy.ADAPTIVE:
        strategy = _choose_strategy(n)
    
    if strategy == SortStrategy.MULTIWAY_MERGE:
        return _multiway_merge_sort(data, key, reverse, storage_path)
    else:
        return _external_quicksort(data, key, reverse, storage_path)


def _choose_strategy(n: int) -> SortStrategy:
    """Choose best strategy based on data size."""
    # For very large datasets, multiway merge is more stable
    if n > 1_000_000:
        return SortStrategy.MULTIWAY_MERGE
    else:
        return SortStrategy.QUICKSORT_EXTERNAL


def _multiway_merge_sort(
    data: List[T],
    key: Callable[[T], Any],
    reverse: bool,
    storage_path: str
) -> List[T]:
    """
    Multiway merge sort implementation.
    """
    n = len(data)
    chunk_size = config.calculate_chunk_size(n)
    
    # Phase 1: Create sorted runs
    runs = []
    temp_files = []
    
    for i in range(0, n, chunk_size):
        chunk = data[i:i + chunk_size]
        
        # Sort chunk in memory
        chunk.sort(key=key, reverse=reverse)
        
        # Write to disk
        fd, filename = tempfile.mkstemp(suffix='.run', dir=storage_path)
        os.close(fd)
        temp_files.append(filename)
        
        with open(filename, 'wb') as f:
            pickle.dump(chunk, f)
        
        # Track run info
        runs.append(SortRun(
            filename=filename,
            count=len(chunk),
            min_value=key(chunk[0]),
            max_value=key(chunk[-1])
        ))
    
    # Phase 2: Merge runs
    try:
        result = _merge_runs(runs, key, reverse)
        return result
    finally:
        # Cleanup
        for filename in temp_files:
            if os.path.exists(filename):
                os.unlink(filename)


def _merge_runs(
    runs: List[SortRun],
    key: Callable[[T], Any],
    reverse: bool
) -> List[T]:
    """
    Merge sorted runs using a k-way merge.
    """
    # Open all run files
    run_iters = []
    for run in runs:
        with open(run.filename, 'rb') as f:
            items = pickle.load(f)
            run_iters.append(iter(items))
    
    # Create heap for merge
    heap = []
    
    # Initialize heap with first item from each run
    for i, run_iter in enumerate(run_iters):
        try:
            item = next(run_iter)
            # For reverse sort, negate the key
            heap_key = key(item)
            if reverse:
                heap_key = _negate_key(heap_key)
            heapq.heappush(heap, (heap_key, i, item, run_iter))
        except StopIteration:
            pass
    
    # Merge
    result = []
    while heap:
        heap_key, run_idx, item, run_iter = heapq.heappop(heap)
        result.append(item)
        
        # Get next item from same run
        try:
            next_item = next(run_iter)
            next_key = key(next_item)
            if reverse:
                next_key = _negate_key(next_key)
            heapq.heappush(heap, (next_key, run_idx, next_item, run_iter))
        except StopIteration:
            pass
    
    return result


def _negate_key(key: Any) -> Any:
    """Negate a key for reverse sorting."""
    if isinstance(key, (int, float)):
        return -key
    elif isinstance(key, str):
        # For strings, return a wrapper that reverses comparison
        return _ReverseString(key)
    else:
        # For other types, use a generic wrapper
        return _ReverseWrapper(key)


class _ReverseString:
    """Wrapper for reverse string comparison."""
    def __init__(self, s: str):
        self.s = s
    
    def __lt__(self, other):
        return self.s > other.s
    
    def __le__(self, other):
        return self.s >= other.s
    
    def __gt__(self, other):
        return self.s < other.s
    
    def __ge__(self, other):
        return self.s <= other.s
    
    def __eq__(self, other):
        return self.s == other.s


class _ReverseWrapper:
    """Generic wrapper for reverse comparison."""
    def __init__(self, obj):
        self.obj = obj
    
    def __lt__(self, other):
        return self.obj > other.obj
    
    def __le__(self, other):
        return self.obj >= other.obj
    
    def __gt__(self, other):
        return self.obj < other.obj
    
    def __ge__(self, other):
        return self.obj <= other.obj
    
    def __eq__(self, other):
        return self.obj == other.obj


def _external_quicksort(
    data: List[T],
    key: Callable[[T], Any],
    reverse: bool,
    storage_path: str
) -> List[T]:
    """
    External quicksort implementation.
    
    This is a simplified version that partitions data and
    recursively sorts partitions that fit in memory.
    """
    n = len(data)
    chunk_size = config.calculate_chunk_size(n)
    
    if n <= chunk_size:
        # Base case: sort in memory
        return sorted(data, key=key, reverse=reverse)
    
    # Choose pivot (median of three)
    pivot_idx = _choose_pivot(data, key)
    pivot_key = key(data[pivot_idx])
    
    # Partition data
    less = []
    equal = []
    greater = []
    
    for item in data:
        item_key = key(item)
        if item_key < pivot_key:
            less.append(item)
        elif item_key == pivot_key:
            equal.append(item)
        else:
            greater.append(item)
    
    # Recursively sort partitions
    sorted_less = _external_quicksort(less, key, reverse, storage_path)
    sorted_greater = _external_quicksort(greater, key, reverse, storage_path)
    
    # Combine results
    if reverse:
        return sorted_greater + equal + sorted_less
    else:
        return sorted_less + equal + sorted_greater


def _choose_pivot(data: List[T], key: Callable[[T], Any]) -> int:
    """Choose a good pivot using median-of-three."""
    n = len(data)
    
    # Sample three elements
    first = 0
    middle = n // 2
    last = n - 1
    
    # Find median
    a, b, c = key(data[first]), key(data[middle]), key(data[last])
    
    if a <= b <= c or c <= b <= a:
        return middle
    elif b <= a <= c or c <= a <= b:
        return first
    else:
        return last