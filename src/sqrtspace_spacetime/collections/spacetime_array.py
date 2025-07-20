"""
SpaceTimeArray: A memory-efficient array that automatically spills to disk.
"""

import os
import pickle
import tempfile
import weakref
from typing import Any, Iterator, Optional, Union, List
from collections.abc import MutableSequence

from sqrtspace_spacetime.config import config
from sqrtspace_spacetime.memory import monitor, MemoryPressureLevel


class SpaceTimeArray(MutableSequence):
    """
    A list-like container that automatically manages memory usage by
    spilling to disk when threshold is reached.
    """
    
    _instances = weakref.WeakSet()
    
    def __init__(self, threshold: Optional[Union[int, str]] = None, storage_path: Optional[str] = None):
        """
        Initialize SpaceTimeArray.
        
        Args:
            threshold: Number of items to keep in memory (None or 'auto' for automatic)
            storage_path: Path for external storage (None for temp)
        """
        if threshold == 'auto' or threshold is None:
            self.threshold = config.calculate_chunk_size(10000)
        else:
            self.threshold = int(threshold)
        self.storage_path = storage_path or config.external_storage_path
        
        self._hot_data: List[Any] = []
        self._cold_indices: set = set()
        self._cold_storage: Optional[str] = None
        self._length = 0
        self._cold_file_handle = None
        
        # Register for memory pressure handling
        SpaceTimeArray._instances.add(self)
        
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, index: Union[int, slice]) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        
        if index < 0:
            index += self._length
        
        if not 0 <= index < self._length:
            raise IndexError("list index out of range")
        
        # Check if in hot storage
        if index not in self._cold_indices:
            hot_index = index - len(self._cold_indices)
            return self._hot_data[hot_index]
        
        # Load from cold storage
        return self._load_from_cold(index)
    
    def __setitem__(self, index: Union[int, slice], value: Any) -> None:
        if isinstance(index, slice):
            for i, v in zip(range(*index.indices(len(self))), value):
                self[i] = v
            return
        
        if index < 0:
            index += self._length
        
        if not 0 <= index < self._length:
            raise IndexError("list assignment index out of range")
        
        if index not in self._cold_indices:
            hot_index = index - len(self._cold_indices)
            self._hot_data[hot_index] = value
        else:
            # Update cold storage
            self._update_cold(index, value)
    
    def __delitem__(self, index: Union[int, slice]) -> None:
        if isinstance(index, slice):
            # Delete in reverse order to maintain indices
            for i in reversed(range(*index.indices(len(self)))):
                del self[i]
            return
        
        if index < 0:
            index += self._length
        
        if not 0 <= index < self._length:
            raise IndexError("list index out of range")
        
        # This is complex with cold storage, so we'll reload everything
        all_data = list(self)
        del all_data[index]
        self.clear()
        self.extend(all_data)
    
    def insert(self, index: int, value: Any) -> None:
        if index < 0:
            index += self._length
        index = max(0, min(index, self._length))
        
        # Simple implementation: reload all, insert, save back
        all_data = list(self)
        all_data.insert(index, value)
        self.clear()
        self.extend(all_data)
    
    def append(self, value: Any) -> None:
        """Append an item to the array."""
        self._hot_data.append(value)
        self._length += 1
        
        # Check if we need to spill
        if len(self._hot_data) > self.threshold:
            self._check_and_spill()
    
    def extend(self, iterable) -> None:
        """Extend array with items from iterable."""
        for item in iterable:
            self.append(item)
    
    def clear(self) -> None:
        """Remove all items."""
        self._hot_data.clear()
        self._cold_indices.clear()
        self._length = 0
        
        if self._cold_storage and os.path.exists(self._cold_storage):
            os.unlink(self._cold_storage)
        self._cold_storage = None
    
    def __iter__(self) -> Iterator[Any]:
        """Iterate over all items."""
        # First yield cold items
        for idx in sorted(self._cold_indices):
            yield self._load_from_cold(idx)
        
        # Then hot items
        for item in self._hot_data:
            yield item
    
    def _check_and_spill(self) -> None:
        """Check memory pressure and spill to disk if needed."""
        # Check memory pressure
        pressure = monitor.check_memory_pressure()
        
        if pressure >= MemoryPressureLevel.MEDIUM or len(self._hot_data) > self.threshold:
            self._spill_to_disk()
    
    def _spill_to_disk(self) -> None:
        """Spill oldest items to disk."""
        if not self._cold_storage:
            fd, self._cold_storage = tempfile.mkstemp(
                suffix='.spacetime',
                dir=self.storage_path
            )
            os.close(fd)
        
        # Determine how many items to spill
        spill_count = len(self._hot_data) // 2
        
        # Load existing cold data
        cold_data = {}
        if os.path.exists(self._cold_storage):
            with open(self._cold_storage, 'rb') as f:
                try:
                    cold_data = pickle.load(f)
                except EOFError:
                    cold_data = {}
        
        # Move items to cold storage
        current_cold_size = len(self._cold_indices)
        for i in range(spill_count):
            cold_data[current_cold_size + i] = self._hot_data[i]
            self._cold_indices.add(current_cold_size + i)
        
        # Remove from hot storage
        self._hot_data = self._hot_data[spill_count:]
        
        # Save cold data
        with open(self._cold_storage, 'wb') as f:
            pickle.dump(cold_data, f)
    
    def _load_from_cold(self, index: int) -> Any:
        """Load an item from cold storage."""
        if not self._cold_storage or not os.path.exists(self._cold_storage):
            raise IndexError(f"Cold storage index {index} not found")
        
        with open(self._cold_storage, 'rb') as f:
            cold_data = pickle.load(f)
        
        return cold_data.get(index)
    
    def _update_cold(self, index: int, value: Any) -> None:
        """Update an item in cold storage."""
        if not self._cold_storage:
            return
        
        with open(self._cold_storage, 'rb') as f:
            cold_data = pickle.load(f)
        
        cold_data[index] = value
        
        with open(self._cold_storage, 'wb') as f:
            pickle.dump(cold_data, f)
    
    def memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        # Rough estimate - actual usage may vary
        return len(self._hot_data) * 50  # Assume 50 bytes per item average
    
    def spill_to_disk(self, path: Optional[str] = None) -> None:
        """Force spill all data to disk."""
        if path:
            self.storage_path = path
        
        while self._hot_data:
            self._spill_to_disk()
    
    def load_to_memory(self) -> None:
        """Load all data back to memory."""
        if not self._cold_storage or not self._cold_indices:
            return
        
        # Load cold data
        with open(self._cold_storage, 'rb') as f:
            cold_data = pickle.load(f)
        
        # Rebuild array in correct order
        all_data = []
        cold_count = 0
        hot_count = 0
        
        for i in range(self._length):
            if i in self._cold_indices:
                all_data.append(cold_data[i])
                cold_count += 1
            else:
                all_data.append(self._hot_data[hot_count])
                hot_count += 1
        
        # Reset storage
        self._hot_data = all_data
        self._cold_indices.clear()
        
        if os.path.exists(self._cold_storage):
            os.unlink(self._cold_storage)
        self._cold_storage = None
    
    def __del__(self):
        """Clean up temporary files."""
        if self._cold_storage and os.path.exists(self._cold_storage):
            try:
                os.unlink(self._cold_storage)
            except:
                pass
    
    @classmethod
    def handle_memory_pressure(cls, level: MemoryPressureLevel) -> None:
        """Class method to handle memory pressure for all instances."""
        if level >= MemoryPressureLevel.HIGH:
            for instance in cls._instances:
                if instance._hot_data:
                    instance._spill_to_disk()