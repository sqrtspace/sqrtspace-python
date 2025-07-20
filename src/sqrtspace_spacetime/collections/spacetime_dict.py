"""
SpaceTimeDict: A memory-efficient dictionary with automatic spillover.
"""

import os
import pickle
import tempfile
import time
from typing import Any, Dict, Iterator, Optional, Tuple
from collections import OrderedDict
from collections.abc import MutableMapping

from sqrtspace_spacetime.config import config
from sqrtspace_spacetime.memory import monitor, MemoryPressureLevel


class SpaceTimeDict(MutableMapping):
    """
    A dictionary that automatically manages memory by moving least-recently-used
    items to disk storage.
    """
    
    def __init__(self, 
                 threshold: Optional[int] = None,
                 storage_path: Optional[str] = None,
                 use_lru: bool = True):
        """
        Initialize SpaceTimeDict.
        
        Args:
            threshold: Number of items to keep in memory
            storage_path: Path for external storage
            use_lru: Use LRU eviction policy
        """
        self.threshold = threshold or config.calculate_chunk_size(10000)
        self.storage_path = storage_path or config.external_storage_path
        self.use_lru = use_lru
        
        # Hot storage (in memory)
        if use_lru:
            self._hot_data: Dict[Any, Any] = OrderedDict()
        else:
            self._hot_data: Dict[Any, Any] = {}
        
        # Cold storage tracking
        self._cold_keys: set = set()
        self._cold_storage: Optional[str] = None
        self._cold_index: Dict[Any, Tuple[int, int]] = {}  # key -> (offset, size)
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._last_access: Dict[Any, float] = {}
    
    def __len__(self) -> int:
        return len(self._hot_data) + len(self._cold_keys)
    
    def __getitem__(self, key: Any) -> Any:
        # Check hot storage first
        if key in self._hot_data:
            self._hits += 1
            if self.use_lru:
                # Move to end (most recent)
                self._hot_data.move_to_end(key)
            self._last_access[key] = time.time()
            return self._hot_data[key]
        
        # Check cold storage
        if key in self._cold_keys:
            self._misses += 1
            value = self._load_from_cold(key)
            
            # Promote to hot storage
            self._promote_to_hot(key, value)
            
            return value
        
        raise KeyError(key)
    
    def __setitem__(self, key: Any, value: Any) -> None:
        # If key exists in cold storage, remove it
        if key in self._cold_keys:
            self._cold_keys.remove(key)
            # Note: We don't actually remove from file to avoid rewriting
        
        # Add to hot storage
        self._hot_data[key] = value
        self._last_access[key] = time.time()
        
        # Check if we need to evict
        if len(self._hot_data) > self.threshold:
            self._evict_to_cold()
    
    def __delitem__(self, key: Any) -> None:
        if key in self._hot_data:
            del self._hot_data[key]
            self._last_access.pop(key, None)
        elif key in self._cold_keys:
            self._cold_keys.remove(key)
            self._cold_index.pop(key, None)
        else:
            raise KeyError(key)
    
    def __iter__(self) -> Iterator[Any]:
        # Iterate hot keys first
        yield from self._hot_data
        # Then cold keys
        yield from self._cold_keys
    
    def __contains__(self, key: Any) -> bool:
        return key in self._hot_data or key in self._cold_keys
    
    def keys(self):
        """Return a view of all keys."""
        return list(self._hot_data.keys()) + list(self._cold_keys)
    
    def values(self):
        """Return a view of all values."""
        for key in self:
            yield self[key]
    
    def items(self):
        """Return a view of all key-value pairs."""
        for key in self:
            yield (key, self[key])
    
    def clear(self) -> None:
        """Remove all items."""
        self._hot_data.clear()
        self._cold_keys.clear()
        self._cold_index.clear()
        self._last_access.clear()
        
        if self._cold_storage and os.path.exists(self._cold_storage):
            os.unlink(self._cold_storage)
        self._cold_storage = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        
        return {
            "hot_items": len(self._hot_data),
            "cold_items": len(self._cold_keys),
            "total_items": len(self),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "memory_usage": self.memory_usage(),
        }
    
    def _evict_to_cold(self) -> None:
        """Evict least recently used items to cold storage."""
        evict_count = max(1, len(self._hot_data) // 4)  # Evict 25%
        
        if not self._cold_storage:
            fd, self._cold_storage = tempfile.mkstemp(
                suffix='.spacetime_dict',
                dir=self.storage_path
            )
            os.close(fd)
        
        # Select items to evict
        if self.use_lru:
            # OrderedDict: oldest items are first
            evict_keys = list(self._hot_data.keys())[:evict_count]
        else:
            # Use access time
            sorted_keys = sorted(
                self._hot_data.keys(),
                key=lambda k: self._last_access.get(k, 0)
            )
            evict_keys = sorted_keys[:evict_count]
        
        # Write to cold storage
        with open(self._cold_storage, 'ab') as f:
            for key in evict_keys:
                value = self._hot_data[key]
                offset = f.tell()
                
                # Serialize key-value pair
                data = pickle.dumps((key, value))
                size = len(data)
                
                # Write size header and data
                f.write(size.to_bytes(4, 'little'))
                f.write(data)
                
                # Update indices
                self._cold_index[key] = (offset, size + 4)
                self._cold_keys.add(key)
                
                # Remove from hot storage
                del self._hot_data[key]
    
    def _load_from_cold(self, key: Any) -> Any:
        """Load a value from cold storage."""
        if key not in self._cold_index:
            raise KeyError(key)
        
        offset, size = self._cold_index[key]
        
        with open(self._cold_storage, 'rb') as f:
            f.seek(offset)
            size_bytes = f.read(4)
            data_size = int.from_bytes(size_bytes, 'little')
            data = f.read(data_size)
            
            stored_key, value = pickle.loads(data)
            assert stored_key == key
            
            return value
    
    def _promote_to_hot(self, key: Any, value: Any) -> None:
        """Promote a cold item to hot storage."""
        # Remove from cold tracking
        self._cold_keys.remove(key)
        
        # Add to hot storage
        self._hot_data[key] = value
        self._last_access[key] = time.time()
        
        # Check if we need to evict something else
        if len(self._hot_data) > self.threshold:
            self._evict_to_cold()
    
    def memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        # Rough estimate
        return len(self._hot_data) * 100  # Assume 100 bytes per item average
    
    def compact(self) -> None:
        """Compact cold storage by removing deleted entries."""
        if not self._cold_storage or not self._cold_keys:
            return
        
        # Create new file
        fd, new_storage = tempfile.mkstemp(
            suffix='.spacetime_dict',
            dir=self.storage_path
        )
        os.close(fd)
        
        new_index = {}
        
        # Copy only active entries
        with open(new_storage, 'wb') as new_f:
            for key in self._cold_keys:
                value = self._load_from_cold(key)
                offset = new_f.tell()
                
                data = pickle.dumps((key, value))
                size = len(data)
                
                new_f.write(size.to_bytes(4, 'little'))
                new_f.write(data)
                
                new_index[key] = (offset, size + 4)
        
        # Replace old storage
        os.unlink(self._cold_storage)
        self._cold_storage = new_storage
        self._cold_index = new_index
    
    def __del__(self):
        """Clean up temporary files."""
        if self._cold_storage and os.path.exists(self._cold_storage):
            try:
                os.unlink(self._cold_storage)
            except:
                pass