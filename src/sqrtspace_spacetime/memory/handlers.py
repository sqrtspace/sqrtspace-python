"""Memory pressure handlers."""

import gc
import time
import logging
from typing import Dict, Any, List, Callable, Optional
from weakref import WeakValueDictionary

from sqrtspace_spacetime.memory.monitor import (
    MemoryPressureHandler, 
    MemoryPressureLevel, 
    MemoryInfo
)


class LoggingHandler(MemoryPressureHandler):
    """Log memory pressure events."""
    
    def __init__(self, 
                 logger: Optional[logging.Logger] = None,
                 min_level: MemoryPressureLevel = MemoryPressureLevel.MEDIUM):
        self.logger = logger or logging.getLogger(__name__)
        self.min_level = min_level
        self._last_log = {}
    
    def can_handle(self, level: MemoryPressureLevel, info: MemoryInfo) -> bool:
        return level >= self.min_level
    
    def handle(self, level: MemoryPressureLevel, info: MemoryInfo) -> None:
        # Avoid spamming logs - only log if level changed or 60s passed
        last_time = self._last_log.get(level, 0)
        if time.time() - last_time < 60 and level in self._last_log:
            return
        
        self._last_log[level] = time.time()
        
        if level == MemoryPressureLevel.CRITICAL:
            self.logger.critical(f"CRITICAL memory pressure: {info}")
        elif level == MemoryPressureLevel.HIGH:
            self.logger.error(f"HIGH memory pressure: {info}")
        elif level == MemoryPressureLevel.MEDIUM:
            self.logger.warning(f"MEDIUM memory pressure: {info}")
        else:
            self.logger.info(f"Memory pressure: {info}")


class CacheEvictionHandler(MemoryPressureHandler):
    """Evict cached data under memory pressure."""
    
    def __init__(self):
        self._caches: List[WeakValueDictionary] = []
        self._eviction_rates = {
            MemoryPressureLevel.LOW: 0.1,      # Evict 10%
            MemoryPressureLevel.MEDIUM: 0.25,   # Evict 25%
            MemoryPressureLevel.HIGH: 0.5,      # Evict 50%
            MemoryPressureLevel.CRITICAL: 0.9,  # Evict 90%
        }
    
    def register_cache(self, cache: Dict[Any, Any]) -> None:
        """Register a cache for eviction."""
        self._caches.append(WeakValueDictionary(cache))
    
    def can_handle(self, level: MemoryPressureLevel, info: MemoryInfo) -> bool:
        return level >= MemoryPressureLevel.LOW and self._caches
    
    def handle(self, level: MemoryPressureLevel, info: MemoryInfo) -> None:
        eviction_rate = self._eviction_rates.get(level, 0)
        if eviction_rate == 0:
            return
        
        for cache in self._caches:
            if not cache:
                continue
            
            size = len(cache)
            if size == 0:
                continue
            
            # Evict entries
            to_evict = int(size * eviction_rate)
            keys = list(cache.keys())[:to_evict]
            
            for key in keys:
                cache.pop(key, None)


class GarbageCollectionHandler(MemoryPressureHandler):
    """Trigger garbage collection under memory pressure."""
    
    def __init__(self, min_interval: float = 5.0):
        self.min_interval = min_interval
        self._last_gc = 0
    
    def can_handle(self, level: MemoryPressureLevel, info: MemoryInfo) -> bool:
        return level >= MemoryPressureLevel.MEDIUM
    
    def handle(self, level: MemoryPressureLevel, info: MemoryInfo) -> None:
        now = time.time()
        
        # Don't GC too frequently
        if now - self._last_gc < self.min_interval:
            return
        
        self._last_gc = now
        
        # More aggressive GC for higher pressure
        if level >= MemoryPressureLevel.HIGH:
            # Full collection
            gc.collect(2)
        else:
            # Quick collection
            gc.collect(0)


class ThrottlingHandler(MemoryPressureHandler):
    """Throttle operations under memory pressure."""
    
    def __init__(self):
        self._throttle_rates = {
            MemoryPressureLevel.LOW: 0,        # No throttling
            MemoryPressureLevel.MEDIUM: 0.1,   # 100ms delay
            MemoryPressureLevel.HIGH: 0.5,     # 500ms delay
            MemoryPressureLevel.CRITICAL: 2.0, # 2s delay
        }
        self._callbacks: List[Callable[[float], None]] = []
    
    def register_callback(self, callback: Callable[[float], None]) -> None:
        """Register callback to be notified of throttle rates."""
        self._callbacks.append(callback)
    
    def can_handle(self, level: MemoryPressureLevel, info: MemoryInfo) -> bool:
        return level >= MemoryPressureLevel.MEDIUM
    
    def handle(self, level: MemoryPressureLevel, info: MemoryInfo) -> None:
        delay = self._throttle_rates.get(level, 0)
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(delay)
            except Exception:
                pass


class SpillToDiskHandler(MemoryPressureHandler):
    """Spill data to disk under memory pressure."""
    
    def __init__(self, spill_path: Optional[str] = None):
        self.spill_path = spill_path
        self._spillable_objects: List[Any] = []
    
    def register_spillable(self, obj: Any) -> None:
        """Register an object that can spill to disk."""
        if hasattr(obj, 'spill_to_disk'):
            self._spillable_objects.append(obj)
    
    def can_handle(self, level: MemoryPressureLevel, info: MemoryInfo) -> bool:
        return level >= MemoryPressureLevel.HIGH and self._spillable_objects
    
    def handle(self, level: MemoryPressureLevel, info: MemoryInfo) -> None:
        for obj in self._spillable_objects:
            try:
                if hasattr(obj, 'memory_usage'):
                    # Only spill large objects
                    if obj.memory_usage() > 10 * 1024 * 1024:  # 10MB
                        obj.spill_to_disk(self.spill_path)
            except Exception:
                pass