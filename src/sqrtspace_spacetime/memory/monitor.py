"""Memory monitoring and pressure detection."""

import gc
import time
import psutil
import threading
from enum import Enum
from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from sqrtspace_spacetime.config import config


class MemoryPressureLevel(Enum):
    """Memory pressure levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    
    def __gt__(self, other):
        if not isinstance(other, MemoryPressureLevel):
            return NotImplemented
        return self.value > other.value
    
    def __ge__(self, other):
        if not isinstance(other, MemoryPressureLevel):
            return NotImplemented
        return self.value >= other.value


@dataclass
class MemoryInfo:
    """Memory usage information."""
    total: int
    available: int
    used: int
    percent: float
    pressure_level: MemoryPressureLevel
    timestamp: float
    
    @property
    def used_gb(self) -> float:
        return self.used / (1024 ** 3)
    
    @property
    def available_gb(self) -> float:
        return self.available / (1024 ** 3)
    
    def __str__(self) -> str:
        return (f"Memory: {self.percent:.1f}% used "
                f"({self.used_gb:.2f}/{self.available_gb:.2f} GB), "
                f"Pressure: {self.pressure_level.name}")


class MemoryPressureHandler(ABC):
    """Abstract base class for memory pressure handlers."""
    
    @abstractmethod
    def can_handle(self, level: MemoryPressureLevel, info: MemoryInfo) -> bool:
        """Check if this handler should handle the given pressure level."""
        pass
    
    @abstractmethod
    def handle(self, level: MemoryPressureLevel, info: MemoryInfo) -> None:
        """Handle memory pressure."""
        pass


class MemoryMonitor:
    """Monitor system memory and detect pressure."""
    
    def __init__(self, 
                 check_interval: float = 1.0,
                 memory_limit: Optional[int] = None):
        """
        Initialize memory monitor.
        
        Args:
            check_interval: Seconds between checks
            memory_limit: Custom memory limit in bytes (None for system limit)
        """
        self.check_interval = check_interval
        self.memory_limit = memory_limit or config.memory_limit
        self.handlers: List[MemoryPressureHandler] = []
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._last_check = 0.0
        self._history: List[MemoryInfo] = []
        self._max_history = 100
        
    def add_handler(self, handler: MemoryPressureHandler) -> None:
        """Add a memory pressure handler."""
        self.handlers.append(handler)
    
    def remove_handler(self, handler: MemoryPressureHandler) -> None:
        """Remove a memory pressure handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)
    
    def get_memory_info(self) -> MemoryInfo:
        """Get current memory information."""
        mem = psutil.virtual_memory()
        
        # Use configured limit if lower than system memory
        total = min(mem.total, self.memory_limit)
        used = mem.used
        available = total - used
        percent = (used / total) * 100
        
        # Determine pressure level
        if percent >= 95:
            level = MemoryPressureLevel.CRITICAL
        elif percent >= 85:
            level = MemoryPressureLevel.HIGH
        elif percent >= 70:
            level = MemoryPressureLevel.MEDIUM
        elif percent >= 50:
            level = MemoryPressureLevel.LOW
        else:
            level = MemoryPressureLevel.NONE
        
        return MemoryInfo(
            total=total,
            available=available,
            used=used,
            percent=percent,
            pressure_level=level,
            timestamp=time.time()
        )
    
    def check_memory_pressure(self) -> MemoryPressureLevel:
        """Check current memory pressure and notify handlers."""
        info = self.get_memory_info()
        
        # Add to history
        self._history.append(info)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        
        # Notify handlers
        for handler in self.handlers:
            if handler.can_handle(info.pressure_level, info):
                try:
                    handler.handle(info.pressure_level, info)
                except Exception as e:
                    # Log but don't crash on handler errors
                    print(f"Handler error: {e}")
        
        self._last_check = time.time()
        return info.pressure_level
    
    def should_check(self) -> bool:
        """Check if enough time has passed for next check."""
        return time.time() - self._last_check >= self.check_interval
    
    def start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            try:
                self.check_memory_pressure()
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def get_memory_trend(self, seconds: int = 60) -> Dict[str, float]:
        """Get memory usage trend over past N seconds."""
        if not self._history:
            return {"avg_percent": 0, "max_percent": 0, "trend": 0}
        
        cutoff = time.time() - seconds
        recent = [h for h in self._history if h.timestamp >= cutoff]
        
        if not recent:
            return {"avg_percent": 0, "max_percent": 0, "trend": 0}
        
        percents = [h.percent for h in recent]
        avg_percent = sum(percents) / len(percents)
        max_percent = max(percents)
        
        # Calculate trend (positive = increasing usage)
        if len(recent) >= 2:
            first_half = percents[:len(percents)//2]
            second_half = percents[len(percents)//2:]
            trend = sum(second_half)/len(second_half) - sum(first_half)/len(first_half)
        else:
            trend = 0
        
        return {
            "avg_percent": avg_percent,
            "max_percent": max_percent,
            "trend": trend
        }
    
    def force_gc(self) -> int:
        """Force garbage collection and return bytes freed."""
        before = self.get_memory_info().used
        gc.collect()
        after = self.get_memory_info().used
        return max(0, before - after)
    
    def wait_for_memory(self, required_bytes: int, timeout: float = 30) -> bool:
        """
        Wait for required memory to become available.
        
        Returns:
            True if memory became available, False if timeout
        """
        start = time.time()
        
        while time.time() - start < timeout:
            info = self.get_memory_info()
            if info.available >= required_bytes:
                return True
            
            # Try to free memory
            self.force_gc()
            
            # Let handlers do their work
            self.check_memory_pressure()
            
            time.sleep(0.5)
        
        return False


# Global monitor instance
monitor = MemoryMonitor()