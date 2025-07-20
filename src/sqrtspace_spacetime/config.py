"""
Configuration management for SpaceTime operations.
"""

import os
import math
import tempfile
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import psutil


class ChunkStrategy(Enum):
    """Strategy for determining chunk sizes."""
    SQRT_N = "sqrt_n"
    MEMORY_BASED = "memory_based"
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


class CompressionType(Enum):
    """Compression algorithms for external storage."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    SNAPPY = "snappy"


@dataclass
class MemoryHierarchy:
    """Memory hierarchy information."""
    l1_cache: int = field(default_factory=lambda: 32 * 1024)  # 32KB
    l2_cache: int = field(default_factory=lambda: 256 * 1024)  # 256KB
    l3_cache: int = field(default_factory=lambda: 8 * 1024 * 1024)  # 8MB
    ram: int = field(default_factory=lambda: psutil.virtual_memory().total)
    disk: int = field(default_factory=lambda: psutil.disk_usage('/').total)
    
    def get_optimal_buffer_size(self, total_size: int) -> int:
        """Calculate optimal buffer size based on memory hierarchy."""
        sqrt_n = int(math.sqrt(total_size))
        
        # Try to fit in L3 cache
        if sqrt_n <= self.l3_cache:
            return sqrt_n
        
        # Otherwise use a fraction of available RAM
        available_ram = psutil.virtual_memory().available
        return min(sqrt_n, int(available_ram * 0.1))


@dataclass
class SpaceTimeConfig:
    """Global configuration for SpaceTime operations."""
    
    # Memory limits
    memory_limit: int = field(default_factory=lambda: int(psutil.virtual_memory().total * 0.8))
    memory_threshold: float = 0.8  # Trigger spillover at 80% usage
    
    # Storage
    external_storage_path: str = field(default_factory=lambda: os.path.join(tempfile.gettempdir(), "spacetime"))
    compression: CompressionType = CompressionType.GZIP
    compression_level: int = 6
    
    # Chunking
    chunk_strategy: ChunkStrategy = ChunkStrategy.SQRT_N
    fixed_chunk_size: int = 10000
    min_chunk_size: int = 100
    max_chunk_size: int = 10_000_000
    
    # Checkpointing
    enable_checkpointing: bool = True
    checkpoint_interval: int = 60  # seconds
    checkpoint_storage: str = "file"  # "file", "redis", "s3"
    
    # Performance
    enable_profiling: bool = False
    parallel_workers: int = field(default_factory=lambda: min(4, os.cpu_count() or 1))
    prefetch_size: int = 2  # Number of chunks to prefetch
    
    # Memory hierarchy
    hierarchy: MemoryHierarchy = field(default_factory=MemoryHierarchy)
    
    _instance: Optional['SpaceTimeConfig'] = None
    
    def __post_init__(self):
        """Initialize storage directory."""
        os.makedirs(self.external_storage_path, exist_ok=True)
    
    @classmethod
    def get_instance(cls) -> 'SpaceTimeConfig':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def set_defaults(cls, **kwargs) -> None:
        """Set default configuration values."""
        instance = cls.get_instance()
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
    
    def calculate_chunk_size(self, total_size: int) -> int:
        """Calculate optimal chunk size based on strategy."""
        if self.chunk_strategy == ChunkStrategy.FIXED:
            return self.fixed_chunk_size
        
        elif self.chunk_strategy == ChunkStrategy.SQRT_N:
            sqrt_n = int(math.sqrt(total_size))
            return max(self.min_chunk_size, min(sqrt_n, self.max_chunk_size))
        
        elif self.chunk_strategy == ChunkStrategy.MEMORY_BASED:
            available = psutil.virtual_memory().available
            # Use 10% of available memory for chunks
            chunk_size = int(available * 0.1 / 8)  # Assume 8 bytes per item
            return max(self.min_chunk_size, min(chunk_size, self.max_chunk_size))
        
        elif self.chunk_strategy == ChunkStrategy.ADAPTIVE:
            # Start with sqrt(n) and adjust based on memory pressure
            base_size = int(math.sqrt(total_size))
            memory_percent = psutil.virtual_memory().percent
            
            if memory_percent > 90:
                # Very high pressure: use minimum size
                return self.min_chunk_size
            elif memory_percent > 70:
                # High pressure: reduce chunk size
                return max(self.min_chunk_size, base_size // 2)
            elif memory_percent < 30:
                # Low pressure: increase chunk size
                return min(self.max_chunk_size, base_size * 2)
            else:
                # Normal pressure: use sqrt(n)
                return max(self.min_chunk_size, min(base_size, self.max_chunk_size))
        
        return self.fixed_chunk_size
    
    def get_compression_module(self):
        """Get compression module based on configuration."""
        if self.compression == CompressionType.GZIP:
            import gzip
            return gzip
        elif self.compression == CompressionType.LZ4:
            try:
                import lz4.frame
                return lz4.frame
            except ImportError:
                import gzip
                return gzip
        elif self.compression == CompressionType.ZSTD:
            try:
                import zstandard
                return zstandard
            except ImportError:
                import gzip
                return gzip
        elif self.compression == CompressionType.SNAPPY:
            try:
                import snappy
                return snappy
            except ImportError:
                import gzip
                return gzip
        else:
            return None
    
    def format_bytes(self, bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024.0:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.2f} PB"
    
    def get_williams_bound(self, time_complexity: int) -> int:
        """Calculate Williams' space bound: SPACE[√(t log t)]."""
        if time_complexity <= 0:
            return 1
        return int(math.sqrt(time_complexity * math.log2(max(2, time_complexity))))


# Global configuration instance
config = SpaceTimeConfig.get_instance()