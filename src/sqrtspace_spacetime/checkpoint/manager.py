"""
Checkpoint manager for saving and restoring computation state.
"""

import time
import uuid
import pickle
import zlib
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

import psutil

from sqrtspace_spacetime.config import config
from sqrtspace_spacetime.memory import monitor


class CheckpointStrategy(Enum):
    """Checkpointing strategies."""
    SQRT_N = "sqrt_n"                    # Checkpoint every √n iterations  
    MEMORY_PRESSURE = "memory_pressure"   # Checkpoint when memory exceeds threshold
    TIME_BASED = "time_based"            # Checkpoint every k seconds
    ADAPTIVE = "adaptive"                # Dynamically adjust based on performance


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""
    strategy: CheckpointStrategy = CheckpointStrategy.SQRT_N
    checkpoint_dir: str = ".checkpoints"
    compression: bool = True
    compression_level: int = 6
    memory_threshold: float = 0.8  # Fraction of available memory
    time_interval: float = 60.0    # Seconds between checkpoints
    min_interval: int = 100        # Minimum iterations between checkpoints
    max_checkpoints: int = 10      # Maximum concurrent checkpoints
    enable_recovery: bool = True
    verbose: bool = False


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_id: str
    iteration: int
    timestamp: float
    state_size: int
    compressed_size: int
    compression_ratio: float
    strategy_used: str
    reason: str
    state_vars: List[str]
    performance_impact: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass 
class CheckpointStats:
    """Statistics about checkpointing performance."""
    total_checkpoints: int = 0
    total_time: float = 0.0
    total_size: int = 0
    compressed_size: int = 0
    average_compression: float = 0.0
    memory_saved: int = 0
    overhead_percent: float = 0.0
    recoveries: int = 0
    strategy_distribution: Dict[str, int] = None
    
    def __post_init__(self):
        if self.strategy_distribution is None:
            self.strategy_distribution = {}


class CheckpointManager:
    """
    Manage checkpoints for long-running computations.
    
    Implements Williams' √n checkpoint intervals for optimal space-time tradeoff.
    """
    
    def __init__(self, 
                 checkpoint_id: Optional[str] = None,
                 config: Optional[CheckpointConfig] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_id: Unique ID for this computation
            config: Checkpoint configuration
        """
        self.checkpoint_id = checkpoint_id or str(uuid.uuid4())
        self.config = config or CheckpointConfig()
        self.stats = CheckpointStats()
        
        # Create checkpoint directory
        self.checkpoint_path = Path(self.config.checkpoint_dir) / self.checkpoint_id
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self._iteration = 0
        self._last_checkpoint_iter = 0
        self._last_checkpoint_time = time.time()
        self._checkpoint_interval = None
        self._total_iterations = None
        
    def should_checkpoint(self, iteration: Optional[int] = None) -> bool:
        """
        Determine if checkpoint is needed.
        
        Args:
            iteration: Current iteration (None to use internal counter)
            
        Returns:
            True if checkpoint should be created
        """
        if iteration is not None:
            self._iteration = iteration
        else:
            self._iteration += 1
        
        # Check strategy
        if self.config.strategy == CheckpointStrategy.SQRT_N:
            return self._should_checkpoint_sqrt_n()
        elif self.config.strategy == CheckpointStrategy.MEMORY_PRESSURE:
            return self._should_checkpoint_memory()
        elif self.config.strategy == CheckpointStrategy.TIME_BASED:
            return self._should_checkpoint_time()
        elif self.config.strategy == CheckpointStrategy.ADAPTIVE:
            return self._should_checkpoint_adaptive()
        
        return False
    
    def _should_checkpoint_sqrt_n(self) -> bool:
        """Check if checkpoint needed using √n strategy."""
        if self._checkpoint_interval is None:
            # Estimate interval if total iterations unknown
            if self._total_iterations:
                self._checkpoint_interval = max(
                    self.config.min_interval,
                    int(self._total_iterations ** 0.5)
                )
            else:
                # Use adaptive estimation
                self._checkpoint_interval = self.config.min_interval
        
        iterations_since = self._iteration - self._last_checkpoint_iter
        return iterations_since >= self._checkpoint_interval
    
    def _should_checkpoint_memory(self) -> bool:
        """Check if checkpoint needed due to memory pressure."""
        mem_info = monitor.get_memory_info()
        return mem_info.percent > self.config.memory_threshold * 100
    
    def _should_checkpoint_time(self) -> bool:
        """Check if checkpoint needed based on time."""
        elapsed = time.time() - self._last_checkpoint_time
        return elapsed >= self.config.time_interval
    
    def _should_checkpoint_adaptive(self) -> bool:
        """Adaptive checkpointing based on multiple factors."""
        # Combine strategies
        sqrt_n = self._should_checkpoint_sqrt_n()
        memory = self._should_checkpoint_memory()
        time_based = self._should_checkpoint_time()
        
        # Checkpoint if any condition is met
        return sqrt_n or memory or time_based
    
    def save(self, state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save checkpoint.
        
        Args:
            state: State dictionary to save
            metadata: Additional metadata
            
        Returns:
            Checkpoint ID
        """
        start_time = time.time()
        
        # Generate checkpoint ID
        checkpoint_file = self.checkpoint_path / f"checkpoint_{self._iteration}.pkl"
        
        # Prepare state
        state_bytes = pickle.dumps(state)
        original_size = len(state_bytes)
        
        # Compress if enabled
        if self.config.compression:
            state_bytes = zlib.compress(state_bytes, self.config.compression_level)
            compressed_size = len(state_bytes)
            compression_ratio = original_size / compressed_size
        else:
            compressed_size = original_size
            compression_ratio = 1.0
        
        # Save checkpoint
        with open(checkpoint_file, 'wb') as f:
            f.write(state_bytes)
        
        # Save metadata
        checkpoint_metadata = CheckpointMetadata(
            checkpoint_id=str(checkpoint_file),
            iteration=self._iteration,
            timestamp=time.time(),
            state_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            strategy_used=self.config.strategy.value,
            reason=self._get_checkpoint_reason(),
            state_vars=list(state.keys()),
            performance_impact={
                'save_time': time.time() - start_time,
                'compression_time': 0.0  # TODO: measure separately
            }
        )
        
        metadata_file = checkpoint_file.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(checkpoint_metadata.to_dict(), f, indent=2)
        
        # Update stats
        self._update_stats(checkpoint_metadata)
        
        # Update tracking
        self._last_checkpoint_iter = self._iteration
        self._last_checkpoint_time = time.time()
        
        # Clean old checkpoints
        self._cleanup_old_checkpoints()
        
        if self.config.verbose:
            print(f"Checkpoint saved: iteration {self._iteration}, "
                  f"size {compressed_size / 1024:.1f}KB, "
                  f"compression {compression_ratio:.1f}x")
        
        return str(checkpoint_file)
    
    def load(self, checkpoint_id: Optional[str] = None) -> Tuple[Dict[str, Any], CheckpointMetadata]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_id: Specific checkpoint to load (None for latest)
            
        Returns:
            Tuple of (state, metadata)
        """
        if checkpoint_id:
            checkpoint_file = Path(checkpoint_id)
        else:
            # Find latest checkpoint
            checkpoints = list(self.checkpoint_path.glob("checkpoint_*.pkl"))
            if not checkpoints:
                raise ValueError("No checkpoints found")
            
            checkpoint_file = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        # Load metadata
        metadata_file = checkpoint_file.with_suffix('.json')
        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
            metadata = CheckpointMetadata(**metadata_dict)
        
        # Load state
        with open(checkpoint_file, 'rb') as f:
            state_bytes = f.read()
        
        # Decompress if needed
        if self.config.compression:
            state_bytes = zlib.decompress(state_bytes)
        
        state = pickle.loads(state_bytes)
        
        # Update stats
        self.stats.recoveries += 1
        
        if self.config.verbose:
            print(f"Checkpoint loaded: iteration {metadata.iteration}")
        
        return state, metadata
    
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints."""
        metadata_files = self.checkpoint_path.glob("checkpoint_*.json")
        checkpoints = []
        
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
                checkpoints.append(CheckpointMetadata(**metadata_dict))
        
        return sorted(checkpoints, key=lambda c: c.iteration)
    
    def delete_checkpoint(self, checkpoint_id: str) -> None:
        """Delete specific checkpoint."""
        checkpoint_file = Path(checkpoint_id)
        metadata_file = checkpoint_file.with_suffix('.json')
        
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        if metadata_file.exists():
            metadata_file.unlink()
    
    def cleanup(self) -> None:
        """Clean up all checkpoints."""
        import shutil
        if self.checkpoint_path.exists():
            shutil.rmtree(self.checkpoint_path)
    
    def set_total_iterations(self, total: int) -> None:
        """
        Set total iterations for optimal √n calculation.
        
        Args:
            total: Total number of iterations
        """
        self._total_iterations = total
        self._checkpoint_interval = max(
            self.config.min_interval,
            int(total ** 0.5)
        )
        
        if self.config.verbose:
            print(f"Checkpoint interval set to {self._checkpoint_interval} "
                  f"(√{total} strategy)")
    
    def get_stats(self) -> CheckpointStats:
        """Get checkpoint statistics."""
        if self.stats.total_checkpoints > 0:
            self.stats.average_compression = (
                self.stats.total_size / self.stats.compressed_size
            )
            self.stats.overhead_percent = (
                self.stats.total_time / (time.time() - self._last_checkpoint_time) * 100
            )
        
        return self.stats
    
    def _get_checkpoint_reason(self) -> str:
        """Get reason for checkpoint."""
        if self.config.strategy == CheckpointStrategy.SQRT_N:
            return f"√n interval reached ({self._checkpoint_interval} iterations)"
        elif self.config.strategy == CheckpointStrategy.MEMORY_PRESSURE:
            mem_info = monitor.get_memory_info()
            return f"Memory pressure: {mem_info.percent:.1f}%"
        elif self.config.strategy == CheckpointStrategy.TIME_BASED:
            return f"Time interval: {self.config.time_interval}s"
        else:
            return "Adaptive strategy triggered"
    
    def _update_stats(self, metadata: CheckpointMetadata) -> None:
        """Update statistics."""
        self.stats.total_checkpoints += 1
        self.stats.total_time += metadata.performance_impact['save_time']
        self.stats.total_size += metadata.state_size
        self.stats.compressed_size += metadata.compressed_size
        
        # Update strategy distribution
        strategy = metadata.strategy_used
        self.stats.strategy_distribution[strategy] = (
            self.stats.strategy_distribution.get(strategy, 0) + 1
        )
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to stay under limit."""
        checkpoints = list(self.checkpoint_path.glob("checkpoint_*.pkl"))
        
        if len(checkpoints) > self.config.max_checkpoints:
            # Sort by modification time
            checkpoints.sort(key=lambda p: p.stat().st_mtime)
            
            # Remove oldest
            for checkpoint in checkpoints[:-self.config.max_checkpoints]:
                self.delete_checkpoint(str(checkpoint))
    
    def create_recovery_code(self, func: Callable) -> str:
        """
        Generate recovery code for function.
        
        Args:
            func: Function to generate recovery for
            
        Returns:
            Recovery code as string
        """
        recovery_template = '''
def recover_{func_name}(checkpoint_id=None):
    """Recover {func_name} from checkpoint."""
    manager = CheckpointManager("{checkpoint_id}")
    
    # Load checkpoint
    state, metadata = manager.load(checkpoint_id)
    
    # Resume computation
    iteration = metadata.iteration
    
    # Restore state variables
    {state_restoration}
    
    # Continue from checkpoint
    # TODO: Add continuation logic
    
    return state
'''
        
        # Get function name
        func_name = func.__name__
        
        # Generate state restoration code
        state_vars = []
        if hasattr(func, '_checkpoint_state'):
            state_vars = func._checkpoint_state
        
        state_restoration = '\n    '.join(
            f"{var} = state.get('{var}')" for var in state_vars
        )
        
        return recovery_template.format(
            func_name=func_name,
            checkpoint_id=self.checkpoint_id,
            state_restoration=state_restoration
        )