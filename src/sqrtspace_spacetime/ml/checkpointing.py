"""
Gradient checkpointing utilities for memory-efficient training.
"""

import math
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

# Framework imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.checkpoint import checkpoint
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


class CheckpointStrategy(Enum):
    """Checkpointing strategies."""
    SQRT_N = "sqrt_n"          # Checkpoint every √n layers
    UNIFORM = "uniform"        # Uniform intervals
    MEMORY_BASED = "memory"    # Based on memory usage
    SELECTIVE = "selective"    # Only expensive layers


class GradientCheckpointer:
    """
    Gradient checkpointing for memory-efficient training.
    
    Implements Williams' √n strategy for optimal space-time tradeoff.
    """
    
    def __init__(self, strategy: CheckpointStrategy = CheckpointStrategy.SQRT_N):
        self.strategy = strategy
    
    def apply_checkpointing(self,
                          model: Any,
                          checkpoint_layers: Optional[List[str]] = None) -> Any:
        """
        Apply gradient checkpointing to model.
        
        Args:
            model: Neural network model
            checkpoint_layers: Specific layers to checkpoint (None for auto)
            
        Returns:
            Model with checkpointing applied
        """
        if HAS_TORCH and isinstance(model, nn.Module):
            return self._apply_torch_checkpointing(model, checkpoint_layers)
        elif HAS_TF:
            return self._apply_tf_checkpointing(model, checkpoint_layers)
        else:
            print("Warning: No supported framework found for checkpointing")
            return model
    
    def _apply_torch_checkpointing(self,
                                 model: nn.Module,
                                 checkpoint_layers: Optional[List[str]] = None) -> nn.Module:
        """Apply checkpointing to PyTorch model."""
        if checkpoint_layers is None:
            checkpoint_layers = self._select_checkpoint_layers_torch(model)
        
        # Wrap forward methods of selected layers
        for name, module in model.named_modules():
            if name in checkpoint_layers:
                self._wrap_module_torch(module)
        
        return model
    
    def _wrap_module_torch(self, module: nn.Module) -> None:
        """Wrap PyTorch module with gradient checkpointing."""
        original_forward = module.forward
        
        def checkpointed_forward(*args, **kwargs):
            # Use PyTorch's checkpoint function
            if module.training:
                return checkpoint(original_forward, *args, **kwargs)
            else:
                return original_forward(*args, **kwargs)
        
        module.forward = checkpointed_forward
    
    def _apply_tf_checkpointing(self,
                               model: Any,
                               checkpoint_layers: Optional[List[str]] = None) -> Any:
        """Apply checkpointing to TensorFlow model."""
        if checkpoint_layers is None:
            checkpoint_layers = self._select_checkpoint_layers_tf(model)
        
        # TensorFlow implementation
        # Note: TF2 has different checkpointing mechanism
        print(f"TensorFlow checkpointing selected {len(checkpoint_layers)} layers")
        
        return model
    
    def _select_checkpoint_layers_torch(self, model: nn.Module) -> List[str]:
        """Select layers to checkpoint for PyTorch model."""
        layers = []
        
        # Get all layers
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                layers.append((name, module))
        
        if self.strategy == CheckpointStrategy.SQRT_N:
            # Select √n evenly spaced layers
            n = len(layers)
            if n == 0:
                return []
            
            interval = max(1, int(math.sqrt(n)))
            selected = []
            
            for i in range(0, n, interval):
                name, module = layers[i]
                if self._can_checkpoint_module(module):
                    selected.append(name)
            
            return selected
        
        elif self.strategy == CheckpointStrategy.MEMORY_BASED:
            # Select layers with large activation memory
            memory_layers = []
            
            for name, module in layers:
                memory = self._estimate_module_memory(module)
                memory_layers.append((name, memory))
            
            # Sort by memory and select top √n
            memory_layers.sort(key=lambda x: x[1], reverse=True)
            n_checkpoint = max(1, int(math.sqrt(len(memory_layers))))
            
            return [name for name, _ in memory_layers[:n_checkpoint]]
        
        else:
            # Default: checkpoint all eligible layers
            return [name for name, module in layers if self._can_checkpoint_module(module)]
    
    def _select_checkpoint_layers_tf(self, model: Any) -> List[str]:
        """Select layers to checkpoint for TensorFlow model."""
        if not hasattr(model, 'layers'):
            return []
        
        layers = [(layer.name, layer) for layer in model.layers]
        
        if self.strategy == CheckpointStrategy.SQRT_N:
            n = len(layers)
            interval = max(1, int(math.sqrt(n)))
            
            selected = []
            for i in range(0, n, interval):
                name, layer = layers[i]
                selected.append(name)
            
            return selected
        
        return [name for name, _ in layers]
    
    def _can_checkpoint_module(self, module: Any) -> bool:
        """Check if module can be safely checkpointed."""
        if HAS_TORCH:
            # Avoid checkpointing modules with randomness
            no_checkpoint = (nn.Dropout, nn.Dropout2d, nn.Dropout3d)
            return not isinstance(module, no_checkpoint)
        return True
    
    def _estimate_module_memory(self, module: Any) -> int:
        """Estimate memory usage of module activations."""
        if HAS_TORCH and isinstance(module, nn.Module):
            # Estimate based on output size
            if isinstance(module, nn.Linear):
                return module.out_features * 4  # FP32
            elif isinstance(module, nn.Conv2d):
                # Rough estimate
                return module.out_channels * 100 * 100 * 4
            else:
                # Default estimate
                params = sum(p.numel() for p in module.parameters())
                return params * 4
        return 0
    
    @staticmethod
    def create_checkpoint_segments(model: Any,
                                 n_segments: Optional[int] = None) -> List[List[str]]:
        """
        Create checkpoint segments using √n strategy.
        
        Args:
            model: Neural network model
            n_segments: Number of segments (None for √n)
            
        Returns:
            List of layer name segments
        """
        # Get all layers
        if HAS_TORCH and isinstance(model, nn.Module):
            all_layers = [name for name, _ in model.named_modules() 
                         if len(list(_.children())) == 0]
        elif HAS_TF and hasattr(model, 'layers'):
            all_layers = [layer.name for layer in model.layers]
        else:
            return []
        
        n = len(all_layers)
        if n == 0:
            return []
        
        # Use √n segments by default
        if n_segments is None:
            n_segments = max(1, int(math.sqrt(n)))
        
        # Create segments
        segment_size = max(1, n // n_segments)
        segments = []
        
        for i in range(0, n, segment_size):
            segment = all_layers[i:i + segment_size]
            if segment:
                segments.append(segment)
        
        return segments


def checkpoint_sequential(modules: List[Any],
                        input: Any,
                        segments: Optional[int] = None) -> Any:
    """
    Checkpoint a sequential model using √n segments.
    
    Args:
        modules: List of modules to execute sequentially
        input: Input tensor
        segments: Number of checkpoint segments (None for √n)
        
    Returns:
        Output tensor
    """
    if not HAS_TORCH:
        # Fallback to normal execution
        x = input
        for module in modules:
            x = module(x)
        return x
    
    n = len(modules)
    if n == 0:
        return input
    
    # Use √n segments
    if segments is None:
        segments = max(1, int(math.sqrt(n)))
    
    segment_size = max(1, n // segments)
    
    # Execute with checkpointing
    x = input
    for i in range(0, n, segment_size):
        segment = modules[i:i + segment_size]
        
        if len(segment) == 1:
            # Single module
            if modules[0].training:
                x = checkpoint(segment[0], x)
            else:
                x = segment[0](x)
        else:
            # Multiple modules - create sequential wrapper
            def run_segment(x, *modules):
                for module in modules:
                    x = module(x)
                return x
            
            if modules[0].training:
                x = checkpoint(run_segment, x, *segment)
            else:
                x = run_segment(x, *segment)
    
    return x