"""
ML Training Memory Optimizer: Optimize neural network training memory usage.

Features:
- Layer-by-layer memory profiling
- Automatic gradient checkpointing with √n intervals
- Mixed precision configuration
- Batch size optimization
- Framework-agnostic (PyTorch/TensorFlow)
"""

import math
import psutil
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from sqrtspace_spacetime.config import config
from sqrtspace_spacetime.memory import monitor

# Try to import ML frameworks
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


class MemoryOptimizationStrategy(Enum):
    """Memory optimization strategies for ML training."""
    GRADIENT_CHECKPOINTING = "gradient_checkpointing"  # Recompute activations
    MIXED_PRECISION = "mixed_precision"                # FP16/BF16 training
    GRADIENT_ACCUMULATION = "gradient_accumulation"    # Smaller effective batch
    MODEL_SHARDING = "model_sharding"                  # Distribute layers
    ACTIVATION_COMPRESSION = "activation_compression"   # Compress intermediate
    DYNAMIC_BATCH_SIZE = "dynamic_batch_size"         # Adjust on the fly


@dataclass
class LayerProfile:
    """Profile of a neural network layer."""
    name: str
    layer_type: str
    parameters: int
    activation_size: int  # Per sample
    gradient_size: int    # Per sample  
    computation_time: float
    memory_bytes: int
    can_checkpoint: bool
    precision: str  # 'fp32', 'fp16', 'int8'


@dataclass
class ModelProfile:
    """Complete model memory profile."""
    total_parameters: int
    total_activations: int  # Per sample
    peak_memory: int
    layers: List[LayerProfile]
    memory_timeline: List[Tuple[str, int]]  # (operation, memory)
    bottleneck_layers: List[str]
    framework: str  # 'pytorch', 'tensorflow', 'generic'


@dataclass
class OptimizationPlan:
    """Optimization plan for model training."""
    strategies: List[MemoryOptimizationStrategy]
    checkpoint_layers: List[str]
    batch_size: int
    gradient_accumulation_steps: int
    mixed_precision_config: Dict[str, Any]
    estimated_memory: int
    estimated_speedup: float
    memory_savings: int
    explanation: str


@dataclass
class TrainingConfig:
    """Configuration for optimized training."""
    original_batch_size: int
    optimized_batch_size: int
    accumulation_steps: int
    checkpoint_segments: List[List[str]]
    precision_map: Dict[str, str]
    memory_limit: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MLMemoryOptimizer:
    """Optimize memory usage for ML model training."""
    
    def __init__(self, memory_limit: Optional[int] = None):
        """
        Initialize optimizer.
        
        Args:
            memory_limit: Memory limit in bytes (None for auto-detect)
        """
        self.memory_limit = memory_limit or int(psutil.virtual_memory().available * 0.8)
    
    def analyze_model(self, 
                     model: Any,
                     input_shape: Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]],
                     batch_size: int = 1) -> ModelProfile:
        """
        Analyze model memory requirements.
        
        Args:
            model: Neural network model
            input_shape: Input shape(s) 
            batch_size: Batch size for analysis
            
        Returns:
            ModelProfile with memory analysis
        """
        if HAS_TORCH and isinstance(model, nn.Module):
            return self._analyze_torch_model(model, input_shape, batch_size)
        elif HAS_TF and hasattr(model, 'layers'):
            return self._analyze_tf_model(model, input_shape, batch_size)
        else:
            return self._analyze_generic_model(model, input_shape, batch_size)
    
    def _analyze_torch_model(self, 
                           model: nn.Module,
                           input_shape: Tuple[int, ...],
                           batch_size: int) -> ModelProfile:
        """Analyze PyTorch model."""
        layers = []
        total_params = 0
        total_activations = 0
        memory_timeline = []
        
        # Count parameters
        for name, param in model.named_parameters():
            total_params += param.numel()
        
        # Analyze layers
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                layer_params = sum(p.numel() for p in module.parameters())
                
                # Estimate activation size (simplified)
                if isinstance(module, nn.Linear):
                    activation_size = module.out_features * batch_size * 4  # fp32
                elif isinstance(module, nn.Conv2d):
                    # Rough estimate
                    activation_size = module.out_channels * 100 * 100 * batch_size * 4
                else:
                    activation_size = layer_params * batch_size * 4
                
                total_activations += activation_size
                
                layers.append(LayerProfile(
                    name=name,
                    layer_type=module.__class__.__name__,
                    parameters=layer_params,
                    activation_size=activation_size // batch_size,
                    gradient_size=layer_params * 4,  # fp32 gradients
                    computation_time=0.001,  # Placeholder
                    memory_bytes=layer_params * 4 + activation_size,
                    can_checkpoint=self._can_checkpoint_layer(module),
                    precision='fp32'
                ))
        
        # Find bottlenecks (top 20% by memory)
        sorted_layers = sorted(layers, key=lambda l: l.memory_bytes, reverse=True)
        bottleneck_count = max(1, len(layers) // 5)
        bottleneck_layers = [l.name for l in sorted_layers[:bottleneck_count]]
        
        return ModelProfile(
            total_parameters=total_params,
            total_activations=total_activations // batch_size,
            peak_memory=total_params * 4 + total_activations,
            layers=layers,
            memory_timeline=memory_timeline,
            bottleneck_layers=bottleneck_layers,
            framework='pytorch'
        )
    
    def _analyze_tf_model(self,
                         model: Any,
                         input_shape: Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]],
                         batch_size: int) -> ModelProfile:
        """Analyze TensorFlow model."""
        layers = []
        total_params = model.count_params()
        total_activations = 0
        
        # Analyze each layer
        for layer in model.layers:
            layer_params = layer.count_params()
            
            # Estimate activation size
            if hasattr(layer, 'output_shape'):
                shape = layer.output_shape
                if isinstance(shape, tuple):
                    activation_size = np.prod(shape[1:]) * batch_size * 4
                else:
                    activation_size = layer_params * batch_size * 4
            else:
                activation_size = layer_params * batch_size * 4
            
            total_activations += activation_size
            
            layers.append(LayerProfile(
                name=layer.name,
                layer_type=layer.__class__.__name__,
                parameters=layer_params,
                activation_size=activation_size // batch_size,
                gradient_size=layer_params * 4,
                computation_time=0.001,
                memory_bytes=layer_params * 4 + activation_size,
                can_checkpoint=True,  # Most TF layers can checkpoint
                precision='fp32'
            ))
        
        # Find bottlenecks
        sorted_layers = sorted(layers, key=lambda l: l.memory_bytes, reverse=True)
        bottleneck_count = max(1, len(layers) // 5)
        bottleneck_layers = [l.name for l in sorted_layers[:bottleneck_count]]
        
        return ModelProfile(
            total_parameters=total_params,
            total_activations=total_activations // batch_size,
            peak_memory=total_params * 4 + total_activations,
            layers=layers,
            memory_timeline=[],
            bottleneck_layers=bottleneck_layers,
            framework='tensorflow'
        )
    
    def _analyze_generic_model(self,
                             model: Any,
                             input_shape: Tuple[int, ...],
                             batch_size: int) -> ModelProfile:
        """Analyze generic model."""
        # Basic heuristics
        estimated_params = 10_000_000  # 10M parameters
        estimated_activations = estimated_params * batch_size
        
        return ModelProfile(
            total_parameters=estimated_params,
            total_activations=estimated_activations,
            peak_memory=estimated_params * 4 + estimated_activations * 4,
            layers=[],
            memory_timeline=[],
            bottleneck_layers=[],
            framework='generic'
        )
    
    def optimize(self,
                model_profile: ModelProfile,
                target_batch_size: int,
                strategies: Optional[List[MemoryOptimizationStrategy]] = None) -> OptimizationPlan:
        """
        Generate optimization plan for model.
        
        Args:
            model_profile: Model profile from analyze_model
            target_batch_size: Desired batch size
            strategies: Strategies to consider (None for auto)
            
        Returns:
            OptimizationPlan with recommendations
        """
        if strategies is None:
            strategies = self._select_strategies(model_profile, target_batch_size)
        
        # Calculate memory requirements
        base_memory = model_profile.total_parameters * 4  # Parameters
        activation_memory = model_profile.total_activations * target_batch_size * 4
        gradient_memory = model_profile.total_parameters * 4  # Gradients
        optimizer_memory = model_profile.total_parameters * 8  # Adam states
        
        total_memory = base_memory + activation_memory + gradient_memory + optimizer_memory
        
        # Initialize plan
        plan = OptimizationPlan(
            strategies=strategies,
            checkpoint_layers=[],
            batch_size=target_batch_size,
            gradient_accumulation_steps=1,
            mixed_precision_config={},
            estimated_memory=total_memory,
            estimated_speedup=1.0,
            memory_savings=0,
            explanation=""
        )
        
        # Apply strategies
        for strategy in strategies:
            if strategy == MemoryOptimizationStrategy.GRADIENT_CHECKPOINTING:
                self._apply_checkpointing(plan, model_profile)
            elif strategy == MemoryOptimizationStrategy.MIXED_PRECISION:
                self._apply_mixed_precision(plan, model_profile)
            elif strategy == MemoryOptimizationStrategy.GRADIENT_ACCUMULATION:
                self._apply_gradient_accumulation(plan, model_profile)
        
        # Calculate final estimates
        plan.memory_savings = total_memory - plan.estimated_memory
        plan.explanation = self._generate_explanation(plan, model_profile)
        
        return plan
    
    def _select_strategies(self,
                         model_profile: ModelProfile,
                         target_batch_size: int) -> List[MemoryOptimizationStrategy]:
        """Select appropriate optimization strategies."""
        strategies = []
        
        # Calculate memory pressure
        required_memory = (model_profile.total_parameters * 4 + 
                          model_profile.total_activations * target_batch_size * 4)
        
        if required_memory > self.memory_limit:
            # High memory pressure - use all strategies
            strategies.append(MemoryOptimizationStrategy.GRADIENT_CHECKPOINTING)
            strategies.append(MemoryOptimizationStrategy.MIXED_PRECISION)
            strategies.append(MemoryOptimizationStrategy.GRADIENT_ACCUMULATION)
        elif required_memory > self.memory_limit * 0.8:
            # Medium pressure
            strategies.append(MemoryOptimizationStrategy.GRADIENT_CHECKPOINTING)
            strategies.append(MemoryOptimizationStrategy.MIXED_PRECISION)
        elif required_memory > self.memory_limit * 0.6:
            # Low pressure
            strategies.append(MemoryOptimizationStrategy.MIXED_PRECISION)
        
        return strategies
    
    def _apply_checkpointing(self,
                           plan: OptimizationPlan,
                           model_profile: ModelProfile) -> None:
        """Apply gradient checkpointing using √n strategy."""
        n_layers = len(model_profile.layers)
        
        if n_layers == 0:
            return
        
        # Use √n checkpointing intervals
        checkpoint_interval = max(1, int(math.sqrt(n_layers)))
        
        # Select layers to checkpoint
        checkpoint_layers = []
        for i in range(0, n_layers, checkpoint_interval):
            if i < len(model_profile.layers):
                layer = model_profile.layers[i]
                if layer.can_checkpoint:
                    checkpoint_layers.append(layer.name)
        
        plan.checkpoint_layers = checkpoint_layers
        
        # Update memory estimate (save ~50% of activation memory)
        saved_memory = sum(l.activation_size * plan.batch_size * 4 
                          for l in model_profile.layers 
                          if l.name in checkpoint_layers) * 0.5
        
        plan.estimated_memory -= int(saved_memory)
        plan.estimated_speedup *= 0.8  # 20% slowdown from recomputation
    
    def _apply_mixed_precision(self,
                             plan: OptimizationPlan,
                             model_profile: ModelProfile) -> None:
        """Apply mixed precision training."""
        plan.mixed_precision_config = {
            'enabled': True,
            'loss_scale': 'dynamic',
            'compute_dtype': 'float16',
            'variable_dtype': 'float32'
        }
        
        # Update memory estimate (save ~50% on activations)
        activation_savings = model_profile.total_activations * plan.batch_size * 2
        plan.estimated_memory -= activation_savings
        plan.estimated_speedup *= 1.5  # Potential speedup on modern GPUs
    
    def _apply_gradient_accumulation(self,
                                   plan: OptimizationPlan,
                                   model_profile: ModelProfile) -> None:
        """Apply gradient accumulation."""
        # Calculate how many accumulation steps needed
        current_memory = plan.estimated_memory
        
        if current_memory > self.memory_limit:
            # Reduce effective batch size
            reduction_factor = current_memory / self.memory_limit
            accumulation_steps = int(math.ceil(reduction_factor))
            
            # Adjust batch size and accumulation
            effective_batch = plan.batch_size // accumulation_steps
            plan.batch_size = max(1, effective_batch)
            plan.gradient_accumulation_steps = accumulation_steps
            
            # Update memory estimate
            plan.estimated_memory = plan.estimated_memory // accumulation_steps
    
    def _can_checkpoint_layer(self, layer: Any) -> bool:
        """Check if layer can be checkpointed."""
        if HAS_TORCH:
            # Most layers can be checkpointed except those with side effects
            no_checkpoint_types = (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
            return not isinstance(layer, no_checkpoint_types)
        return True
    
    def _generate_explanation(self,
                            plan: OptimizationPlan,
                            model_profile: ModelProfile) -> str:
        """Generate human-readable explanation."""
        explanations = []
        
        explanations.append(f"Model Analysis:")
        explanations.append(f"- Total parameters: {model_profile.total_parameters:,}")
        explanations.append(f"- Peak memory estimate: {plan.estimated_memory / (1024**3):.2f} GB")
        explanations.append(f"- Memory savings: {plan.memory_savings / (1024**3):.2f} GB")
        
        if MemoryOptimizationStrategy.GRADIENT_CHECKPOINTING in plan.strategies:
            explanations.append(f"\nGradient Checkpointing:")
            explanations.append(f"- Checkpointing {len(plan.checkpoint_layers)} layers using √n strategy")
            explanations.append(f"- This trades ~20% compute time for ~50% activation memory")
        
        if MemoryOptimizationStrategy.MIXED_PRECISION in plan.strategies:
            explanations.append(f"\nMixed Precision:")
            explanations.append(f"- Using FP16 for forward pass, FP32 for gradients")
            explanations.append(f"- Reduces activation memory by ~50%")
        
        if plan.gradient_accumulation_steps > 1:
            explanations.append(f"\nGradient Accumulation:")
            explanations.append(f"- Accumulating over {plan.gradient_accumulation_steps} steps")
            explanations.append(f"- Effective batch size: {plan.batch_size * plan.gradient_accumulation_steps}")
        
        return "\n".join(explanations)
    
    def get_training_config(self,
                          plan: OptimizationPlan,
                          model_profile: ModelProfile) -> TrainingConfig:
        """
        Generate training configuration from optimization plan.
        
        Args:
            plan: Optimization plan
            model_profile: Model profile
            
        Returns:
            TrainingConfig ready for use
        """
        # Group checkpoint layers into segments
        checkpoint_segments = []
        if plan.checkpoint_layers:
            # Create √n segments
            n_segments = int(math.sqrt(len(plan.checkpoint_layers)))
            segment_size = max(1, len(plan.checkpoint_layers) // n_segments)
            
            for i in range(0, len(plan.checkpoint_layers), segment_size):
                segment = plan.checkpoint_layers[i:i + segment_size]
                if segment:
                    checkpoint_segments.append(segment)
        
        # Create precision map
        precision_map = {}
        if MemoryOptimizationStrategy.MIXED_PRECISION in plan.strategies:
            for layer in model_profile.layers:
                # Use FP16 for compute-heavy layers
                if layer.layer_type in ['Linear', 'Conv2d', 'Dense', 'Conv2D']:
                    precision_map[layer.name] = 'fp16'
                else:
                    precision_map[layer.name] = 'fp32'
        
        return TrainingConfig(
            original_batch_size=plan.batch_size * plan.gradient_accumulation_steps,
            optimized_batch_size=plan.batch_size,
            accumulation_steps=plan.gradient_accumulation_steps,
            checkpoint_segments=checkpoint_segments,
            precision_map=precision_map,
            memory_limit=self.memory_limit
        )