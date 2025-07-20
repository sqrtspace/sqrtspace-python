"""
Decorators for automatic checkpointing.
"""

import functools
import inspect
from typing import Any, Callable, List, Optional, Union

from sqrtspace_spacetime.checkpoint.manager import (
    CheckpointManager,
    CheckpointConfig,
    CheckpointStrategy
)


def auto_checkpoint(
    total_iterations: Optional[int] = None,
    strategy: CheckpointStrategy = CheckpointStrategy.ADAPTIVE,
    checkpoint_vars: Optional[List[str]] = None,
    checkpoint_dir: str = ".checkpoints",
    verbose: bool = True
) -> Callable:
    """
    Decorator to automatically checkpoint long-running functions.
    
    Args:
        total_iterations: Total iterations (for √n strategy)
        strategy: Checkpointing strategy
        checkpoint_vars: Variables to checkpoint (None for auto-detect)
        checkpoint_dir: Directory for checkpoints
        verbose: Print checkpoint info
        
    Example:
        @auto_checkpoint(total_iterations=1000000)
        def process_data(data):
            for i, item in enumerate(data):
                # Process item
                checkpoint_state = {'i': i, 'processed': processed}
                yield checkpoint_state
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create checkpoint manager
            config = CheckpointConfig(
                strategy=strategy,
                checkpoint_dir=checkpoint_dir,
                verbose=verbose
            )
            manager = CheckpointManager(config=config)
            
            if total_iterations:
                manager.set_total_iterations(total_iterations)
            
            # Check if resuming from checkpoint
            resume_checkpoint = kwargs.pop('resume_checkpoint', None)
            if resume_checkpoint:
                state, metadata = manager.load(resume_checkpoint)
                print(f"Resuming from checkpoint at iteration {metadata.iteration}")
                # Update function state
                if 'update_state' in kwargs:
                    kwargs['update_state'](state)
            
            # Wrap generator functions
            if inspect.isgeneratorfunction(func):
                return _checkpoint_generator(func, manager, checkpoint_vars, 
                                           *args, **kwargs)
            else:
                # For regular functions, checkpoint based on time/memory
                result = None
                for i in range(total_iterations or 1):
                    if manager.should_checkpoint(i):
                        # Get state from function
                        if hasattr(func, 'get_checkpoint_state'):
                            state = func.get_checkpoint_state()
                        else:
                            state = {'iteration': i, 'args': args, 'kwargs': kwargs}
                        
                        manager.save(state)
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Break if function doesn't need iterations
                    if total_iterations is None:
                        break
                
                return result
        
        # Store checkpoint info on function
        wrapper.checkpoint_manager = None
        wrapper.checkpoint_config = CheckpointConfig(
            strategy=strategy,
            checkpoint_dir=checkpoint_dir
        )
        
        return wrapper
    
    return decorator


def checkpoint_method(
    checkpoint_attrs: Optional[List[str]] = None,
    strategy: CheckpointStrategy = CheckpointStrategy.ADAPTIVE
) -> Callable:
    """
    Decorator for checkpointing class methods.
    
    Args:
        checkpoint_attrs: Instance attributes to checkpoint
        strategy: Checkpointing strategy
        
    Example:
        class DataProcessor:
            @checkpoint_method(checkpoint_attrs=['processed_count', 'results'])
            def process_batch(self, batch):
                for item in batch:
                    self.process_item(item)
                    self.processed_count += 1
    """
    def decorator(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            # Get or create checkpoint manager
            if not hasattr(self, '_checkpoint_manager'):
                config = CheckpointConfig(strategy=strategy)
                self._checkpoint_manager = CheckpointManager(config=config)
            
            # Execute method with checkpointing
            if inspect.isgeneratorfunction(method):
                return _checkpoint_method_generator(
                    method, self, self._checkpoint_manager, 
                    checkpoint_attrs, *args, **kwargs
                )
            else:
                # Regular method
                result = method(self, *args, **kwargs)
                
                # Check if checkpoint needed
                if self._checkpoint_manager.should_checkpoint():
                    state = _get_instance_state(self, checkpoint_attrs)
                    self._checkpoint_manager.save(state)
                
                return result
        
        return wrapper
    
    return decorator


def resumable(
    checkpoint_dir: str = ".checkpoints",
    auto_resume: bool = True
) -> Callable:
    """
    Make function resumable from checkpoints.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        auto_resume: Automatically resume from latest checkpoint
        
    Example:
        @resumable()
        def long_computation():
            for i in range(1000000):
                # Computation
                if should_checkpoint(i):
                    save_checkpoint({'i': i, 'state': state})
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create checkpoint manager
            manager = CheckpointManager(
                checkpoint_id=f"{func.__module__}.{func.__name__}",
                config=CheckpointConfig(checkpoint_dir=checkpoint_dir)
            )
            
            # Check for existing checkpoints
            checkpoints = manager.list_checkpoints()
            
            if checkpoints and auto_resume:
                latest = checkpoints[-1]
                print(f"Found checkpoint at iteration {latest.iteration}")
                
                # Resume from checkpoint
                state, metadata = manager.load()
                
                # Call function with resume state
                return func(*args, resume_state=state, resume_iteration=metadata.iteration, **kwargs)
            else:
                # Normal execution
                return func(*args, **kwargs)
        
        # Add checkpoint methods to function
        wrapper.save_checkpoint = lambda state: manager.save(state)
        wrapper.list_checkpoints = lambda: manager.list_checkpoints()
        wrapper.cleanup_checkpoints = lambda: manager.cleanup()
        
        return wrapper
    
    return decorator


def _checkpoint_generator(func: Callable, manager: CheckpointManager,
                         checkpoint_vars: Optional[List[str]],
                         *args, **kwargs):
    """Handle checkpointing for generator functions."""
    generator = func(*args, **kwargs)
    iteration = 0
    
    try:
        while True:
            # Get next value
            if iteration == 0 and 'resume_state' in kwargs:
                # Skip to resume point
                resume_iter = kwargs['resume_state'].get('iteration', 0)
                for _ in range(resume_iter):
                    next(generator)
                iteration = resume_iter
            
            value = next(generator)
            
            # Check if checkpoint needed
            if manager.should_checkpoint(iteration):
                # Get state
                if isinstance(value, dict):
                    state = value
                else:
                    state = {'iteration': iteration, 'value': value}
                
                # Add checkpoint vars if specified
                if checkpoint_vars:
                    frame = inspect.currentframe().f_back
                    for var in checkpoint_vars:
                        if var in frame.f_locals:
                            state[var] = frame.f_locals[var]
                
                manager.save(state)
            
            yield value
            iteration += 1
            
    except StopIteration:
        pass
    finally:
        if manager.config.verbose:
            stats = manager.get_stats()
            print(f"\nCheckpoint stats: {stats.total_checkpoints} checkpoints, "
                  f"{stats.average_compression:.1f}x compression")


def _checkpoint_method_generator(method: Callable, instance: Any,
                               manager: CheckpointManager,
                               checkpoint_attrs: Optional[List[str]],
                               *args, **kwargs):
    """Handle checkpointing for generator methods."""
    generator = method(instance, *args, **kwargs)
    iteration = 0
    
    try:
        while True:
            value = next(generator)
            
            if manager.should_checkpoint(iteration):
                state = _get_instance_state(instance, checkpoint_attrs)
                state['iteration'] = iteration
                manager.save(state)
            
            yield value
            iteration += 1
            
    except StopIteration:
        pass


def _get_instance_state(instance: Any, attrs: Optional[List[str]] = None) -> dict:
    """Extract state from instance."""
    if attrs:
        return {attr: getattr(instance, attr, None) for attr in attrs}
    else:
        # Auto-detect state (exclude private and callable)
        state = {}
        for attr in dir(instance):
            if not attr.startswith('_') and hasattr(instance, attr):
                value = getattr(instance, attr)
                if not callable(value):
                    try:
                        # Test if pickleable
                        import pickle
                        pickle.dumps(value)
                        state[attr] = value
                    except:
                        pass
        return state