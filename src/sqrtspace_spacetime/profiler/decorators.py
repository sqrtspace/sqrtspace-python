"""Decorators for easy profiling."""

import functools
import time
from typing import Any, Callable, Optional

from sqrtspace_spacetime.profiler.profiler import SpaceTimeProfiler


def profile(output_file: Optional[str] = None, 
           print_summary: bool = True) -> Callable:
    """
    Decorator to profile a function.
    
    Args:
        output_file: Optional file to save report
        print_summary: Print summary to console
        
    Example:
        @profile(output_file="profile.json")
        def my_function():
            # Process data
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            profiler = SpaceTimeProfiler()
            result, report = profiler.profile(func, *args, **kwargs)
            
            if print_summary:
                print(report.summary)
            
            if output_file:
                report.save(output_file)
            
            # Store report on function for access
            wrapper.last_report = report
            
            return result
        
        wrapper.last_report = None
        return wrapper
    
    return decorator


def profile_memory(threshold_mb: float = 100,
                  alert: bool = True) -> Callable:
    """
    Decorator to profile memory usage.
    
    Args:
        threshold_mb: Memory threshold in MB to trigger alert
        alert: Print alert if threshold exceeded
        
    Example:
        @profile_memory(threshold_mb=500)
        def process_large_data():
            # Process data
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import psutil
            process = psutil.Process()
            
            start_memory = process.memory_info().rss
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
            finally:
                end_memory = process.memory_info().rss
                end_time = time.time()
                
                memory_used = (end_memory - start_memory) / (1024 * 1024)
                duration = end_time - start_time
                
                # Store metrics
                wrapper.memory_used = memory_used
                wrapper.duration = duration
                
                if alert and memory_used > threshold_mb:
                    print(f"⚠️  Memory Alert: {func.__name__} used {memory_used:.1f}MB "
                          f"(threshold: {threshold_mb}MB)")
                    print(f"   Consider using SpaceTime collections for memory efficiency")
                
                if alert:
                    print(f"Memory: {memory_used:.1f}MB, Time: {duration:.2f}s")
            
            return result
        
        wrapper.memory_used = None
        wrapper.duration = None
        return wrapper
    
    return decorator


def profile_time(threshold_seconds: float = 1.0,
                alert: bool = True) -> Callable:
    """
    Decorator to profile execution time.
    
    Args:
        threshold_seconds: Time threshold to trigger alert
        alert: Print alert if threshold exceeded
        
    Example:
        @profile_time(threshold_seconds=5.0)
        def slow_operation():
            # Time-consuming operation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                wrapper.duration = duration
                
                if alert and duration > threshold_seconds:
                    print(f"⏱️  Time Alert: {func.__name__} took {duration:.2f}s "
                          f"(threshold: {threshold_seconds}s)")
                
                if alert:
                    print(f"Execution time: {duration:.2f}s")
            
            return result
        
        wrapper.duration = None
        return wrapper
    
    return decorator


class ProfileContext:
    """Context manager for profiling code blocks."""
    
    def __init__(self, name: str = "block", print_summary: bool = True):
        self.name = name
        self.print_summary = print_summary
        self.profiler = None
        self.report = None
        self._monitoring = False
    
    def __enter__(self):
        self.profiler = SpaceTimeProfiler()
        self.profiler.start_monitoring()
        self._start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self._start_time
        self.profiler.stop_monitoring()
        
        # Generate simple report
        if self.print_summary:
            peak_memory = max((m[1] for m in self.profiler.memory_timeline), default=0)
            print(f"\nProfile: {self.name}")
            print(f"Duration: {duration:.2f}s")
            print(f"Peak Memory: {peak_memory / (1024*1024):.1f}MB")
            
            if peak_memory > 100 * 1024 * 1024:  # 100MB
                print("💡 Consider using SpaceTime collections for memory optimization")


# Convenience instance
profile_context = ProfileContext