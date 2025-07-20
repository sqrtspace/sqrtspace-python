"""
SpaceTime Profiler: Profile applications to identify optimization opportunities.

Features:
- Memory pattern analysis (sequential, random, strided)
- Bottleneck detection (memory vs CPU)
- Memory hierarchy awareness (L1/L2/L3/RAM/Disk)
- Hotspot identification
- AI-generated recommendations
"""

import time
import threading
import psutil
import numpy as np
import tracemalloc
import cProfile
import pstats
import io
from collections import defaultdict, deque
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from sqrtspace_spacetime.config import config


class AccessPattern(Enum):
    """Memory access patterns."""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    STRIDED = "strided"
    UNKNOWN = "unknown"


@dataclass
class MemoryAccess:
    """Single memory access event."""
    timestamp: float
    address: int
    size: int
    operation: str  # 'read' or 'write'
    function: str
    line_number: int


@dataclass
class Hotspot:
    """Memory hotspot information."""
    function: str
    file_path: str
    line_number: int
    memory_allocated: int
    memory_freed: int
    net_memory: int
    allocation_count: int
    cpu_time: float
    access_pattern: AccessPattern
    recommendations: List[str]


@dataclass
class BottleneckAnalysis:
    """Analysis of performance bottlenecks."""
    type: str  # 'memory', 'cpu', 'io'
    severity: float  # 0.0 to 1.0
    description: str
    evidence: Dict[str, Any]
    recommendations: List[str]


@dataclass
class ProfilingReport:
    """Complete profiling report."""
    timestamp: str
    duration: float
    peak_memory: int
    total_allocations: int
    memory_timeline: List[Tuple[float, int]]
    cpu_timeline: List[Tuple[float, float]]
    hotspots: List[Hotspot]
    bottlenecks: List[BottleneckAnalysis]
    access_patterns: Dict[str, AccessPattern]
    hierarchy_transitions: Dict[str, int]
    optimization_opportunities: List[Dict[str, Any]]
    summary: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """Save report to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class MemoryTracer:
    """Trace memory accesses and allocations."""
    
    def __init__(self, max_samples: int = 100000):
        self.accesses = deque(maxlen=max_samples)
        self.allocations = defaultdict(list)
        self.start_time = time.time()
        self._tracemalloc_snapshot = None
    
    def start(self):
        """Start memory tracing."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
    
    def stop(self):
        """Stop memory tracing."""
        if tracemalloc.is_tracing():
            self._tracemalloc_snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()
    
    def analyze_pattern(self, accesses: List[MemoryAccess]) -> AccessPattern:
        """Analyze access pattern from recent accesses."""
        if len(accesses) < 10:
            return AccessPattern.UNKNOWN
        
        # Extract addresses
        addresses = [a.address for a in accesses[-100:]]
        
        # Calculate differences
        diffs = np.diff(addresses)
        if len(diffs) == 0:
            return AccessPattern.UNKNOWN
        
        # Check for sequential pattern
        if np.all(diffs > 0) and np.std(diffs) < np.mean(diffs) * 0.1:
            return AccessPattern.SEQUENTIAL
        
        # Check for strided pattern
        unique_diffs = set(diffs)
        if len(unique_diffs) < 5 and np.std(diffs) < 100:
            return AccessPattern.STRIDED
        
        # Otherwise random
        return AccessPattern.RANDOM
    
    def get_top_allocators(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top memory allocators from tracemalloc."""
        if not self._tracemalloc_snapshot:
            return []
        
        top_stats = self._tracemalloc_snapshot.statistics('lineno')[:limit]
        
        result = []
        for stat in top_stats:
            result.append({
                'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                'size': stat.size,
                'count': stat.count,
                'average': stat.size // stat.count if stat.count > 0 else 0
            })
        
        return result


class SpaceTimeProfiler:
    """Main profiler class."""
    
    def __init__(self, sample_interval: float = 0.01):
        self.sample_interval = sample_interval
        self.memory_tracer = MemoryTracer()
        
        # Tracking data
        self.memory_timeline = []
        self.cpu_timeline = []
        self.io_timeline = []
        self.function_stats = defaultdict(lambda: {
            'calls': 0,
            'memory': 0,
            'time': 0.0,
            'allocations': []
        })
        
        self._monitoring = False
        self._monitor_thread = None
        self._start_time = None
    
    def start_monitoring(self):
        """Start background monitoring."""
        self._monitoring = True
        self._start_time = time.time()
        self.memory_tracer.start()
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        self.memory_tracer.stop()
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        process = psutil.Process()
        
        while self._monitoring:
            timestamp = time.time() - self._start_time
            
            # Memory usage
            mem_info = process.memory_info()
            self.memory_timeline.append((timestamp, mem_info.rss))
            
            # CPU usage
            cpu_percent = process.cpu_percent(interval=None)
            self.cpu_timeline.append((timestamp, cpu_percent))
            
            # IO counters (if available)
            try:
                io_counters = process.io_counters()
                self.io_timeline.append((timestamp, {
                    'read_bytes': io_counters.read_bytes,
                    'write_bytes': io_counters.write_bytes,
                    'read_count': io_counters.read_count,
                    'write_count': io_counters.write_count
                }))
            except:
                pass
            
            time.sleep(self.sample_interval)
    
    def profile(self, func: Callable, *args, **kwargs) -> Tuple[Any, ProfilingReport]:
        """Profile a function execution."""
        # Start monitoring
        self.start_monitoring()
        
        # CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
        finally:
            # Stop profiling
            end_time = time.time()
            profiler.disable()
            self.stop_monitoring()
        
        # Generate report
        report = self._generate_report(
            duration=end_time - start_time,
            cpu_profile=profiler
        )
        
        return result, report
    
    def _generate_report(self, duration: float, cpu_profile: cProfile.Profile) -> ProfilingReport:
        """Generate comprehensive profiling report."""
        # Get peak memory
        peak_memory = max((m[1] for m in self.memory_timeline), default=0)
        
        # Analyze components
        hotspots = self._analyze_hotspots(cpu_profile)
        bottlenecks = self._analyze_bottlenecks()
        patterns = self._analyze_access_patterns()
        transitions = self._count_hierarchy_transitions()
        opportunities = self._find_optimization_opportunities(hotspots, bottlenecks)
        
        # Generate summary
        summary = self._generate_summary(duration, peak_memory, hotspots, bottlenecks)
        
        return ProfilingReport(
            timestamp=datetime.now().isoformat(),
            duration=duration,
            peak_memory=peak_memory,
            total_allocations=len(self.memory_tracer.allocations),
            memory_timeline=self.memory_timeline,
            cpu_timeline=self.cpu_timeline,
            hotspots=hotspots,
            bottlenecks=bottlenecks,
            access_patterns=patterns,
            hierarchy_transitions=transitions,
            optimization_opportunities=opportunities,
            summary=summary
        )
    
    def _analyze_hotspots(self, cpu_profile: cProfile.Profile) -> List[Hotspot]:
        """Identify performance hotspots."""
        stats = pstats.Stats(cpu_profile)
        stats.sort_stats('cumulative')
        
        hotspots = []
        top_allocators = self.memory_tracer.get_top_allocators()
        
        # Create lookup for memory stats
        memory_by_file = {stat['file']: stat for stat in top_allocators}
        
        # Analyze top functions
        for func_info, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:20]:
            filename, line_number, function_name = func_info
            
            # Get memory info if available
            mem_info = memory_by_file.get(f"{filename}:{line_number}", {})
            
            # Skip built-in functions
            if filename.startswith('<') or 'site-packages' in filename:
                continue
            
            # Determine access pattern (simplified)
            pattern = AccessPattern.UNKNOWN
            
            # Generate recommendations
            recommendations = []
            if ct > duration * 0.1:  # More than 10% of time
                recommendations.append("Consider optimizing this function - it's a CPU hotspot")
            if mem_info.get('size', 0) > peak_memory * 0.1:  # More than 10% of memory
                recommendations.append("This function allocates significant memory - consider √n optimization")
            
            hotspots.append(Hotspot(
                function=function_name,
                file_path=filename,
                line_number=line_number,
                memory_allocated=mem_info.get('size', 0),
                memory_freed=0,  # Not tracked in simple version
                net_memory=mem_info.get('size', 0),
                allocation_count=mem_info.get('count', 0),
                cpu_time=ct,
                access_pattern=pattern,
                recommendations=recommendations
            ))
        
        return hotspots
    
    def _analyze_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Analyze performance bottlenecks."""
        bottlenecks = []
        
        # Memory bottleneck analysis
        if self.memory_timeline:
            mem_values = [m[1] for m in self.memory_timeline]
            mem_growth = mem_values[-1] - mem_values[0] if len(mem_values) > 1 else 0
            
            if mem_growth > 100 * 1024 * 1024:  # 100MB growth
                bottlenecks.append(BottleneckAnalysis(
                    type="memory",
                    severity=min(1.0, mem_growth / (1024 * 1024 * 1024)),  # GB scale
                    description=f"Significant memory growth detected: {mem_growth / (1024*1024):.1f}MB",
                    evidence={
                        "start_memory": mem_values[0],
                        "end_memory": mem_values[-1],
                        "growth": mem_growth
                    },
                    recommendations=[
                        "Consider using SpaceTime collections for large datasets",
                        "Implement streaming processing with √n buffering",
                        "Use external sorting/grouping algorithms"
                    ]
                ))
        
        # CPU bottleneck analysis
        if self.cpu_timeline:
            cpu_values = [c[1] for c in self.cpu_timeline]
            avg_cpu = np.mean(cpu_values) if cpu_values else 0
            
            if avg_cpu > 80:  # 80% CPU usage
                bottlenecks.append(BottleneckAnalysis(
                    type="cpu",
                    severity=min(1.0, avg_cpu / 100),
                    description=f"High CPU usage detected: {avg_cpu:.1f}% average",
                    evidence={
                        "average_cpu": avg_cpu,
                        "peak_cpu": max(cpu_values) if cpu_values else 0
                    },
                    recommendations=[
                        "Profile CPU hotspots for optimization opportunities",
                        "Consider parallel processing with √n chunk size",
                        "Use more efficient algorithms"
                    ]
                ))
        
        return bottlenecks
    
    def _analyze_access_patterns(self) -> Dict[str, AccessPattern]:
        """Analyze memory access patterns by function."""
        # Simplified implementation
        return {"overall": AccessPattern.UNKNOWN}
    
    def _count_hierarchy_transitions(self) -> Dict[str, int]:
        """Count memory hierarchy transitions."""
        # Simplified implementation
        transitions = {
            "L1_to_L2": 0,
            "L2_to_L3": 0,
            "L3_to_RAM": 0,
            "RAM_to_Disk": 0
        }
        
        # Estimate based on memory growth
        if self.memory_timeline:
            mem_values = [m[1] for m in self.memory_timeline]
            max_mem = max(mem_values) if mem_values else 0
            
            if max_mem > 32 * 1024:  # > L1
                transitions["L1_to_L2"] += 1
            if max_mem > 256 * 1024:  # > L2
                transitions["L2_to_L3"] += 1
            if max_mem > 8 * 1024 * 1024:  # > L3
                transitions["L3_to_RAM"] += 1
            if max_mem > 1024 * 1024 * 1024:  # > 1GB
                transitions["RAM_to_Disk"] += 1
        
        return transitions
    
    def _find_optimization_opportunities(self, 
                                       hotspots: List[Hotspot],
                                       bottlenecks: List[BottleneckAnalysis]) -> List[Dict[str, Any]]:
        """Find SpaceTime optimization opportunities."""
        opportunities = []
        
        # Check for large memory allocations
        for hotspot in hotspots:
            if hotspot.memory_allocated > 10 * 1024 * 1024:  # 10MB
                opportunities.append({
                    "type": "large_allocation",
                    "location": f"{hotspot.file_path}:{hotspot.line_number}",
                    "function": hotspot.function,
                    "memory": hotspot.memory_allocated,
                    "suggestion": "Use SpaceTimeArray or SpaceTimeDict for large collections",
                    "potential_savings": f"{hotspot.memory_allocated * 0.9 / (1024*1024):.1f}MB"
                })
        
        # Check for memory growth patterns
        memory_bottleneck = next((b for b in bottlenecks if b.type == "memory"), None)
        if memory_bottleneck:
            opportunities.append({
                "type": "memory_growth",
                "severity": memory_bottleneck.severity,
                "suggestion": "Implement streaming processing with Stream class",
                "example": "Stream.from_file('data.csv').map(process).chunk(√n).foreach(save)"
            })
        
        return opportunities
    
    def _generate_summary(self, duration: float, peak_memory: int,
                         hotspots: List[Hotspot], 
                         bottlenecks: List[BottleneckAnalysis]) -> str:
        """Generate human-readable summary."""
        summary_parts = [
            f"Profile Summary",
            f"===============",
            f"Duration: {duration:.2f}s",
            f"Peak Memory: {peak_memory / (1024*1024):.1f}MB",
            f"Hotspots Found: {len(hotspots)}",
            f"Bottlenecks: {len(bottlenecks)}",
        ]
        
        if bottlenecks:
            summary_parts.append("\nMain Bottlenecks:")
            for b in bottlenecks[:3]:
                summary_parts.append(f"- {b.type.upper()}: {b.description}")
        
        if hotspots:
            summary_parts.append("\nTop Hotspots:")
            for h in hotspots[:3]:
                summary_parts.append(f"- {h.function} ({h.cpu_time:.2f}s, {h.memory_allocated/(1024*1024):.1f}MB)")
        
        # Add SpaceTime recommendation
        if peak_memory > 100 * 1024 * 1024:  # 100MB
            summary_parts.append("\nSpaceTime Optimization Potential: HIGH")
            summary_parts.append("Consider using SpaceTime collections and algorithms for √n memory reduction")
        
        return "\n".join(summary_parts)