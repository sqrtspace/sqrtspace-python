# SqrtSpace SpaceTime for Python

[![PyPI version](https://badge.fury.io/py/sqrtspace-spacetime.svg)](https://badge.fury.io/py/sqrtspace-spacetime)
[![Python Versions](https://img.shields.io/pypi/pyversions/sqrtspace-spacetime.svg)](https://pypi.org/project/sqrtspace-spacetime/)
[![License](https://img.shields.io/pypi/l/sqrtspace-spacetime.svg)](https://github.com/sqrtspace/sqrtspace-python/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/sqrtspace-spacetime/badge/?version=latest)](https://sqrtspace-spacetime.readthedocs.io/en/latest/?badge=latest)

Memory-efficient algorithms and data structures for Python using Williams' √n space-time tradeoffs.

**Paper Repository**: [github.com/sqrtspace/sqrtspace-paper](https://github.com/sqrtspace/sqrtspace-paper)  

## Installation

```bash
pip install sqrtspace-spacetime
```

For ML features:
```bash
pip install sqrtspace-spacetime[ml]
```

For all features:
```bash
pip install sqrtspace-spacetime[all]
```

## Core Concepts

SpaceTime implements theoretical computer science results showing that many algorithms can achieve better memory usage by accepting slightly slower runtime. The key insight is using √n memory instead of n memory, where n is the input size.

### Key Features

- **Memory-Efficient Collections**: Arrays and dictionaries that automatically spill to disk
- **External Algorithms**: Sort and group large datasets using minimal memory
- **Streaming Operations**: Process files larger than RAM with elegant API
- **Auto-Checkpointing**: Resume long computations from where they left off
- **Memory Profiling**: Identify optimization opportunities in your code
- **ML Optimizations**: Reduce neural network training memory by up to 90%

## Quick Start

### Basic Usage

```python
from sqrtspace_spacetime import SpaceTimeArray, external_sort, Stream

# Memory-efficient array that spills to disk
array = SpaceTimeArray(threshold=10000)
for i in range(1000000):
    array.append(i)

# Sort large datasets with minimal memory
huge_list = list(range(10000000, 0, -1))
sorted_data = external_sort(huge_list)  # Uses only √n memory

# Stream processing
Stream.from_csv('huge_file.csv') \
    .filter(lambda row: row['value'] > 100) \
    .map(lambda row: row['value'] * 1.1) \
    .group_by(lambda row: row['category']) \
    .to_csv('processed.csv')
```

## Examples

### Basic Examples
See [`examples/basic_usage.py`](examples/basic_usage.py) for comprehensive examples of:
- SpaceTimeArray and SpaceTimeDict usage
- External sorting and grouping
- Stream processing
- Memory profiling
- Auto-checkpointing

### FastAPI Web Application
Check out [`examples/fastapi-app/`](examples/fastapi-app/) for a production-ready web application featuring:
- Streaming endpoints for large datasets
- Server-Sent Events (SSE) for real-time data
- Memory-efficient CSV exports
- Checkpointed background tasks
- ML model serving with memory constraints

See the [FastAPI example README](examples/fastapi-app/README.md) for detailed documentation.

### Machine Learning Pipeline
Explore [`examples/ml-pipeline/`](examples/ml-pipeline/) for ML-specific patterns:
- Training models on datasets larger than RAM
- Memory-efficient feature extraction
- Checkpointed training loops
- Streaming predictions
- Integration with PyTorch and TensorFlow

See the [ML Pipeline README](examples/ml-pipeline/README.md) for complete documentation.

### Memory-Efficient Collections

```python
from sqrtspace_spacetime import SpaceTimeArray, SpaceTimeDict

# Array that automatically manages memory
array = SpaceTimeArray(threshold=1000)  # Keep 1000 items in memory
for i in range(1000000):
    array.append(f"item_{i}")

# Dictionary with LRU eviction to disk
cache = SpaceTimeDict(threshold=10000)
for key, value in huge_dataset:
    cache[key] = expensive_computation(value)
```

### External Algorithms

```python
from sqrtspace_spacetime import external_sort, external_groupby

# Sort 100M items using only ~10K memory
data = list(range(100_000_000, 0, -1))
sorted_data = external_sort(data)

# Group by with aggregation
sales = [
    {'store': 'A', 'amount': 100},
    {'store': 'B', 'amount': 200},
    # ... millions more
]

by_store = external_groupby(
    sales,
    key_func=lambda x: x['store']
)

# Aggregate with minimal memory
from sqrtspace_spacetime.algorithms import groupby_sum
totals = groupby_sum(
    sales,
    key_func=lambda x: x['store'],
    value_func=lambda x: x['amount']
)
```

### Streaming Operations

```python
from sqrtspace_spacetime import Stream

# Process large files efficiently
stream = Stream.from_csv('sales_2023.csv')
    .filter(lambda row: row['amount'] > 0)
    .map(lambda row: {
        'month': row['date'][:7],
        'amount': float(row['amount'])
    })
    .group_by(lambda row: row['month'])
    .to_csv('monthly_summary.csv')

# Chain operations
top_products = Stream.from_jsonl('products.jsonl') \
    .filter(lambda p: p['in_stock']) \
    .sort(key=lambda p: p['revenue'], reverse=True) \
    .take(100) \
    .collect()
```

### Auto-Checkpointing

```python
from sqrtspace_spacetime.checkpoint import auto_checkpoint

@auto_checkpoint(total_iterations=1000000)
def process_large_dataset(data):
    results = []
    for i, item in enumerate(data):
        # Process item
        result = expensive_computation(item)
        results.append(result)
        
        # Yield state for checkpointing
        yield {'i': i, 'results': results}
    
    return results

# Automatically resumes from checkpoint if interrupted
results = process_large_dataset(huge_dataset)
```

### Memory Profiling

```python
from sqrtspace_spacetime.profiler import profile, profile_memory

@profile(output_file="profile.json")
def my_algorithm(data):
    # Process data
    return results

# Get detailed memory analysis
result, report = my_algorithm(data)
print(report.summary)

# Simple memory tracking
@profile_memory(threshold_mb=100)
def memory_heavy_function():
    # Alerts if memory usage exceeds threshold
    large_list = list(range(10000000))
    return sum(large_list)
```

### ML Memory Optimization

```python
from sqrtspace_spacetime.ml import MLMemoryOptimizer
import torch.nn as nn

# Analyze model memory usage
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

optimizer = MLMemoryOptimizer()
profile = optimizer.analyze_model(model, input_shape=(784,), batch_size=32)

# Get optimization plan
plan = optimizer.optimize(profile, target_batch_size=128)
print(plan.explanation)

# Apply optimizations
config = optimizer.get_training_config(plan, profile)
```

## Advanced Features

### Memory Pressure Handling

```python
from sqrtspace_spacetime.memory import MemoryMonitor, LoggingHandler

# Monitor memory pressure
monitor = MemoryMonitor()
monitor.add_handler(LoggingHandler())

# Your arrays automatically respond to memory pressure
array = SpaceTimeArray()
# Arrays spill to disk when memory is low
```

### Configuration

```python
from sqrtspace_spacetime import SpaceTimeConfig

# Global configuration
SpaceTimeConfig.set_defaults(
    memory_limit=2 * 1024**3,  # 2GB
    chunk_strategy='sqrt_n',
    compression='gzip',
    external_storage_path='/fast/ssd/temp'
)
```

### Parallel Processing

```python
from sqrtspace_spacetime.batch import BatchProcessor

processor = BatchProcessor(
    memory_threshold=0.8,
    checkpoint_enabled=True
)

# Process in memory-efficient batches
result = processor.process(
    huge_list,
    lambda batch: [transform(item) for item in batch]
)

print(f"Processed {result.get_success_count()} items")
```

## Real-World Examples

### Processing Large CSV Files

```python
from sqrtspace_spacetime import Stream
from sqrtspace_spacetime.profiler import profile_memory

@profile_memory(threshold_mb=500)
def analyze_sales_data(filename):
    # Stream process to stay under memory limit
    return Stream.from_csv(filename) \
        .filter(lambda row: row['status'] == 'completed') \
        .map(lambda row: {
            'product': row['product_id'],
            'revenue': float(row['price']) * int(row['quantity'])
        }) \
        .group_by(lambda row: row['product']) \
        .sort(key=lambda group: sum(r['revenue'] for r in group[1]), reverse=True) \
        .take(10) \
        .collect()

top_products = analyze_sales_data('sales_2023.csv')
```

### Training Large Neural Networks

```python
from sqrtspace_spacetime.ml import MLMemoryOptimizer, GradientCheckpointer
import torch.nn as nn

# Memory-efficient training
def train_large_model(model, train_loader, epochs=10):
    # Analyze memory requirements
    optimizer = MLMemoryOptimizer()
    profile = optimizer.analyze_model(model, input_shape=(3, 224, 224), batch_size=32)
    
    # Get optimization plan
    plan = optimizer.optimize(profile, target_batch_size=128)
    
    # Apply gradient checkpointing
    checkpointer = GradientCheckpointer()
    model = checkpointer.apply_checkpointing(model, plan.checkpoint_layers)
    
    # Train with optimized settings
    for epoch in range(epochs):
        for batch in train_loader:
            # Training loop with automatic memory management
            pass
```

### Data Pipeline with Checkpoints

```python
from sqrtspace_spacetime import Stream
from sqrtspace_spacetime.checkpoint import auto_checkpoint

@auto_checkpoint(total_iterations=1000000)
def process_user_events(event_file):
    processed = 0
    
    for event in Stream.from_jsonl(event_file):
        # Complex processing
        user_profile = enhance_profile(event)
        recommendations = generate_recommendations(user_profile)
        
        save_to_database(recommendations)
        processed += 1
        
        # Checkpoint state
        yield {'processed': processed, 'last_event': event['id']}
    
    return processed

# Automatically resumes if interrupted
total = process_user_events('events.jsonl')
```

## Performance Benchmarks

| Operation | Standard Python | SpaceTime | Memory Reduction | Time Overhead |
|-----------|----------------|-----------|------------------|---------------|
| Sort 10M integers | 400MB | 20MB | 95% | 40% |
| Process 1GB CSV | 1GB | 32MB | 97% | 20% |
| Group by on 1M rows | 200MB | 14MB | 93% | 30% |
| Neural network training | 8GB | 2GB | 75% | 15% |

## API Reference

### Collections
- `SpaceTimeArray`: Memory-efficient list with disk spillover
- `SpaceTimeDict`: Memory-efficient dictionary with LRU eviction

### Algorithms
- `external_sort()`: Sort large datasets with √n memory
- `external_groupby()`: Group large datasets with √n memory
- `external_join()`: Join large datasets efficiently

### Streaming
- `Stream`: Lazy evaluation stream processing
- `FileStream`: Stream lines from files
- `CSVStream`: Stream CSV rows
- `JSONLStream`: Stream JSON Lines

### Memory Management
- `MemoryMonitor`: Monitor memory pressure
- `MemoryPressureHandler`: Custom pressure handlers

### Checkpointing
- `@auto_checkpoint`: Automatic checkpointing decorator
- `CheckpointManager`: Manual checkpoint control

### ML Optimization
- `MLMemoryOptimizer`: Analyze and optimize models
- `GradientCheckpointer`: Apply gradient checkpointing

### Profiling
- `@profile`: Full profiling decorator
- `@profile_memory`: Memory-only profiling
- `SpaceTimeProfiler`: Programmatic profiling

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Citation

If you use SpaceTime in your research, please cite:

```bibtex
@software{sqrtspace_spacetime,
  title = {SqrtSpace SpaceTime: Memory-Efficient Python Library},
  author={Friedel Jr., David H.},
  year = {2025},
  url = {https://github.com/sqrtspace/sqrtspace-python}
}
```

## Links

- [Documentation](https://sqrtspace-spacetime.readthedocs.io)
- [PyPI Package](https://pypi.org/project/sqrtspace-spacetime/)
- [GitHub Repository](https://github.com/sqrtspace/sqrtspace-python)
- [Issue Tracker](https://github.com/sqrtspace/sqrtspace-python/issues)
