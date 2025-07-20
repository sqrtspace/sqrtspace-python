#!/usr/bin/env python3
"""
Basic usage examples for SqrtSpace SpaceTime.
"""

import time
import random
from sqrtspace_spacetime import (
    SpaceTimeArray,
    SpaceTimeDict,
    external_sort,
    external_groupby,
    Stream,
    SpaceTimeConfig,
)
from sqrtspace_spacetime.profiler import profile, profile_memory
from sqrtspace_spacetime.checkpoint import auto_checkpoint


def example_spacetime_array():
    """Example: Memory-efficient array with automatic spillover."""
    print("\n=== SpaceTimeArray Example ===")
    
    # Create array that keeps only 1000 items in memory
    array = SpaceTimeArray(threshold=1000)
    
    # Add 10,000 items
    print("Adding 10,000 items to SpaceTimeArray...")
    for i in range(10000):
        array.append(f"item_{i}")
    
    print(f"Array length: {len(array)}")
    print(f"Sample items: {array[0]}, {array[5000]}, {array[9999]}")
    
    # Demonstrate memory efficiency
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.1f} MB (much less than storing all in memory)")


def example_external_sort():
    """Example: Sort large dataset with minimal memory."""
    print("\n=== External Sort Example ===")
    
    # Generate large random dataset
    print("Generating 1M random numbers...")
    data = [random.randint(1, 1000000) for _ in range(1000000)]
    
    # Sort using √n memory
    print("Sorting with external_sort (√n memory)...")
    start = time.time()
    sorted_data = external_sort(data)
    elapsed = time.time() - start
    
    # Verify sorting
    is_sorted = all(sorted_data[i] <= sorted_data[i+1] for i in range(len(sorted_data)-1))
    print(f"Sorted correctly: {is_sorted}")
    print(f"Time taken: {elapsed:.2f}s")
    print(f"First 10 elements: {sorted_data[:10]}")


def example_streaming():
    """Example: Process data streams efficiently."""
    print("\n=== Stream Processing Example ===")
    
    # Create sample data
    data = [
        {'name': 'Alice', 'age': 25, 'score': 85},
        {'name': 'Bob', 'age': 30, 'score': 90},
        {'name': 'Charlie', 'age': 25, 'score': 78},
        {'name': 'David', 'age': 30, 'score': 92},
        {'name': 'Eve', 'age': 25, 'score': 88},
    ]
    
    # Stream processing
    result = Stream.from_iterable(data) \
        .filter(lambda x: x['age'] == 25) \
        .map(lambda x: {'name': x['name'], 'grade': 'A' if x['score'] >= 85 else 'B'}) \
        .collect()
    
    print("Filtered and transformed data:")
    for item in result:
        print(f"  {item}")


@profile_memory(threshold_mb=50)
def example_memory_profiling():
    """Example: Profile memory usage."""
    print("\n=== Memory Profiling Example ===")
    
    # Simulate memory-intensive operation
    data = []
    for i in range(100000):
        data.append({
            'id': i,
            'value': random.random(),
            'text': f"Item number {i}" * 10
        })
    
    # Process data
    result = sum(item['value'] for item in data)
    return result


@auto_checkpoint(total_iterations=100)
def example_checkpointing(data):
    """Example: Auto-checkpoint long computation."""
    print("\n=== Checkpointing Example ===")
    
    results = []
    for i, item in enumerate(data):
        # Simulate expensive computation
        time.sleep(0.01)
        result = item ** 2
        results.append(result)
        
        # Yield state for checkpointing
        if i % 10 == 0:
            print(f"Processing item {i}...")
        yield {'i': i, 'results': results}
    
    return results


def example_groupby():
    """Example: Group large dataset efficiently."""
    print("\n=== External GroupBy Example ===")
    
    # Generate sales data
    sales = []
    stores = ['Store_A', 'Store_B', 'Store_C', 'Store_D']
    
    print("Generating 100K sales records...")
    for i in range(100000):
        sales.append({
            'store': random.choice(stores),
            'amount': random.uniform(10, 1000),
            'product': f'Product_{random.randint(1, 100)}'
        })
    
    # Group by store
    print("Grouping by store...")
    grouped = external_groupby(sales, key_func=lambda x: x['store'])
    
    # Calculate totals
    for store, transactions in grouped.items():
        total = sum(t['amount'] for t in transactions)
        print(f"{store}: {len(transactions)} transactions, ${total:,.2f} total")


def example_spacetime_dict():
    """Example: Memory-efficient dictionary with LRU eviction."""
    print("\n=== SpaceTimeDict Example ===")
    
    # Create cache with 100-item memory limit
    cache = SpaceTimeDict(threshold=100)
    
    # Simulate caching expensive computations
    print("Caching 1000 expensive computations...")
    for i in range(1000):
        key = f"computation_{i}"
        # Simulate expensive computation
        value = i ** 2 + random.random()
        cache[key] = value
    
    print(f"Total items: {len(cache)}")
    print(f"Items in memory: {len(cache._hot_data)}")
    print(f"Items on disk: {len(cache._cold_keys)}")
    
    # Access patterns
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")


def main():
    """Run all examples."""
    print("=== SqrtSpace SpaceTime Examples ===")
    
    # Configure SpaceTime
    SpaceTimeConfig.set_defaults(
        memory_limit=512 * 1024 * 1024,  # 512MB
        chunk_strategy='sqrt_n',
        compression='gzip'
    )
    
    # Run examples
    example_spacetime_array()
    example_external_sort()
    example_streaming()
    example_memory_profiling()
    example_groupby()
    example_spacetime_dict()
    
    # Checkpointing example
    data = list(range(100))
    results = list(example_checkpointing(data))
    print(f"Checkpointing completed. Processed {len(results)} items.")
    
    print("\n=== All examples completed! ===")


if __name__ == "__main__":
    main()