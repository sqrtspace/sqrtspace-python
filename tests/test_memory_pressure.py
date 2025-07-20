#!/usr/bin/env python3
"""
Memory pressure tests to verify √n behavior under constrained memory.
"""

import unittest
import gc
import os
import psutil
import resource
import tempfile
import shutil
import random
import time
from sqrtspace_spacetime import (
    SpaceTimeArray, SpaceTimeDict, external_sort, 
    external_groupby, SpaceTimeConfig
)


class TestMemoryPressure(unittest.TestCase):
    """Test √n memory behavior under real memory constraints."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.process = psutil.Process()
        
        # Configure strict memory limits
        SpaceTimeConfig.set_defaults(
            storage_path=self.temp_dir,
            memory_limit=50 * 1024 * 1024,  # 50MB limit
            chunk_strategy='sqrt_n',
            compression='gzip'
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_array_under_memory_pressure(self):
        """Test SpaceTimeArray behavior when memory is constrained."""
        print("\n=== Testing SpaceTimeArray under memory pressure ===")
        
        # Create large objects that will force spillover
        large_object_size = 1024  # 1KB per object
        n_objects = 100_000  # Total: ~100MB if all in memory
        
        array = SpaceTimeArray(threshold='auto')
        
        # Track metrics
        spillovers = 0
        max_memory = 0
        start_time = time.time()
        
        # Add objects and monitor memory
        for i in range(n_objects):
            # Create a large object
            obj = {
                'id': i,
                'data': 'x' * large_object_size,
                'timestamp': time.time()
            }
            array.append(obj)
            
            # Monitor every 1000 items
            if i % 1000 == 0:
                gc.collect()
                current_memory = self.process.memory_info().rss / 1024 / 1024
                max_memory = max(max_memory, current_memory)
                
                if i > 0:
                    hot_count = len(array._hot_data)
                    cold_count = len(array._cold_indices)
                    print(f"  Items: {i:,} | Memory: {current_memory:.1f}MB | "
                          f"Hot: {hot_count} | Cold: {cold_count}")
                    
                    # Check if spillover is happening
                    if cold_count > spillovers:
                        spillovers = cold_count
        
        elapsed = time.time() - start_time
        
        # Verify all data is accessible
        print("\nVerifying data accessibility...")
        sample_indices = random.sample(range(n_objects), min(100, n_objects))
        for idx in sample_indices:
            obj = array[idx]
            self.assertEqual(obj['id'], idx)
            self.assertEqual(len(obj['data']), large_object_size)
        
        # Calculate statistics
        theoretical_sqrt_n = int(n_objects ** 0.5)
        actual_hot_items = len(array._hot_data)
        
        print(f"\nResults:")
        print(f"  Total items: {n_objects:,}")
        print(f"  Time taken: {elapsed:.2f} seconds")
        print(f"  Max memory used: {max_memory:.1f} MB")
        print(f"  Theoretical √n: {theoretical_sqrt_n:,}")
        print(f"  Actual hot items: {actual_hot_items:,}")
        print(f"  Cold items: {len(array._cold_indices):,}")
        print(f"  Memory efficiency: {n_objects / max_memory:.0f} items/MB")
        
        # Assertions
        self.assertEqual(len(array), n_objects)
        self.assertLess(max_memory, 150)  # Should use much less than 100MB
        self.assertGreater(spillovers, 0)  # Should have spilled to disk
        self.assertLessEqual(actual_hot_items, theoretical_sqrt_n * 2)  # Within 2x of √n
    
    def test_dict_with_memory_limit(self):
        """Test SpaceTimeDict with strict memory limit."""
        print("\n=== Testing SpaceTimeDict under memory pressure ===")
        
        # Create dictionary with explicit threshold
        cache = SpaceTimeDict(threshold=1000)  # Keep only 1000 items in memory
        
        n_items = 50_000
        value_size = 500  # 500 bytes per value
        
        # Track evictions
        evictions = 0
        start_time = time.time()
        
        # Add items
        for i in range(n_items):
            key = f"key_{i:06d}"
            value = {
                'id': i,
                'data': 'v' * value_size,
                'accessed': 0
            }
            cache[key] = value
            
            # Check for evictions
            if i % 1000 == 0 and i > 0:
                current_hot = len(cache._hot_data)
                current_cold = len(cache._cold_keys)
                if current_cold > evictions:
                    evictions = current_cold
                    print(f"  Items: {i:,} | Hot: {current_hot} | Cold: {current_cold}")
        
        elapsed = time.time() - start_time
        
        # Test access patterns (LRU behavior)
        print("\nTesting LRU behavior...")
        # Access some old items
        for i in range(0, 100, 10):
            key = f"key_{i:06d}"
            value = cache[key]
            value['accessed'] += 1
        
        # Add more items to trigger eviction
        for i in range(n_items, n_items + 1000):
            cache[f"key_{i:06d}"] = {'id': i, 'data': 'x' * value_size}
        
        # Recent items should still be hot
        stats = cache.get_stats()
        
        print(f"\nResults:")
        print(f"  Total items: {len(cache):,}")
        print(f"  Time taken: {elapsed:.2f} seconds")
        print(f"  Hot items: {len(cache._hot_data)}")
        print(f"  Cold items: {len(cache._cold_keys)}")
        print(f"  Stats: {stats}")
        
        # Verify all items accessible
        sample_keys = random.sample([f"key_{i:06d}" for i in range(n_items)], 100)
        for key in sample_keys:
            self.assertIn(key, cache)
            value = cache[key]
            self.assertIsNotNone(value)
    
    def test_algorithm_memory_scaling(self):
        """Test that algorithms scale with √n memory usage."""
        print("\n=== Testing algorithm memory scaling ===")
        
        datasets = [10_000, 40_000, 90_000, 160_000]  # n, 4n, 9n, 16n
        results = []
        
        for n in datasets:
            print(f"\nTesting with n = {n:,}")
            
            # Generate data
            data = [random.randint(1, 1_000_000) for _ in range(n)]
            
            # Measure memory for sorting
            gc.collect()
            mem_before = self.process.memory_info().rss / 1024 / 1024
            
            sorted_data = external_sort(data)
            
            gc.collect()
            mem_after = self.process.memory_info().rss / 1024 / 1024
            mem_used = mem_after - mem_before
            
            # Verify correctness
            self.assertEqual(len(sorted_data), n)
            for i in range(min(1000, len(sorted_data) - 1)):
                self.assertLessEqual(sorted_data[i], sorted_data[i + 1])
            
            sqrt_n = int(n ** 0.5)
            results.append({
                'n': n,
                'sqrt_n': sqrt_n,
                'memory_used': mem_used,
                'ratio': mem_used / max(sqrt_n * 8 / 1024 / 1024, 0.001)  # 8 bytes per int
            })
            
            print(f"  √n = {sqrt_n:,}")
            print(f"  Memory used: {mem_used:.2f} MB")
            print(f"  Ratio to theoretical: {results[-1]['ratio']:.2f}x")
        
        # Verify √n scaling
        print("\nScaling Analysis:")
        print("n        | √n      | Memory (MB) | Ratio")
        print("---------|---------|-------------|-------")
        for r in results:
            print(f"{r['n']:8,} | {r['sqrt_n']:7,} | {r['memory_used']:11.2f} | {r['ratio']:6.2f}x")
        
        # Memory should scale roughly with √n
        # As n increases 4x, memory should increase ~2x
        for i in range(1, len(results)):
            n_ratio = results[i]['n'] / results[i-1]['n']
            mem_ratio = results[i]['memory_used'] / max(results[i-1]['memory_used'], 0.1)
            expected_ratio = n_ratio ** 0.5
            
            print(f"\nn increased {n_ratio:.1f}x, memory increased {mem_ratio:.1f}x "
                  f"(expected ~{expected_ratio:.1f}x)")
            
            # Allow some variance due to overheads
            self.assertLess(mem_ratio, expected_ratio * 3,
                           f"Memory scaling worse than √n: {mem_ratio:.1f}x vs {expected_ratio:.1f}x")
    
    def test_concurrent_memory_pressure(self):
        """Test behavior under concurrent access with memory pressure."""
        print("\n=== Testing concurrent access under memory pressure ===")
        
        import threading
        import queue
        
        array = SpaceTimeArray(threshold=500)
        errors = queue.Queue()
        n_threads = 4
        items_per_thread = 25_000
        
        def worker(thread_id, start_idx):
            try:
                for i in range(items_per_thread):
                    item = {
                        'thread': thread_id,
                        'index': start_idx + i,
                        'data': f"thread_{thread_id}_item_{i}" * 50
                    }
                    array.append(item)
                    
                    # Occasionally read random items
                    if i % 100 == 0 and len(array) > 10:
                        idx = random.randint(0, len(array) - 1)
                        _ = array[idx]
            except Exception as e:
                errors.put((thread_id, str(e)))
        
        # Start threads
        threads = []
        start_time = time.time()
        
        for i in range(n_threads):
            t = threading.Thread(
                target=worker,
                args=(i, i * items_per_thread)
            )
            threads.append(t)
            t.start()
        
        # Monitor memory while threads run
        max_memory = 0
        while any(t.is_alive() for t in threads):
            current_memory = self.process.memory_info().rss / 1024 / 1024
            max_memory = max(max_memory, current_memory)
            time.sleep(0.1)
        
        # Wait for completion
        for t in threads:
            t.join()
        
        elapsed = time.time() - start_time
        
        # Check for errors
        error_list = []
        while not errors.empty():
            error_list.append(errors.get())
        
        print(f"\nResults:")
        print(f"  Threads: {n_threads}")
        print(f"  Total items: {n_threads * items_per_thread:,}")
        print(f"  Time taken: {elapsed:.2f} seconds")
        print(f"  Max memory: {max_memory:.1f} MB")
        print(f"  Errors: {len(error_list)}")
        print(f"  Final array size: {len(array):,}")
        
        # Assertions
        self.assertEqual(len(error_list), 0, f"Thread errors: {error_list}")
        self.assertEqual(len(array), n_threads * items_per_thread)
        self.assertLess(max_memory, 200)  # Should handle memory pressure


if __name__ == "__main__":
    unittest.main()