#!/usr/bin/env python3
"""
Tests for SpaceTimeArray with memory pressure simulation.
"""

import unittest
import tempfile
import shutil
import os
import gc
import psutil
from sqrtspace_spacetime import SpaceTimeArray, SpaceTimeConfig


class TestSpaceTimeArray(unittest.TestCase):
    """Test SpaceTimeArray functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        SpaceTimeConfig.set_defaults(
            storage_path=self.temp_dir,
            memory_limit=50 * 1024 * 1024,  # 50MB for testing
            chunk_strategy='sqrt_n'
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_operations(self):
        """Test basic array operations."""
        array = SpaceTimeArray(threshold=100)
        
        # Test append
        for i in range(50):
            array.append(f"item_{i}")
        
        self.assertEqual(len(array), 50)
        self.assertEqual(array[0], "item_0")
        self.assertEqual(array[49], "item_49")
        
        # Test negative indexing
        self.assertEqual(array[-1], "item_49")
        self.assertEqual(array[-50], "item_0")
        
        # Test slice
        slice_result = array[10:20]
        self.assertEqual(len(slice_result), 10)
        self.assertEqual(slice_result[0], "item_10")
    
    def test_automatic_spillover(self):
        """Test automatic spillover to disk."""
        # Create array with small threshold
        array = SpaceTimeArray(threshold=10)
        
        # Add more items than threshold
        for i in range(100):
            array.append(f"value_{i}")
        
        # Check that spillover happened
        self.assertEqual(len(array), 100)
        self.assertGreater(len(array._cold_indices), 0)
        self.assertLessEqual(len(array._hot_data), array.threshold)
        
        # Verify all items are accessible
        for i in range(100):
            self.assertEqual(array[i], f"value_{i}")
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        # Create array with auto threshold
        array = SpaceTimeArray()
        
        # Generate large data items
        large_item = "x" * 10000  # 10KB string
        
        # Add items until memory pressure detected
        for i in range(1000):
            array.append(f"{large_item}_{i}")
            
            # Check memory usage periodically
            if i % 100 == 0:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                # Ensure we're not using excessive memory
                self.assertLess(memory_mb, 300, f"Memory usage too high at iteration {i}")
        
        # Verify all items still accessible
        self.assertEqual(len(array), 1000)
        self.assertTrue(array[0].endswith("_0"))
        self.assertTrue(array[999].endswith("_999"))
    
    def test_large_dataset_sqrt_n_memory(self):
        """Test √n memory usage with large dataset."""
        # Configure for sqrt_n strategy
        SpaceTimeConfig.set_defaults(chunk_strategy='sqrt_n')
        
        n = 10000  # Total items
        sqrt_n = int(n ** 0.5)  # Expected memory items
        
        array = SpaceTimeArray()
        
        # Track initial memory
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Add n items
        for i in range(n):
            array.append({"id": i, "data": f"item_{i}" * 10})
        
        # Force garbage collection
        gc.collect()
        
        # Check memory usage
        final_memory = process.memory_info().rss
        memory_increase_mb = (final_memory - initial_memory) / 1024 / 1024
        
        # Verify sqrt_n behavior
        self.assertEqual(len(array), n)
        self.assertLessEqual(len(array._hot_data), min(1000, sqrt_n * 10))  # Allow buffer due to min chunk size
        self.assertGreaterEqual(len(array._cold_indices), n - min(1000, sqrt_n * 10))
        
        # Memory should be much less than storing all items
        # Rough estimate: each item ~100 bytes, so n items = ~1MB
        # With sqrt_n, should use ~10KB in memory
        self.assertLess(memory_increase_mb, 10, f"Memory increase {memory_increase_mb}MB is too high")
        
        # Verify random access still works
        import random
        for _ in range(100):
            idx = random.randint(0, n - 1)
            self.assertEqual(array[idx]["id"], idx)
    
    def test_persistence_across_sessions(self):
        """Test that storage path is properly created and used."""
        storage_path = os.path.join(self.temp_dir, "persist_test")
        
        # Create array with custom storage path
        array = SpaceTimeArray(threshold=10, storage_path=storage_path)
        
        # Verify storage path is created
        self.assertTrue(os.path.exists(storage_path))
        
        # Add data and force spillover
        for i in range(50):
            array.append(f"persistent_{i}")
        
        # Force spillover
        array._check_and_spill()
        
        # Verify data is still accessible
        self.assertEqual(len(array), 50)
        for i in range(50):
            self.assertEqual(array[i], f"persistent_{i}")
        
        # Verify cold storage file exists
        self.assertIsNotNone(array._cold_storage)
        self.assertTrue(os.path.exists(array._cold_storage))
    
    def test_concurrent_access(self):
        """Test thread-safe access to array."""
        import threading
        
        array = SpaceTimeArray(threshold=100)
        errors = []
        
        def writer(start, count):
            try:
                for i in range(start, start + count):
                    array.append(f"thread_{i}")
            except Exception as e:
                errors.append(e)
        
        def reader(count):
            try:
                for _ in range(count):
                    if len(array) > 0:
                        _ = array[0]  # Just access, don't verify
            except Exception as e:
                errors.append(e)
        
        # Create threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=writer, args=(i * 100, 100))
            threads.append(t)
        
        for i in range(3):
            t = threading.Thread(target=reader, args=(50,))
            threads.append(t)
        
        # Run threads
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Check for errors
        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        self.assertEqual(len(array), 500)


if __name__ == "__main__":
    unittest.main()