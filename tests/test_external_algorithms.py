#!/usr/bin/env python3
"""
Tests for external algorithms with memory pressure.
"""

import unittest
import random
import gc
import psutil
import time
from sqrtspace_spacetime import external_sort, external_sort_key, external_groupby, SpaceTimeConfig


class TestExternalAlgorithms(unittest.TestCase):
    """Test external algorithms under memory constraints."""
    
    def setUp(self):
        """Set up test environment."""
        SpaceTimeConfig.set_defaults(
            memory_limit=100 * 1024 * 1024,  # 100MB limit
            chunk_strategy='sqrt_n'
        )
        self.process = psutil.Process()
    
    def test_external_sort_small(self):
        """Test external sort with small dataset."""
        data = [random.randint(1, 1000) for _ in range(1000)]
        sorted_data = external_sort(data)
        
        # Verify sorting
        self.assertEqual(len(sorted_data), len(data))
        for i in range(len(sorted_data) - 1):
            self.assertLessEqual(sorted_data[i], sorted_data[i + 1])
        
        # Verify all elements present
        self.assertEqual(sorted(data), sorted_data)
    
    def test_external_sort_large_with_memory_tracking(self):
        """Test external sort with large dataset and memory tracking."""
        n = 1_000_000  # 1 million items
        
        # Generate data
        print(f"\nGenerating {n:,} random integers...")
        data = [random.randint(1, 10_000_000) for _ in range(n)]
        
        # Track memory before sorting
        gc.collect()
        memory_before = self.process.memory_info().rss / 1024 / 1024
        peak_memory = memory_before
        
        # Sort with memory tracking
        print("Sorting with external_sort...")
        start_time = time.time()
        
        # Create a custom monitoring function
        memory_samples = []
        def monitor_memory():
            current = self.process.memory_info().rss / 1024 / 1024
            memory_samples.append(current)
            return current
        
        # Sort data
        sorted_data = external_sort(data)
        
        # Measure final state
        gc.collect()
        memory_after = self.process.memory_info().rss / 1024 / 1024
        elapsed = time.time() - start_time
        
        # Sample memory during verification
        for i in range(0, len(sorted_data) - 1, 10000):
            self.assertLessEqual(sorted_data[i], sorted_data[i + 1])
            if i % 100000 == 0:
                peak_memory = max(peak_memory, monitor_memory())
        
        # Calculate statistics
        memory_increase = memory_after - memory_before
        theoretical_sqrt_n = int(n ** 0.5)
        
        print(f"\nExternal Sort Statistics:")
        print(f"  Items sorted: {n:,}")
        print(f"  Time taken: {elapsed:.2f} seconds")
        print(f"  Memory before: {memory_before:.1f} MB")
        print(f"  Memory after: {memory_after:.1f} MB")
        print(f"  Peak memory: {peak_memory:.1f} MB")
        print(f"  Memory increase: {memory_increase:.1f} MB")
        print(f"  Theoretical √n: {theoretical_sqrt_n:,} items")
        print(f"  Items per MB: {n / max(memory_increase, 0.1):,.0f}")
        
        # Verify memory efficiency
        # With 1M items, sqrt(n) = 1000, so memory should be much less than full dataset
        self.assertLess(memory_increase, 50, f"Memory increase {memory_increase:.1f} MB is too high")
        
        # Verify correctness on sample
        sample_indices = random.sample(range(len(sorted_data) - 1), min(1000, len(sorted_data) - 1))
        for i in sample_indices:
            self.assertLessEqual(sorted_data[i], sorted_data[i + 1])
    
    def test_external_groupby_memory_efficiency(self):
        """Test external groupby with memory tracking."""
        n = 100_000
        
        # Generate data with limited number of groups
        print(f"\nGenerating {n:,} items for groupby...")
        categories = [f"category_{i}" for i in range(100)]
        data = [
            {
                "id": i,
                "category": random.choice(categories),
                "value": random.randint(1, 1000),
                "data": f"data_{i}" * 10  # Make items larger
            }
            for i in range(n)
        ]
        
        # Track memory
        gc.collect()
        memory_before = self.process.memory_info().rss / 1024 / 1024
        
        # Group by category
        print("Grouping by category...")
        start_time = time.time()
        grouped = external_groupby(data, key_func=lambda x: x["category"])
        elapsed = time.time() - start_time
        
        # Measure memory
        gc.collect()
        memory_after = self.process.memory_info().rss / 1024 / 1024
        memory_increase = memory_after - memory_before
        
        print(f"\nExternal GroupBy Statistics:")
        print(f"  Items grouped: {n:,}")
        print(f"  Groups created: {len(grouped)}")
        print(f"  Time taken: {elapsed:.2f} seconds")
        print(f"  Memory increase: {memory_increase:.1f} MB")
        print(f"  Items per MB: {n / max(memory_increase, 0.1):,.0f}")
        
        # Verify correctness
        self.assertEqual(len(grouped), len(categories))
        total_items = sum(len(group) for group in grouped.values())
        self.assertEqual(total_items, n)
        
        # Verify grouping
        for category, items in grouped.items():
            for item in items[:10]:  # Check first 10 items in each group
                self.assertEqual(item["category"], category)
        
        # Memory should be reasonable
        self.assertLess(memory_increase, 100, f"Memory increase {memory_increase:.1f} MB is too high")
    
    def test_stress_test_combined_operations(self):
        """Stress test with combined operations."""
        n = 50_000
        
        print(f"\nRunning stress test with {n:,} items...")
        
        # Generate complex data
        data = []
        for i in range(n):
            data.append({
                "id": i,
                "group": f"group_{i % 50}",
                "value": random.randint(1, 1000),
                "score": random.random(),
                "text": f"This is item {i} with some text" * 5
            })
        
        # Track initial memory
        gc.collect()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Operation 1: Group by
        print("  1. Grouping data...")
        grouped = external_groupby(data, key_func=lambda x: x["group"])
        
        # Operation 2: Sort each group
        print("  2. Sorting each group...")
        for group_key, group_items in grouped.items():
            # Sort by value
            sorted_items = external_sort_key(
                group_items,
                key=lambda x: x["value"]
            )
            grouped[group_key] = sorted_items
        
        # Operation 3: Extract top items from each group
        print("  3. Extracting top items...")
        top_items = []
        for group_items in grouped.values():
            # Get top 10 by value
            top_items.extend(group_items[-10:])
        
        # Operation 4: Final sort
        print("  4. Final sort of top items...")
        final_sorted = external_sort_key(
            top_items,
            key=lambda x: x["score"],
            reverse=True
        )
        
        # Measure final memory
        gc.collect()
        final_memory = self.process.memory_info().rss / 1024 / 1024
        total_memory_increase = final_memory - initial_memory
        
        print(f"\nStress Test Results:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Total increase: {total_memory_increase:.1f} MB")
        print(f"  Groups processed: {len(grouped)}")
        print(f"  Top items selected: {len(top_items)}")
        
        # Verify results
        self.assertEqual(len(grouped), 50)  # 50 groups
        self.assertEqual(len(top_items), 50 * 10)  # Top 10 from each
        self.assertEqual(len(final_sorted), len(top_items))
        
        # Verify sorting
        for i in range(len(final_sorted) - 1):
            self.assertGreaterEqual(
                final_sorted[i]["score"],
                final_sorted[i + 1]["score"]
            )
        
        # Memory should still be reasonable after all operations
        self.assertLess(
            total_memory_increase, 
            150, 
            f"Memory increase {total_memory_increase:.1f} MB is too high"
        )


if __name__ == "__main__":
    unittest.main()