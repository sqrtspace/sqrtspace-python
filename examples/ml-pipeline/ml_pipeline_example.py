#!/usr/bin/env python3
"""
Machine Learning Pipeline with SqrtSpace SpaceTime

Demonstrates memory-efficient ML workflows including:
- Large dataset processing
- Feature extraction with checkpointing
- Model training with memory constraints
- Batch prediction with streaming
- Cross-validation with external sorting
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
import time
from typing import Iterator, Tuple, List, Dict, Any

from sqrtspace_spacetime import (
    SpaceTimeArray,
    SpaceTimeDict,
    Stream,
    external_sort,
    external_groupby,
    SpaceTimeConfig
)
from sqrtspace_spacetime.checkpoint import auto_checkpoint, CheckpointManager
from sqrtspace_spacetime.memory import MemoryPressureMonitor, profile_memory
from sqrtspace_spacetime.ml import SpaceTimeOptimizer
from sqrtspace_spacetime.profiler import profile


# Configure SpaceTime for ML workloads
SpaceTimeConfig.set_defaults(
    memory_limit=1024 * 1024 * 1024,  # 1GB
    chunk_strategy='sqrt_n',
    compression='lz4'  # Fast compression for numerical data
)


class SpaceTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Memory-efficient feature extractor using SpaceTime"""
    
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.feature_stats = SpaceTimeDict(threshold=100)
        self.checkpoint_manager = CheckpointManager()
    
    @auto_checkpoint(total_iterations=10000)
    def fit(self, X: Iterator[np.ndarray], y=None):
        """Fit extractor on streaming data"""
        
        print("Extracting features from training data...")
        
        # Accumulate statistics in SpaceTime collections
        feature_sums = SpaceTimeArray(threshold=self.max_features)
        feature_counts = SpaceTimeArray(threshold=self.max_features)
        
        for batch_idx, batch in enumerate(X):
            for row in batch:
                # Update running statistics
                if len(feature_sums) < len(row):
                    feature_sums.extend([0] * (len(row) - len(feature_sums)))
                    feature_counts.extend([0] * (len(row) - len(feature_counts)))
                
                for i, value in enumerate(row):
                    feature_sums[i] += value
                    feature_counts[i] += 1
            
            # Checkpoint every 100 batches
            if batch_idx % 100 == 0:
                yield {
                    'batch_idx': batch_idx,
                    'feature_sums': feature_sums,
                    'feature_counts': feature_counts
                }
        
        # Calculate means
        self.feature_means_ = []
        for i in range(len(feature_sums)):
            mean = feature_sums[i] / feature_counts[i] if feature_counts[i] > 0 else 0
            self.feature_means_.append(mean)
            self.feature_stats[f'mean_{i}'] = mean
        
        return self
    
    def transform(self, X: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        """Transform streaming data"""
        
        for batch in X:
            # Normalize using stored means
            transformed = np.array(batch)
            for i, mean in enumerate(self.feature_means_):
                transformed[:, i] -= mean
            
            yield transformed


class MemoryEfficientMLPipeline:
    """Complete ML pipeline with memory management"""
    
    def __init__(self, memory_limit: str = "512MB"):
        self.memory_monitor = MemoryPressureMonitor(memory_limit)
        self.checkpoint_manager = CheckpointManager()
        self.feature_extractor = SpaceTimeFeatureExtractor()
        self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        self.optimizer = SpaceTimeOptimizer(
            memory_limit=memory_limit,
            checkpoint_frequency=100
        )
    
    @profile_memory(threshold_mb=256)
    def load_data_streaming(self, file_path: str, chunk_size: int = 1000) -> Iterator:
        """Load large dataset in memory-efficient chunks"""
        
        print(f"Loading data from {file_path} in chunks of {chunk_size}...")
        
        # Simulate loading large CSV in chunks
        for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
            # Convert to numpy array
            X = chunk.drop('target', axis=1).values
            y = chunk['target'].values
            
            # Check memory pressure
            if self.memory_monitor.should_cleanup():
                print(f"Memory pressure detected at chunk {chunk_idx}, triggering cleanup")
                import gc
                gc.collect()
            
            yield X, y
    
    def preprocess_with_external_sort(self, data_iterator: Iterator) -> Tuple[SpaceTimeArray, SpaceTimeArray]:
        """Preprocess and sort data using external algorithms"""
        
        print("Preprocessing data with external sorting...")
        
        X_all = SpaceTimeArray(threshold=10000)
        y_all = SpaceTimeArray(threshold=10000)
        
        # Collect all data
        for X_batch, y_batch in data_iterator:
            X_all.extend(X_batch.tolist())
            y_all.extend(y_batch.tolist())
        
        # Sort by target value for stratified splitting
        print(f"Sorting {len(y_all)} samples by target value...")
        
        # Create index pairs
        indexed_data = [(i, y) for i, y in enumerate(y_all)]
        
        # External sort by target value
        sorted_indices = external_sort(
            indexed_data,
            key_func=lambda x: x[1]
        )
        
        # Reorder data
        X_sorted = SpaceTimeArray(threshold=10000)
        y_sorted = SpaceTimeArray(threshold=10000)
        
        for idx, _ in sorted_indices:
            X_sorted.append(X_all[idx])
            y_sorted.append(y_all[idx])
        
        return X_sorted, y_sorted
    
    def extract_features_checkpointed(self, X: SpaceTimeArray) -> SpaceTimeArray:
        """Extract features with checkpointing"""
        
        print("Extracting features with checkpointing...")
        
        job_id = f"feature_extraction_{int(time.time())}"
        
        # Check for existing checkpoint
        checkpoint = self.checkpoint_manager.restore(job_id)
        start_idx = checkpoint.get('last_idx', 0) if checkpoint else 0
        
        features = SpaceTimeArray(threshold=10000)
        
        # Load partial results if resuming
        if checkpoint and 'features' in checkpoint:
            features = checkpoint['features']
        
        # Process in batches
        batch_size = 100
        for i in range(start_idx, len(X), batch_size):
            batch = X[i:i + batch_size]
            
            # Simulate feature extraction
            batch_features = []
            for sample in batch:
                # Example: statistical features
                features_dict = {
                    'mean': np.mean(sample),
                    'std': np.std(sample),
                    'min': np.min(sample),
                    'max': np.max(sample),
                    'median': np.median(sample)
                }
                batch_features.append(list(features_dict.values()))
            
            features.extend(batch_features)
            
            # Checkpoint every 1000 samples
            if (i + batch_size) % 1000 == 0:
                self.checkpoint_manager.save(job_id, {
                    'last_idx': i + batch_size,
                    'features': features
                })
                print(f"Checkpoint saved at index {i + batch_size}")
        
        # Clean up checkpoint
        self.checkpoint_manager.delete(job_id)
        
        return features
    
    @profile
    def train_with_memory_constraints(self, X: SpaceTimeArray, y: SpaceTimeArray):
        """Train model with memory-aware batch processing"""
        
        print("Training model with memory constraints...")
        
        # Convert to numpy arrays in batches
        batch_size = min(1000, len(X))
        
        for epoch in range(3):  # Multiple epochs
            print(f"\nEpoch {epoch + 1}/3")
            
            # Shuffle data
            indices = list(range(len(X)))
            np.random.shuffle(indices)
            
            # Train in mini-batches
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i + batch_size]
                
                X_batch = np.array([X[idx] for idx in batch_indices])
                y_batch = np.array([y[idx] for idx in batch_indices])
                
                # Partial fit (for models that support it)
                if hasattr(self.model, 'partial_fit'):
                    self.model.partial_fit(X_batch, y_batch)
                else:
                    # For RandomForest, we'll fit on full data once
                    if epoch == 0 and i == 0:
                        # Collect all data for initial fit
                        X_train = np.array(X.to_list())
                        y_train = np.array(y.to_list())
                        self.model.fit(X_train, y_train)
                        break
                
                # Check memory
                if self.memory_monitor.should_cleanup():
                    import gc
                    gc.collect()
                    print(f"Memory cleanup at batch {i // batch_size}")
    
    def evaluate_with_external_grouping(self, X: SpaceTimeArray, y: SpaceTimeArray) -> Dict[str, float]:
        """Evaluate model using external grouping for metrics"""
        
        print("Evaluating model performance...")
        
        # Make predictions in batches
        predictions = SpaceTimeArray(threshold=10000)
        
        batch_size = 1000
        for i in range(0, len(X), batch_size):
            X_batch = np.array(X[i:i + batch_size])
            y_pred = self.model.predict(X_batch)
            predictions.extend(y_pred.tolist())
        
        # Group by actual vs predicted for confusion matrix
        results = []
        for i in range(len(y)):
            results.append({
                'actual': y[i],
                'predicted': predictions[i],
                'correct': y[i] == predictions[i]
            })
        
        # Use external groupby for metrics
        accuracy_groups = external_groupby(
            results,
            key_func=lambda x: x['correct']
        )
        
        correct_count = len(accuracy_groups.get(True, []))
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        # Class-wise metrics
        class_groups = external_groupby(
            results,
            key_func=lambda x: (x['actual'], x['predicted'])
        )
        
        return {
            'accuracy': accuracy,
            'total_samples': total_count,
            'correct_predictions': correct_count,
            'class_distribution': {str(k): len(v) for k, v in class_groups.items()}
        }
    
    def save_model_checkpoint(self, path: str):
        """Save model with metadata"""
        
        checkpoint = {
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'metadata': {
                'timestamp': time.time(),
                'memory_limit': self.memory_monitor.memory_limit,
                'feature_stats': dict(self.feature_extractor.feature_stats)
            }
        }
        
        joblib.dump(checkpoint, path)
        print(f"Model saved to {path}")


def generate_synthetic_data(n_samples: int = 100000, n_features: int = 50):
    """Generate synthetic dataset for demonstration"""
    
    print(f"Generating synthetic dataset: {n_samples} samples, {n_features} features...")
    
    # Generate in chunks to avoid memory issues
    chunk_size = 10000
    
    with open('synthetic_data.csv', 'w') as f:
        # Write header
        headers = [f'feature_{i}' for i in range(n_features)] + ['target']
        f.write(','.join(headers) + '\n')
        
        # Generate data in chunks
        for i in range(0, n_samples, chunk_size):
            chunk_samples = min(chunk_size, n_samples - i)
            
            # Generate features
            X = np.random.randn(chunk_samples, n_features)
            
            # Generate target (binary classification)
            # Target depends on sum of first 10 features
            y = (X[:, :10].sum(axis=1) > 0).astype(int)
            
            # Write to CSV
            for j in range(chunk_samples):
                row = list(X[j]) + [y[j]]
                f.write(','.join(map(str, row)) + '\n')
            
            if (i + chunk_size) % 50000 == 0:
                print(f"Generated {i + chunk_size} samples...")
    
    print("Synthetic data generation complete!")


def main():
    """Run complete ML pipeline example"""
    
    print("=== SqrtSpace SpaceTime ML Pipeline Example ===\n")
    
    # Generate synthetic data
    generate_synthetic_data(n_samples=100000, n_features=50)
    
    # Create pipeline
    pipeline = MemoryEfficientMLPipeline(memory_limit="512MB")
    
    # Load and preprocess data
    print("\n1. Loading data with streaming...")
    data_iterator = pipeline.load_data_streaming('synthetic_data.csv', chunk_size=5000)
    
    print("\n2. Preprocessing with external sort...")
    X_sorted, y_sorted = pipeline.preprocess_with_external_sort(data_iterator)
    print(f"Loaded {len(X_sorted)} samples")
    
    print("\n3. Extracting features with checkpointing...")
    X_features = pipeline.extract_features_checkpointed(X_sorted)
    
    print("\n4. Training model with memory constraints...")
    # Split data (80/20)
    split_idx = int(0.8 * len(X_features))
    X_train = SpaceTimeArray(X_features[:split_idx])
    y_train = SpaceTimeArray(y_sorted[:split_idx])
    X_test = SpaceTimeArray(X_features[split_idx:])
    y_test = SpaceTimeArray(y_sorted[split_idx:])
    
    pipeline.train_with_memory_constraints(X_train, y_train)
    
    print("\n5. Evaluating with external grouping...")
    metrics = pipeline.evaluate_with_external_grouping(X_test, y_test)
    
    print("\n=== Results ===")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Total Test Samples: {metrics['total_samples']}")
    print(f"Correct Predictions: {metrics['correct_predictions']}")
    
    print("\n6. Saving model checkpoint...")
    pipeline.save_model_checkpoint('spacetime_model.joblib')
    
    # Memory statistics
    print("\n=== Memory Statistics ===")
    memory_info = pipeline.memory_monitor.get_memory_info()
    print(f"Peak Memory Usage: {memory_info['peak_mb']:.2f} MB")
    print(f"Current Memory Usage: {memory_info['used_mb']:.2f} MB")
    print(f"Memory Limit: {memory_info['limit_mb']:.2f} MB")
    
    print("\n=== Pipeline Complete! ===")


if __name__ == "__main__":
    main()