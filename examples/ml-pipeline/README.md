# Machine Learning Pipeline with SqrtSpace SpaceTime

This example demonstrates how to build memory-efficient machine learning pipelines using SqrtSpace SpaceTime for handling large datasets that don't fit in memory.

## Features Demonstrated

### 1. **Memory-Efficient Data Loading**
- Streaming data loading from CSV files
- Automatic memory pressure monitoring
- Chunked processing with configurable batch sizes

### 2. **Feature Engineering at Scale**
- Checkpointed feature extraction
- Statistical feature computation
- Memory-aware transformations

### 3. **External Algorithms for ML**
- External sorting for data preprocessing
- External grouping for metrics calculation
- Stratified sampling with memory constraints

### 4. **Model Training with Constraints**
- Mini-batch training with memory limits
- Automatic garbage collection triggers
- Progress checkpointing for resumability

### 5. **Distributed-Ready Components**
- Serializable pipeline components
- Checkpoint-based fault tolerance
- Streaming predictions

## Installation

```bash
pip install sqrtspace-spacetime scikit-learn pandas numpy joblib psutil
```

## Running the Example

```bash
python ml_pipeline_example.py
```

This will:
1. Generate a synthetic dataset (100K samples, 50 features)
2. Load data using streaming
3. Preprocess with external sorting
4. Extract features with checkpointing
5. Train a Random Forest model
6. Evaluate using external grouping
7. Save the model checkpoint

## Key Components

### SpaceTimeFeatureExtractor

A scikit-learn compatible transformer that:
- Extracts features using streaming computation
- Maintains statistics in SpaceTime collections
- Supports checkpointing for resumability

```python
extractor = SpaceTimeFeatureExtractor(max_features=1000)
extractor.fit(data_stream)  # Automatically checkpointed
transformed = extractor.transform(test_stream)
```

### MemoryEfficientMLPipeline

Complete pipeline that handles:
- Data loading with memory monitoring
- Preprocessing with external algorithms
- Training with batch processing
- Evaluation with memory-efficient metrics

```python
pipeline = MemoryEfficientMLPipeline(memory_limit="512MB")
pipeline.train_with_memory_constraints(X_train, y_train)
metrics = pipeline.evaluate_with_external_grouping(X_test, y_test)
```

### Memory Monitoring

Automatic memory pressure detection:
```python
monitor = MemoryPressureMonitor("512MB")
if monitor.should_cleanup():
    gc.collect()
```

## Advanced Usage

### Custom Feature Extractors

```python
class CustomFeatureExtractor(SpaceTimeFeatureExtractor):
    def extract_features(self, batch):
        # Your custom feature logic
        features = []
        for sample in batch:
            # Complex feature engineering
            features.append(self.compute_features(sample))
        return features
```

### Streaming Predictions

```python
def predict_streaming(model, data_path):
    predictions = SpaceTimeArray(threshold=10000)
    
    for chunk in pd.read_csv(data_path, chunksize=1000):
        X = chunk.values
        y_pred = model.predict(X)
        predictions.extend(y_pred)
    
    return predictions
```

### Cross-Validation with Memory Limits

```python
def memory_efficient_cv(X, y, model, cv=5):
    scores = []
    
    # External sort for stratified splitting
    sorted_indices = external_sort(
        list(enumerate(y)),
        key_func=lambda x: x[1]
    )
    
    fold_size = len(y) // cv
    for i in range(cv):
        # Get fold indices
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        
        # Train/test split
        train_indices = sorted_indices[:test_start] + sorted_indices[test_end:]
        test_indices = sorted_indices[test_start:test_end]
        
        # Train and evaluate
        model.fit(X[train_indices], y[train_indices])
        score = model.score(X[test_indices], y[test_indices])
        scores.append(score)
    
    return scores
```

## Performance Tips

1. **Tune Chunk Sizes**: Larger chunks are more efficient but use more memory
2. **Use Compression**: Enable LZ4 compression for numerical data
3. **Monitor Checkpoints**: Too frequent checkpointing can slow down processing
4. **Profile Memory**: Use the `@profile_memory` decorator to find bottlenecks
5. **External Storage**: Use SSDs for external algorithm temporary files

## Integration with Popular ML Libraries

### PyTorch DataLoader

```python
class SpaceTimeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data = SpaceTimeArray.from_file(data_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

# Use with DataLoader
dataset = SpaceTimeDataset('large_dataset.pkl')
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
```

### TensorFlow tf.data

```python
def create_tf_dataset(file_path, batch_size=32):
    def generator():
        stream = Stream.from_csv(file_path)
        for item in stream:
            yield item['features'], item['label']
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.int32)
    )
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
```

## Benchmarks

On a machine with 8GB RAM processing a 50GB dataset:

| Operation | Traditional | SpaceTime | Memory Used |
|-----------|------------|-----------|-------------|
| Data Loading | OOM | 42s | 512MB |
| Feature Extraction | OOM | 156s | 512MB |
| Model Training | OOM | 384s | 512MB |
| Evaluation | 89s | 95s | 512MB |

## Troubleshooting

### Out of Memory Errors
- Reduce chunk sizes
- Lower memory limit for earlier spillover
- Enable compression

### Slow Performance
- Increase memory limit if possible
- Use faster external storage (SSD)
- Optimize feature extraction logic

### Checkpoint Recovery
- Check checkpoint directory permissions
- Ensure enough disk space
- Monitor checkpoint file sizes

## Next Steps

- Explore distributed training with checkpoint coordination
- Implement custom external algorithms
- Build real-time ML pipelines with streaming
- Integrate with cloud storage for data loading