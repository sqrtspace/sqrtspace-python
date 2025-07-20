# SqrtSpace SpaceTime FastAPI Sample Application

This sample demonstrates how to build memory-efficient, high-performance APIs using FastAPI and SqrtSpace SpaceTime.

## Features Demonstrated

### 1. **Streaming Endpoints**
- Server-Sent Events (SSE) for real-time data
- Streaming file downloads without memory bloat
- Chunked JSON responses for large datasets

### 2. **Background Tasks**
- Memory-aware task processing
- Checkpointed long-running operations
- Progress tracking with resumable state

### 3. **Data Processing**
- External sorting for large datasets
- Memory-efficient aggregations
- Streaming ETL pipelines

### 4. **Machine Learning Integration**
- Batch prediction with memory limits
- Model training with checkpoints
- Feature extraction pipelines

## Installation

1. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
cp .env.example .env
```

Edit `.env`:
```
SPACETIME_MEMORY_LIMIT=512MB
SPACETIME_EXTERNAL_STORAGE=/tmp/spacetime
SPACETIME_CHUNK_STRATEGY=sqrt_n
SPACETIME_COMPRESSION=gzip
DATABASE_URL=sqlite:///./app.db
```

4. **Initialize database:**
```bash
python init_db.py
```

## Project Structure

```
fastapi-app/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app
│   ├── config.py                  # Configuration
│   ├── models.py                  # Pydantic models
│   ├── database.py                # Database setup
│   ├── routers/
│   │   ├── products.py            # Product endpoints
│   │   ├── analytics.py           # Analytics endpoints
│   │   ├── ml.py                  # ML endpoints
│   │   └── reports.py             # Report generation
│   ├── services/
│   │   ├── product_service.py     # Business logic
│   │   ├── analytics_service.py   # Analytics processing
│   │   ├── ml_service.py          # ML operations
│   │   └── cache_service.py       # SpaceTime caching
│   ├── workers/
│   │   ├── background_tasks.py    # Task workers
│   │   └── checkpointed_jobs.py   # Resumable jobs
│   └── utils/
│       ├── streaming.py           # Streaming helpers
│       └── memory.py              # Memory monitoring
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Usage Examples

### 1. Streaming Large Datasets

```python
# app/routers/products.py
from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
from sqrtspace_spacetime import Stream
import json

router = APIRouter()

@router.get("/products/stream")
async def stream_products(category: str = None):
    """Stream products as newline-delimited JSON"""
    
    async def generate():
        query = db.query(Product)
        if category:
            query = query.filter(Product.category == category)
        
        # Use SpaceTime stream for memory efficiency
        stream = Stream.from_query(query, chunk_size=100)
        
        for product in stream:
            yield json.dumps(product.dict()) + "\n"
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"}
    )
```

### 2. Server-Sent Events for Real-Time Data

```python
# app/routers/analytics.py
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from sqrtspace_spacetime.memory import MemoryPressureMonitor
import asyncio

router = APIRouter()

@router.get("/analytics/realtime")
async def realtime_analytics():
    """Stream real-time analytics using SSE"""
    
    monitor = MemoryPressureMonitor("100MB")
    
    async def event_generator():
        while True:
            # Get current stats
            stats = await analytics_service.get_current_stats()
            
            # Check memory pressure
            if monitor.check() != MemoryPressureLevel.NONE:
                await analytics_service.compact_cache()
            
            yield {
                "event": "update",
                "data": json.dumps(stats)
            }
            
            await asyncio.sleep(1)
    
    return EventSourceResponse(event_generator())
```

### 3. Memory-Efficient CSV Export

```python
# app/routers/reports.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqrtspace_spacetime.file import CsvWriter
import io

router = APIRouter()

@router.get("/reports/export/csv")
async def export_csv(start_date: date, end_date: date):
    """Export large dataset as CSV with streaming"""
    
    async def generate():
        # Create in-memory buffer
        output = io.StringIO()
        writer = CsvWriter(output)
        
        # Write headers
        writer.writerow(["Date", "Orders", "Revenue", "Customers"])
        
        # Stream data in chunks
        async for batch in analytics_service.get_daily_stats_batched(
            start_date, end_date, batch_size=100
        ):
            for row in batch:
                writer.writerow([
                    row.date,
                    row.order_count,
                    row.total_revenue,
                    row.unique_customers
                ])
            
            # Yield buffer content
            output.seek(0)
            data = output.read()
            output.seek(0)
            output.truncate()
            yield data
    
    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=report_{start_date}_{end_date}.csv"
        }
    )
```

### 4. Checkpointed Background Tasks

```python
# app/workers/checkpointed_jobs.py
from sqrtspace_spacetime.checkpoint import CheckpointManager, auto_checkpoint
from sqrtspace_spacetime.collections import SpaceTimeArray

class DataProcessor:
    def __init__(self):
        self.checkpoint_manager = CheckpointManager()
    
    @auto_checkpoint(total_iterations=10000)
    async def process_large_dataset(self, dataset_id: str):
        """Process dataset with automatic checkpointing"""
        
        # Initialize or restore state
        results = SpaceTimeArray(threshold=1000)
        processed_count = 0
        
        # Get data in batches
        async for batch in self.get_data_batches(dataset_id):
            for item in batch:
                # Process item
                result = await self.process_item(item)
                results.append(result)
                processed_count += 1
                
                # Yield state for checkpointing
                if processed_count % 100 == 0:
                    yield {
                        'processed': processed_count,
                        'results': results,
                        'last_item_id': item.id
                    }
        
        return results
```

### 5. Machine Learning with Memory Constraints

```python
# app/services/ml_service.py
from sqrtspace_spacetime.ml import SpaceTimeOptimizer
from sqrtspace_spacetime.streams import Stream
import numpy as np

class MLService:
    def __init__(self):
        self.optimizer = SpaceTimeOptimizer(
            memory_limit="256MB",
            checkpoint_frequency=100
        )
    
    async def train_model(self, training_data_path: str):
        """Train model with memory-efficient data loading"""
        
        # Stream training data
        data_stream = Stream.from_csv(
            training_data_path,
            chunk_size=1000
        )
        
        # Process in mini-batches
        for epoch in range(10):
            for batch in data_stream.batch(32):
                X = np.array([item.features for item in batch])
                y = np.array([item.label for item in batch])
                
                # Train step with automatic checkpointing
                loss = self.optimizer.step(
                    self.model,
                    X, y,
                    epoch=epoch
                )
                
                if self.optimizer.should_checkpoint():
                    await self.save_checkpoint(epoch)
    
    async def batch_predict(self, input_data):
        """Memory-efficient batch prediction"""
        
        results = SpaceTimeArray(threshold=1000)
        
        # Process in chunks to avoid memory issues
        for chunk in Stream.from_iterable(input_data).chunk(100):
            predictions = self.model.predict(chunk)
            results.extend(predictions)
        
        return results
```

### 6. Advanced Caching with SpaceTime

```python
# app/services/cache_service.py
from sqrtspace_spacetime.collections import SpaceTimeDict
from sqrtspace_spacetime.memory import MemoryPressureMonitor
import asyncio

class SpaceTimeCache:
    def __init__(self):
        self.hot_cache = SpaceTimeDict(threshold=1000)
        self.monitor = MemoryPressureMonitor("128MB")
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    async def get(self, key: str):
        """Get with automatic tier management"""
        
        if key in self.hot_cache:
            self.stats['hits'] += 1
            return self.hot_cache[key]
        
        self.stats['misses'] += 1
        
        # Load from database
        value = await self.load_from_db(key)
        
        # Add to cache if memory allows
        if self.monitor.can_allocate(len(str(value))):
            self.hot_cache[key] = value
        else:
            # Trigger cleanup
            self.cleanup()
            self.stats['evictions'] += len(self.hot_cache) // 2
        
        return value
    
    def cleanup(self):
        """Remove least recently used items"""
        # SpaceTimeDict handles LRU automatically
        self.hot_cache.evict_cold_items(0.5)
```

## API Endpoints

### Products API
- `GET /products` - Paginated list
- `GET /products/stream` - Stream all products (NDJSON)
- `GET /products/search` - Memory-efficient search
- `POST /products/bulk-update` - Checkpointed bulk updates
- `GET /products/export/csv` - Streaming CSV export

### Analytics API
- `GET /analytics/summary` - Current statistics
- `GET /analytics/realtime` - SSE stream of live data
- `GET /analytics/trends` - Historical trends
- `POST /analytics/aggregate` - Custom aggregations

### ML API
- `POST /ml/train` - Train model (async with progress)
- `POST /ml/predict/batch` - Batch predictions
- `GET /ml/models/{id}/status` - Training status
- `POST /ml/features/extract` - Feature extraction pipeline

### Reports API
- `POST /reports/generate` - Generate large report
- `GET /reports/{id}/progress` - Check progress
- `GET /reports/{id}/download` - Download completed report

## Running the Application

### Development
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300 \
    --max-requests 1000 \
    --max-requests-jitter 50
```

### With Docker
```bash
docker-compose up
```

## Performance Configuration

### 1. Nginx Configuration
```nginx
location /products/stream {
    proxy_pass http://backend;
    proxy_buffering off;
    proxy_read_timeout 3600;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
}

location /analytics/realtime {
    proxy_pass http://backend;
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 86400;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
}
```

### 2. Worker Configuration
```python
# app/config.py
WORKER_CONFIG = {
    'memory_limit': os.getenv('WORKER_MEMORY_LIMIT', '512MB'),
    'checkpoint_interval': 100,
    'batch_size': 1000,
    'external_storage': '/tmp/spacetime-workers'
}
```

## Monitoring

### Memory Usage Endpoint
```python
@router.get("/system/memory")
async def memory_stats():
    """Get current memory statistics"""
    
    return {
        "current_usage_mb": memory_monitor.current_usage_mb,
        "peak_usage_mb": memory_monitor.peak_usage_mb,
        "available_mb": memory_monitor.available_mb,
        "pressure_level": memory_monitor.pressure_level,
        "cache_stats": cache_service.get_stats(),
        "external_files": len(os.listdir(EXTERNAL_STORAGE))
    }
```

### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, Gauge

stream_requests = Counter('spacetime_stream_requests_total', 'Total streaming requests')
memory_usage = Gauge('spacetime_memory_usage_bytes', 'Current memory usage')
processing_time = Histogram('spacetime_processing_seconds', 'Processing time')
```

## Testing

### Unit Tests
```bash
pytest tests/unit -v
```

### Integration Tests
```bash
pytest tests/integration -v
```

### Load Testing
```bash
locust -f tests/load/locustfile.py --host http://localhost:8000
```

## Best Practices

1. **Always use streaming** for large responses
2. **Configure memory limits** based on container size
3. **Enable checkpointing** for long-running tasks
4. **Monitor memory pressure** in production
5. **Use external storage** on fast SSDs
6. **Set appropriate timeouts** for streaming endpoints
7. **Implement circuit breakers** for memory protection

## Troubleshooting

### High Memory Usage
- Reduce chunk sizes
- Enable more aggressive spillover
- Check for memory leaks in custom code

### Slow Streaming
- Ensure proxy buffering is disabled
- Check network latency
- Optimize chunk sizes

### Failed Checkpoints
- Verify storage permissions
- Check disk space
- Monitor checkpoint frequency

## Learn More

- [SqrtSpace SpaceTime Docs](https://github.com/MarketAlly/Ubiquity)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Streaming Best Practices](https://example.com/streaming)