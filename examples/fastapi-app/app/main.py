"""
FastAPI application demonstrating SqrtSpace SpaceTime integration
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from sqrtspace_spacetime import SpaceTimeConfig
from sqrtspace_spacetime.memory import MemoryPressureMonitor

from .config import settings
from .routers import products, analytics, ml, reports
from .services.cache_service import SpaceTimeCache
from .utils.memory import memory_monitor_middleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
cache = SpaceTimeCache()
memory_monitor = MemoryPressureMonitor(settings.SPACETIME_MEMORY_LIMIT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting FastAPI with SqrtSpace SpaceTime")
    
    # Configure SpaceTime
    SpaceTimeConfig.set_defaults(
        memory_limit=settings.SPACETIME_MEMORY_LIMIT,
        external_storage=settings.SPACETIME_EXTERNAL_STORAGE,
        chunk_strategy=settings.SPACETIME_CHUNK_STRATEGY,
        compression=settings.SPACETIME_COMPRESSION
    )
    
    # Initialize services
    app.state.cache = cache
    app.state.memory_monitor = memory_monitor
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    cache.cleanup()


# Create FastAPI app
app = FastAPI(
    title="SqrtSpace SpaceTime FastAPI Demo",
    description="Memory-efficient API with √n space-time tradeoffs",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.middleware("http")(memory_monitor_middleware)

# Include routers
app.include_router(products.router, prefix="/products", tags=["products"])
app.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
app.include_router(ml.router, prefix="/ml", tags=["machine-learning"])
app.include_router(reports.router, prefix="/reports", tags=["reports"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SqrtSpace SpaceTime FastAPI Demo",
        "docs": "/docs",
        "memory_usage": memory_monitor.get_memory_info()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    memory_info = memory_monitor.get_memory_info()
    
    return {
        "status": "healthy",
        "memory": {
            "usage_mb": memory_info["used_mb"],
            "available_mb": memory_info["available_mb"],
            "percentage": memory_info["percentage"],
            "pressure": memory_monitor.check().value
        },
        "cache": cache.get_stats()
    }


@app.get("/system/memory")
async def system_memory():
    """Detailed memory statistics"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    return {
        "process": {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "vms_mb": process.memory_info().vms / 1024 / 1024,
            "cpu_percent": process.cpu_percent(interval=0.1),
            "num_threads": process.num_threads()
        },
        "spacetime": {
            "memory_limit": settings.SPACETIME_MEMORY_LIMIT,
            "external_storage": settings.SPACETIME_EXTERNAL_STORAGE,
            "pressure_level": memory_monitor.check().value,
            "cache_stats": cache.get_stats()
        },
        "system": {
            "total_memory_mb": psutil.virtual_memory().total / 1024 / 1024,
            "available_memory_mb": psutil.virtual_memory().available / 1024 / 1024,
            "memory_percent": psutil.virtual_memory().percent,
            "swap_percent": psutil.swap_memory().percent
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)