"""
Product endpoints demonstrating streaming and memory-efficient operations
"""
from fastapi import APIRouter, Query, Response, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Optional, List
import json
import csv
import io
from datetime import datetime

from sqrtspace_spacetime import Stream, external_sort
from sqrtspace_spacetime.checkpoint import CheckpointManager

from ..models import Product, ProductUpdate, BulkUpdateRequest, ImportStatus
from ..services.product_service import ProductService
from ..database import get_db

router = APIRouter()
product_service = ProductService()
checkpoint_manager = CheckpointManager()


@router.get("/")
async def list_products(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None
):
    """Get paginated list of products"""
    filters = {}
    if category:
        filters['category'] = category
    if min_price is not None:
        filters['min_price'] = min_price
    if max_price is not None:
        filters['max_price'] = max_price
    
    return await product_service.get_products(skip, limit, filters)


@router.get("/stream")
async def stream_products(
    category: Optional[str] = None,
    format: str = Query("ndjson", regex="^(ndjson|json)$")
):
    """
    Stream all products as NDJSON or JSON array.
    Memory-efficient streaming for large datasets.
    """
    
    async def generate_ndjson():
        async for product in product_service.stream_products(category):
            yield json.dumps(product.dict()) + "\n"
    
    async def generate_json():
        yield "["
        first = True
        async for product in product_service.stream_products(category):
            if not first:
                yield ","
            yield json.dumps(product.dict())
            first = False
        yield "]"
    
    if format == "ndjson":
        return StreamingResponse(
            generate_ndjson(),
            media_type="application/x-ndjson",
            headers={"X-Accel-Buffering": "no"}
        )
    else:
        return StreamingResponse(
            generate_json(),
            media_type="application/json",
            headers={"X-Accel-Buffering": "no"}
        )


@router.get("/export/csv")
async def export_csv(
    category: Optional[str] = None,
    columns: Optional[List[str]] = Query(None)
):
    """Export products as CSV with streaming"""
    
    if not columns:
        columns = ["id", "name", "sku", "category", "price", "stock", "created_at"]
    
    async def generate():
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns)
        
        # Write header
        writer.writeheader()
        output.seek(0)
        yield output.read()
        output.seek(0)
        output.truncate()
        
        # Stream products in batches
        batch_count = 0
        async for batch in product_service.stream_products_batched(category, batch_size=100):
            for product in batch:
                writer.writerow({col: getattr(product, col) for col in columns})
            
            output.seek(0)
            data = output.read()
            output.seek(0)
            output.truncate()
            yield data
            
            batch_count += 1
            if batch_count % 10 == 0:
                # Yield empty string to keep connection alive
                yield ""
    
    filename = f"products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/search")
async def search_products(
    q: str = Query(..., min_length=2),
    sort_by: str = Query("relevance", regex="^(relevance|price_asc|price_desc|name)$"),
    limit: int = Query(100, ge=1, le=1000)
):
    """
    Search products with memory-efficient sorting.
    Uses external sort for large result sets.
    """
    results = await product_service.search_products(q, sort_by, limit)
    
    # Use external sort if results are large
    if len(results) > 1000:
        sort_key = {
            'price_asc': lambda x: x['price'],
            'price_desc': lambda x: -x['price'],
            'name': lambda x: x['name'],
            'relevance': lambda x: -x['relevance_score']
        }[sort_by]
        
        results = external_sort(results, key_func=sort_key)
    
    return {"results": results[:limit], "total": len(results)}


@router.post("/bulk-update")
async def bulk_update_prices(
    request: BulkUpdateRequest,
    background_tasks: BackgroundTasks
):
    """
    Bulk update product prices with checkpointing.
    Can be resumed if interrupted.
    """
    job_id = f"bulk_update_{datetime.now().timestamp()}"
    
    # Check for existing checkpoint
    checkpoint = checkpoint_manager.restore(job_id)
    if checkpoint:
        return {
            "message": "Resuming previous job",
            "job_id": job_id,
            "progress": checkpoint.get("progress", 0)
        }
    
    # Start background task
    background_tasks.add_task(
        product_service.bulk_update_prices,
        request,
        job_id
    )
    
    return {
        "message": "Bulk update started",
        "job_id": job_id,
        "status_url": f"/products/bulk-update/{job_id}/status"
    }


@router.get("/bulk-update/{job_id}/status")
async def bulk_update_status(job_id: str):
    """Check status of bulk update job"""
    checkpoint = checkpoint_manager.restore(job_id)
    
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "status": checkpoint.get("status", "running"),
        "progress": checkpoint.get("progress", 0),
        "total": checkpoint.get("total", 0),
        "updated": checkpoint.get("updated", 0),
        "errors": checkpoint.get("errors", [])
    }


@router.post("/import/csv")
async def import_csv(
    file_url: str,
    background_tasks: BackgroundTasks
):
    """Import products from CSV file"""
    import_id = f"import_{datetime.now().timestamp()}"
    
    background_tasks.add_task(
        product_service.import_from_csv,
        file_url,
        import_id
    )
    
    return {
        "message": "Import started",
        "import_id": import_id,
        "status_url": f"/products/import/{import_id}/status"
    }


@router.get("/import/{import_id}/status")
async def import_status(import_id: str):
    """Check status of import job"""
    status = await product_service.get_import_status(import_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Import job not found")
    
    return status


@router.get("/statistics")
async def product_statistics():
    """
    Get product statistics using memory-efficient aggregations.
    Uses external grouping for large datasets.
    """
    stats = await product_service.calculate_statistics()
    
    return {
        "total_products": stats["total_products"],
        "total_value": stats["total_value"],
        "by_category": stats["by_category"],
        "price_distribution": stats["price_distribution"],
        "stock_alerts": stats["stock_alerts"],
        "processing_info": {
            "memory_used_mb": stats["memory_used_mb"],
            "external_operations": stats["external_operations"]
        }
    }