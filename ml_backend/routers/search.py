from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import traceback

router = APIRouter(
    prefix="/api/search",
    tags=["Search Engine"],
    responses={404: {"description": "Not found"}},
)

# Global engine instance to avoid reloading indices per request
engine = None

class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 10

# Engine is now initialized strictly via FastAPI lifespan in main.py

@router.post("/")
def search_products(q: SearchQuery):
    global engine
    if engine is None:
        try:
            from models.search_engine.engine import HybridSearchEngine
            engine = HybridSearchEngine()
        except Exception as e:
            raise HTTPException(status_code=500, detail="Search engine indices are not built yet. Run the indexer script.")
    
    if not q.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty")
        
    try:
        results = engine.search(q.query, top_k=q.top_k)
        return {"query": q.query, "results": results}
    except Exception as e:
        import logging
        logger = logging.getLogger("ml_backend")
        logger.error(f"Search failed for query '{q.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search operation failed.")
