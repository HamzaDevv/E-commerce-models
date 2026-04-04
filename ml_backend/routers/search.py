from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from models.search_engine.engine import HybridSearchEngine
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

@router.on_event("startup")
async def startup_event():
    global engine
    try:
        engine = HybridSearchEngine()
    except Exception as e:
        print(f"Warning: Failed to load search engine indices on startup. Did you run the indexer? Error: {e}")

@router.post("/")
async def search_products(q: SearchQuery):
    global engine
    if engine is None:
        try:
            engine = HybridSearchEngine()
        except Exception as e:
            raise HTTPException(status_code=500, detail="Search engine indices are not built yet. Run the indexer script.")
    
    if not q.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty")
        
    try:
        results = engine.search(q.query, top_k=q.top_k)
        return {"query": q.query, "results": results}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
