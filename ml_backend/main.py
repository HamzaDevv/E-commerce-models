import os
# Prevent OpenMP thread-pool collisions between PyTorch and Faiss.
# KMP_DUPLICATE_LIB_OK: macOS-specific fallback for duplicate libiomp5.
# OMP_NUM_THREADS=1: Forces single-threaded OpenMP — prevents segfaults
#   when multiple native libs (PyTorch, Faiss, etc.) each try to spawn
#   their own thread pools. Safe for inference; FastAPI handles concurrency.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'True')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routers import search, recommendation, basket_rag

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n--- [App Initialization: Booting ML Engines Sequence] ---")
    
    # STEP 1: Boot BasketGPT (autoregressive generation engine — PyTorch)
    # Must be first: locks the OpenMP thread pool before any Faiss C++ lib loads.
    try:
        from models.basket_engine.engine import BasketCompletionEngine
        recommendation.basket_engine = BasketCompletionEngine()
        print("✅ BasketGPT Engine (PyTorch) booted successfully.")
    except Exception as e:
        print(f"⚠️  BasketGPT Engine failed to load: {e}")

    # STEP 2: Boot BasketRAG (contrastive retrieval engine — PyTorch + Faiss)
    try:
        from models.basket_rag.engine import BasketRAGEngine
        basket_rag.basket_rag_engine = BasketRAGEngine()
        print("✅ BasketRAG Engine (PyTorch + Faiss) booted successfully.")
    except Exception as e:
        print(f"⚠️  BasketRAG Engine failed to load: {e}")

    # STEP 3: Boot Hybrid Search Engine (Faiss BM25+Vector)
    # Always last — Faiss must not seize the thread pool before PyTorch does.
    try:
        from models.search_engine.engine import HybridSearchEngine
        search.engine = HybridSearchEngine()
        print("✅ Hybrid Search Engine (Faiss) booted successfully.")
    except Exception as e:
        print(f"⚠️  Search Engine failed to load: {e}")
        
    print("----------------------------------------------------------\n")
    yield
    # (Shutdown code would go here)

app = FastAPI(
    title="Instamart ML Models API",
    description="Centralized API for various E-Commerce Machine Learning Models including Search, Recommendations, and Dynamic Pricing.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific domains in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the routers for different sub-systems
app.include_router(recommendation.router)
app.include_router(basket_rag.router)
app.include_router(search.router)

@app.get("/")
def root():
    return {"message": "Welcome to the Instamart ML Backend API", "status": "online"}
