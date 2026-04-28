import os
import logging
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Prevent OpenMP thread-pool collisions between PyTorch and Faiss.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'True')
os.environ.setdefault('OMP_NUM_THREADS', '1')

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ml_backend")

from routers import search, recommendation, basket_rag

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- [App Initialization: Booting ML Engines Sequence] ---")
    
    # STEP 1: Boot BasketGPT
    try:
        from models.basket_engine.engine import BasketCompletionEngine
        recommendation.basket_engine = BasketCompletionEngine()
        logger.info("✅ BasketGPT Engine (PyTorch) booted successfully.")
    except Exception as e:
        logger.error(f"❌ BasketGPT Engine failed to load: {e}")

    # STEP 2: Boot BasketRAG
    try:
        from models.basket_rag.engine import BasketRAGEngine
        basket_rag.basket_rag_engine = BasketRAGEngine()
        logger.info("✅ BasketRAG Engine (PyTorch + Faiss) booted successfully.")
    except Exception as e:
        logger.error(f"❌ BasketRAG Engine failed to load: {e}")

    # STEP 3: Boot Hybrid Search Engine
    try:
        from models.search_engine.engine import HybridSearchEngine
        search.engine = HybridSearchEngine()
        logger.info("✅ Hybrid Search Engine (BM25 + Fuzzy) booted successfully.")
    except Exception as e:
        logger.error(f"❌ Search Engine failed to load: {e}")
        
    logger.info("----------------------------------------------------------\n")
    yield
    logger.info("Shutting down ML Backend...")

# Custom Middleware for Request Size Limiting (1MB)
class LimitUploadSize(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            if "content-length" in request.headers:
                if int(request.headers["content-length"]) > self.max_upload_size:
                    return JSONResponse(status_code=413, content={"detail": "Request entity too large"})
        return await call_next(request)

app = FastAPI(
    title="NeuralGrocer ML Models API",
    description="Secured API for E-Commerce Machine Learning Models.",
    version="1.1.0",
    lifespan=lifespan
)

# 1. Trusted Host Middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*.local"]
)

# 2. CORS Lockdown
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
allowed_origins = [origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 3. Request Size Limit (1MB)
app.add_middleware(LimitUploadSize, max_upload_size=1_048_576)

# 4. Global Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please contact support."}
    )

# Mount Routers
app.include_router(recommendation.router)
app.include_router(basket_rag.router)
app.include_router(search.router)

@app.get("/")
def root():
    return {
        "message": "NeuralGrocer ML Backend API",
        "status": "online",
        "engines": {
            "basket_gpt": "loaded" if recommendation.basket_engine else "missing",
            "basket_rag": "loaded" if basket_rag.basket_rag_engine else "missing",
            "search": "loaded" if search.engine else "missing"
        }
    }

@app.get("/health")
def health():
    import psutil
    process = psutil.Process(os.getpid())
    return {
        "status": "healthy",
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "uptime": "active"
    }
