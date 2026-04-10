"""
Basket-RAG recommendation router.

Endpoints:
    POST /api/recommendation/basket-rag          — Get recommendations for a cart
    GET  /api/recommendation/basket-rag/health   — Engine health check
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import traceback

router = APIRouter(
    prefix="/api/recommendation",
    tags=["Recommendation Engine"],
)

# Populated by main.py lifespan manager (lazy singleton)
basket_rag_engine = None


# ── Request / Response schemas ───────────────────────────────────────────────

class BasketRAGRequest(BaseModel):
    """Request body for Basket-RAG retrieval-augmented recommendations."""

    cart_product_ids: List[int] = Field(
        ...,
        description="Product IDs currently in the user's cart.",
        min_length=1,
        max_length=60,
        examples=[[24852, 13176, 21137, 196]],
    )
    top_k: Optional[int] = Field(
        default=10,
        ge=1,
        le=50,
        description=(
            "Maximum number of recommendations to return. "
            "The actual count may be lower if Faron's F1 optimisation selects a smaller K."
        ),
    )
    n_retrieve: Optional[int] = Field(
        default=100,
        ge=10,
        le=500,
        description=(
            "Number of similar historical baskets to retrieve from the Faiss index "
            "before scoring. Higher = broader candidate pool, slightly slower."
        ),
    )
    use_f1_cutoff: Optional[bool] = Field(
        default=True,
        description=(
            "Apply Faron's F1-optimisation to determine the ideal recommendation count. "
            "Disable to always return exactly top_k items."
        ),
    )
    recency_boost: Optional[float] = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Small additive boost for globally popular/trending products.",
    )
    popularity_penalty: Optional[float] = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Penalty multiplier for globally popular items to discourage "
            "\"Apple / Avocado\" dominance in recommendations."
        ),
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/basket-rag")
def basket_rag_recommend(request: BasketRAGRequest):
    """
    🛒 Basket-RAG — Retrieval-Augmented Recommendations

    Pipeline:
    1. Encodes the user's current cart into a query vector (BasketEncoder CLS token).
    2. Performs ANN search against all indexed historical basket embeddings (Faiss).
    3. Scores candidate products by weighted similarity + popularity bias.
    4. Returns F1-optimised ranked recommendation list.
    """
    global basket_rag_engine

    if basket_rag_engine is None:
        # Lazy-load fallback (in case lifespan initialisation missed it)
        try:
            from models.basket_rag.engine import BasketRAGEngine
            basket_rag_engine = BasketRAGEngine()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=(
                    "BasketRAG model artifacts not found. "
                    "Train the model on Colab and place the output files in "
                    "ml_backend/data/basket_rag/. "
                    f"Details: {str(e)}"
                ),
            )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"BasketRAGEngine failed to initialise: {str(e)}",
            )

    try:
        result = basket_rag_engine.recommend(
            cart_product_ids=request.cart_product_ids,
            top_k=request.top_k,
            n_retrieve=request.n_retrieve,
            use_f1_cutoff=request.use_f1_cutoff,
            recency_boost=request.recency_boost,
            popularity_penalty=request.popularity_penalty,
        )
        return result

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"BasketRAG recommendation failed: {str(e)}",
        )


@router.get("/basket-rag/health")
def basket_rag_health():
    """Check if the BasketRAG engine is loaded and ready."""
    global basket_rag_engine
    if basket_rag_engine is None:
        return {
            "status":  "not_loaded",
            "model":   "Basket-RAG",
            "message": (
                "Engine not yet initialised. "
                "Train and deploy artifacts, or call /basket-rag to trigger lazy-load."
            ),
        }
    return {
        "status":        "ready",
        "model":         "Basket-RAG",
        "vocab_size":    basket_rag_engine.vocab_size,
        "index_vectors": basket_rag_engine.faiss_index.ntotal,
        "embed_dim":     basket_rag_engine.config.get("embed_dim"),
        "config":        basket_rag_engine.config,
    }
