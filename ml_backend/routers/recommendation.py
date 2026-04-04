from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import traceback

router = APIRouter(
    prefix="/api/recommendation",
    tags=["Recommendation Engine"],
)

# Global engine instance (lazy-loaded)
basket_engine = None


class BasketCompleteRequest(BaseModel):
    """Request body for basket completion."""
    cart_product_ids: List[int] = Field(
        ...,
        description="List of product IDs currently in the user's cart",
        min_length=1,
        max_length=50,
        examples=[[24852, 13176, 21137]]
    )
    num_suggestions: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of product suggestions to generate (1-10)"
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.1,
        le=2.0,
        description="Sampling temperature. Lower (0.1-0.5) = conservative/popular items. Higher (0.8-2.0) = diverse/surprising suggestions."
    )
    top_k: Optional[int] = Field(
        default=50,
        ge=1,
        le=200,
        description="Top-K sampling: only sample from top K most likely products"
    )


# Engine is now initialized strictly via FastAPI lifespan in main.py


@router.post("/basket-complete")
def basket_complete(request: BasketCompleteRequest):
    """
    🛒 Basket Completion — Autoregressive Product Generation
    
    Given product IDs in the user's cart, uses BasketGPT (a causal transformer)
    to autoregressively predict complementary products — like GPT generates text,
    but for shopping baskets.
    
    Each predicted product is appended to the context before predicting the next,
    so suggestions are coherent and build upon each other.
    """
    global basket_engine
    
    if basket_engine is None:
        try:
            from models.basket_engine.engine import BasketCompletionEngine
            basket_engine = BasketCompletionEngine()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"BasketGPT model not available. Train and deploy the model first. Error: {str(e)}"
            )
    
    try:
        result = basket_engine.recommend(
            cart_product_ids=request.cart_product_ids,
            num_suggestions=request.num_suggestions,
            temperature=request.temperature,
            top_k=request.top_k,
        )
        return result
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Basket completion failed: {str(e)}"
        )


@router.get("/basket-complete/health")
async def basket_health():
    """Check if the BasketGPT model is loaded and ready."""
    global basket_engine
    if basket_engine is None:
        return {
            "status": "not_loaded",
            "message": "BasketGPT model not loaded. Train and deploy the model first."
        }
    return {
        "status": "ready",
        "model": "BasketGPT",
        "config": basket_engine.config,
        "parameters": basket_engine.model.count_parameters(),
    }
