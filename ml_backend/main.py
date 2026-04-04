from fastapi import FastAPI
from routers import search, recommendation

app = FastAPI(
    title="Instamart ML Models API",
    description="Centralized API for various E-Commerce Machine Learning Models including Search, Recommendations, and Dynamic Pricing.",
    version="1.0.0"
)

# Mount the routers for different sub-systems
app.include_router(search.router)
app.include_router(recommendation.router)

@app.get("/")
def root():
    return {"message": "Welcome to the Instamart ML Backend API", "status": "online"}
