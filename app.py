import time
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from similarity import find_similar_products, df_products, id_to_idx

app = FastAPI(
    title="Product Similarity Search Service",
    description="Enterprise-grade similarity search API for Amazon Fashion Products using FAISS and SentenceTransformers",
    version="1.0.0"
)

# Startup event to ensure index is loaded
@app.on_event("startup")
def startup_event():
    # Calling this forces the module load and indexing if not already loaded
    from similarity import load_search_index
    load_search_index()
    print("FastAPI application started and FAISS index loaded.")

@app.get("/")
def read_root():
    return {
        "service": "Product Similarity Search API",
        "status": "healthy",
        "total_indexed_products": len(df_products) if df_products is not None else 0,
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    """Kubernetes liveness and readiness probe endpoint."""
    if df_products is not None and len(df_products) > 0:
        return {"status": "UP", "timestamp": time.time()}
    raise HTTPException(status_code=503, detail="Search index not initialized")

@app.get("/products/{product_id}", response_model=Dict[str, Any])
def get_product(product_id: str):
    """Retrieve details of a specific product by ID."""
    if df_products is None or product_id not in id_to_idx:
        raise HTTPException(status_code=404, detail=f"Product with ID '{product_id}' not found.")
    
    idx = id_to_idx[product_id]
    row = df_products.iloc[idx]
    
    return {
        "uniq_id": row["uniq_id"],
        "asin": row["asin"] if pd_not_na(row["asin"]) else None,
        "product_name": row["product_name"] if pd_not_na(row["product_name"]) else None,
        "brand": row["brand"] if pd_not_na(row["brand"]) else None,
        "sales_price": row["sales_price_clean"],
        "weight_grams": row["weight_clean"],
        "rating": row["rating_clean"],
        "image_ids": row["image_ids"],
        "product_url": row["product_url"] if pd_not_na(row["product_url"]) else None
    }

@app.get("/find_similar_products", response_model=List[str])
def get_similar_products(
    product_id: str = Query(..., description="The unique ID (uniq_id) of the target product"),
    num_similar: int = Query(..., ge=1, le=100, description="Number of similar products to return (between 1 and 100)"),
    w_text: float = Query(0.40, ge=0.0, le=1.0, description="Weight for text semantic similarity (0.0 to 1.0)"),
    w_brand: float = Query(0.20, ge=0.0, le=1.0, description="Weight for brand similarity (0.0 to 1.0)"),
    w_image: float = Query(0.15, ge=0.0, le=1.0, description="Weight for visual similarity based on image ID overlaps (0.0 to 1.0)"),
    w_price: float = Query(0.15, ge=0.0, le=1.0, description="Weight for sales price similarity (0.0 to 1.0)"),
    w_rating: float = Query(0.05, ge=0.0, le=1.0, description="Weight for rating similarity (0.0 to 1.0)"),
    w_weight: float = Query(0.05, ge=0.0, le=1.0, description="Weight for product weight similarity (0.0 to 1.0)")
):
    """
    Retrieves the top `num_similar` product IDs that are most similar to the given `product_id`.
    Allows customizing the similarity components by passing optional weight parameters.
    """
    # Validate weights sum
    total_weight = w_text + w_brand + w_image + w_price + w_rating + w_weight
    if total_weight <= 0.0:
        raise HTTPException(
            status_code=400, 
            detail="The sum of all similarity weights must be greater than zero."
        )
        
    # Normalize weights so they sum to 1.0
    w_text_norm = w_text / total_weight
    w_brand_norm = w_brand / total_weight
    w_image_norm = w_image / total_weight
    w_price_norm = w_price / total_weight
    w_rating_norm = w_rating / total_weight
    w_weight_norm = w_weight / total_weight
    
    try:
        similar_ids = find_similar_products(
            product_id=product_id,
            num_similar=num_similar,
            w_text=w_text_norm,
            w_brand=w_brand_norm,
            w_image=w_image_norm,
            w_price=w_price_norm,
            w_rating=w_rating_norm,
            w_weight=w_weight_norm
        )
        return similar_ids
        
    except KeyError as ke:
        raise HTTPException(status_code=404, detail=str(ke))
    except Exception as e:
        # Return internal server error with message detail for evaluation clarity
        raise HTTPException(status_code=500, detail=f"Similarity computation error: {str(e)}")

def pd_not_na(val):
    """Helper to check if a value is not null/NaN in pandas."""
    import pandas as pd
    return not pd.isna(val)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
