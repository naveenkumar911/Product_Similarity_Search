import os
import pickle
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Any

# File Paths
DATA_DIR = "data"
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(DATA_DIR, "processed_products.pkl")

# Global variables for loaded data
df_products: pd.DataFrame = None
embeddings: np.ndarray = None
faiss_index: faiss.IndexFlatIP = None
id_to_idx: Dict[str, int] = {}

def load_search_index():
    """
    Loads precomputed metadata and embeddings, then constructs the FAISS index.
    Called on module load or explicitly during app initialization.
    """
    global df_products, embeddings, faiss_index, id_to_idx
    
    if df_products is not None:
        return  # Already initialized
        
    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Precomputed data not found. Please run 'precompute.py' first to generate files."
        )
        
    print("Loading precomputed data...")
    # Load metadata
    with open(METADATA_PATH, "rb") as f:
        df_products = pickle.load(f)
        
    # Reset index to ensure direct row alignment with embeddings
    df_products = df_products.reset_index(drop=True)
    
    # Load embeddings
    embeddings = np.load(EMBEDDINGS_PATH)
    
    # Map product_id (uniq_id) to index position
    id_to_idx = {row["uniq_id"]: idx for idx, row in df_products.iterrows()}
    
    # Normalize embeddings for Cosine Similarity using Inner Product index
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    normalized_embeddings = embeddings / norms
    
    # Build FAISS Flat Inner Product Index
    dim = normalized_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(normalized_embeddings)
    print(f"FAISS index loaded successfully with {faiss_index.ntotal} vectors.")

# Initialize at startup
load_search_index()

def calculate_jaccard_similarity(list_a: List[Any], list_b: List[Any]) -> float:
    """Computes Jaccard Similarity between two lists of elements."""
    set_a = set(list_a)
    set_b = set(list_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

def find_similar_products(
    product_id: str, 
    num_similar: int,
    w_text: float = 0.40,
    w_brand: float = 0.20,
    w_image: float = 0.15,
    w_price: float = 0.15,
    w_rating: float = 0.05,
    w_weight: float = 0.05
) -> List[str]:
    """
    Finds the top `num_similar` product IDs that are most similar to `product_id`
    using a two-stage retrieval & reranking process.
    
    Parameters:
    - product_id: Unique ID of the query product.
    - num_similar: Number of similar products to return.
    - w_text: Weight for text semantic similarity.
    - w_brand: Weight for brand similarity.
    - w_image: Weight for visual (image ID similarity).
    - w_price: Weight for price difference.
    - w_rating: Weight for rating difference.
    - w_weight: Weight for product weight similarity.
    """
    # Ensure indices are initialized
    load_search_index()
    
    # 1. Fetch query product
    if product_id not in id_to_idx:
        raise KeyError(f"Product ID '{product_id}' not found in the dataset.")
        
    query_idx = id_to_idx[product_id]
    query_row = df_products.iloc[query_idx]
    
    # Get query embedding and normalize it
    query_emb = embeddings[query_idx].reshape(1, -1)
    norm = np.linalg.norm(query_emb)
    if norm > 0:
        query_emb_norm = query_emb / norm
    else:
        query_emb_norm = query_emb
        
    # 2. Stage 1: Retrieve top candidates using FAISS
    # We retrieve K candidates (e.g. 500) to ensure we have enough relevant candidates
    # after filtering out the query product itself.
    retrieval_k = min(500, len(df_products))
    similarities, indices = faiss_index.search(query_emb_norm, retrieval_k)
    
    candidate_indices = indices[0]
    candidate_text_sims = similarities[0]
    
    # 3. Stage 2: Compute hybrid scores and rerank
    candidates = []
    
    # Query product attributes
    q_brand = query_row["brand"]
    q_price = query_row["sales_price_clean"]
    q_weight = query_row["weight_clean"]
    q_rating = query_row["rating_clean"]
    q_images = query_row["image_ids"]
    
    # Scales for exponential decay
    scale_price = 500.0   # $500 diff results in exp(-1) ~ 0.37
    scale_weight = 1000.0 # 1000g diff results in exp(-1) ~ 0.37
    
    for idx, text_sim in zip(candidate_indices, candidate_text_sims):
        # Retrieve candidate product
        cand_id = df_products.iloc[idx]["uniq_id"]
        
        # Exclude the query product itself
        if cand_id == product_id:
            continue
            
        cand_row = df_products.iloc[idx]
        
        # Brand similarity (1.0 if identical non-empty, else 0.0)
        c_brand = cand_row["brand"]
        brand_sim = 1.0 if q_brand and c_brand and q_brand.lower() == c_brand.lower() else 0.0
        
        # Image link similarity (Jaccard overlap of image IDs)
        image_sim = calculate_jaccard_similarity(q_images, cand_row["image_ids"])
        
        # Price similarity (Exponential decay)
        c_price = cand_row["sales_price_clean"]
        price_sim = np.exp(-abs(q_price - c_price) / scale_price)
        
        # Rating similarity (Proximity out of 4 stars difference)
        c_rating = cand_row["rating_clean"]
        rating_sim = 1.0 - (abs(q_rating - c_rating) / 4.0)
        
        # Weight similarity (Exponential decay)
        c_weight = cand_row["weight_clean"]
        weight_sim = np.exp(-abs(q_weight - c_weight) / scale_weight)
        
        # Calculate combined hybrid score
        hybrid_score = (
            w_text * text_sim +
            w_brand * brand_sim +
            w_image * image_sim +
            w_price * price_sim +
            w_rating * rating_sim +
            w_weight * weight_sim
        )
        
        candidates.append({
            "uniq_id": cand_id,
            "hybrid_score": float(hybrid_score),
            "rating": c_rating,
            "price": c_price
        })
        
    # Convert to DataFrame to perform structured multi-key tie-breaker sorting
    df_cand = pd.DataFrame(candidates)
    
    if df_cand.empty:
        return []
        
    # Sort hierarchy:
    # 1. hybrid_score descending
    # 2. rating descending (tie-breaker 1)
    # 3. price ascending (tie-breaker 2: cheaper is better)
    # 4. uniq_id ascending (ensures determinism)
    df_cand = df_cand.sort_values(
        by=["hybrid_score", "rating", "price", "uniq_id"],
        ascending=[False, False, True, True]
    )
    
    # Return top num_similar product IDs
    return df_cand["uniq_id"].head(num_similar).tolist()
