import os
import re
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Paths
DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(DATA_DIR, "processed_products.pkl")

def parse_weight(weight_val):
    """
    Parses weight strings (e.g. '240 g', '454 Grams', '1.2 kg') and converts them to grams.
    Returns None for invalid or missing values like '999999999'.
    """
    if pd.isna(weight_val):
        return None
    
    weight_str = str(weight_val).strip().lower()
    
    # 999999999 is a common default/missing value placeholder
    if "999999999" in weight_str or not weight_str:
        return None
        
    # Regex to extract numeric value (requiring at least one digit) and optional unit
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)?", weight_str)
    if match:
        try:
            val = float(match.group(1))
            unit = match.group(2)
            if unit:
                if "kg" in unit or "kilogram" in unit:
                    return val * 1000.0
                elif "lb" in unit or "pound" in unit:
                    return val * 453.592
                elif "oz" in unit or "ounce" in unit:
                    return val * 28.3495
            return val
        except ValueError:
            return None
    return None

def extract_image_ids(url_str):
    """
    Extracts base image ID hashes from pipe-separated Amazon image URLs.
    Example: https://images-na.ssl-images-amazon.com/images/I/51Wj2WownyL._SR38,50_.jpg -> 51Wj2WownyL
    """
    if pd.isna(url_str) or not isinstance(url_str, str):
        return []
    
    parts = url_str.split("|")
    image_ids = []
    for part in parts:
        part = part.strip()
        if part:
            # Get the file portion of the path, split by dots or underscores
            filename = part.split("/")[-1]
            img_id = filename.split(".")[0].split("_")[0]
            if img_id:
                image_ids.append(img_id)
    return image_ids

def main():
    print("Starting data precomputation...")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}. Make sure it is extracted.")
    
    # Load dataset
    print(f"Loading raw dataset from {RAW_DATA_PATH}...")
    df = pd.read_json(RAW_DATA_PATH, lines=True)
    print(f"Loaded {len(df)} records.")
    
    # Drop rows without uniq_id
    df = df.dropna(subset=["uniq_id"])
    print(f"Kept {len(df)} records after dropping rows without uniq_id.")
    
    # Preprocess brand
    df["brand"] = df["brand"].fillna("").astype(str).str.strip()
    
    # Preprocess sales_price
    median_price = df["sales_price"].median()
    if pd.isna(median_price):
        median_price = 500.0  # fallback fallback
    df["sales_price_clean"] = df["sales_price"].fillna(median_price).astype(float)
    
    # Preprocess rating
    median_rating = df["rating"].median()
    if pd.isna(median_rating):
        median_rating = 4.0
    df["rating_clean"] = df["rating"].fillna(median_rating).astype(float)
    
    # Preprocess weight
    df["weight_parsed"] = df["weight"].apply(parse_weight)
    median_weight = df["weight_parsed"].median()
    if pd.isna(median_weight):
        median_weight = 250.0  # fallback
    df["weight_clean"] = df["weight_parsed"].fillna(median_weight).astype(float)
    
    # Preprocess image IDs (using small/medium/large URLs fallback)
    df["image_urls"] = df["image_urls__small"].fillna(df["medium"]).fillna(df["large"])
    df["image_ids"] = df["image_urls"].apply(extract_image_ids)
    
    # Create concatenated text representation for embedding
    print("Constructing text representation for semantic search...")
    texts = []
    for _, row in df.iterrows():
        parts = []
        name = str(row.get("product_name", "")).strip()
        if name and name.lower() != "nan":
            parts.append(name)
        brand = str(row.get("brand", "")).strip()
        if brand and brand.lower() != "nan":
            parts.append(brand)
        keywords = str(row.get("meta_keywords", "")).strip()
        if keywords and keywords.lower() != "nan":
            parts.append(keywords)
        texts.append(" ".join(parts))
        
    df["text_repr"] = texts
    
    # Generate Sentence Embeddings
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Generating sentence embeddings (this may take a few minutes)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=128, convert_to_numpy=True)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Save embeddings
    print(f"Saving embeddings to {EMBEDDINGS_PATH}...")
    np.save(EMBEDDINGS_PATH, embeddings)
    
    # Save clean metadata
    print(f"Saving metadata to {METADATA_PATH}...")
    clean_cols = [
        "uniq_id", "asin", "product_name", "brand", "sales_price_clean", 
        "weight_clean", "rating_clean", "image_ids", "product_url"
    ]
    df_clean = df[clean_cols]
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(df_clean, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Precomputation completed successfully!")

if __name__ == "__main__":
    main()
