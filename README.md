# Product Similarity Search Service

A **FastAPI** micro‑service that provides enterprise‑grade similarity search over Amazon Fashion product data.  
It leverages **FAISS** for fast nearest‑neighbor indexing and **SentenceTransformers** for semantic text embeddings, combined with brand, image‑id, price, rating and weight heuristics.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running Locally](#running-locally)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Docker & Deployment](#docker--deployment)
- [FAQ & Troubleshooting](#faq--troubleshooting)
- [License](#license)

---

## Features

- **Fast similarity lookup** with FAISS index loaded at startup.
- **Hybrid similarity scoring** – text, brand, image‑overlap, price, rating, weight.
- **Customizable weight parameters** via query strings.
- **Kubernetes‑ready** health‑check endpoints (`/health`).
- **OpenAPI documentation** (`/docs`).

---

## Prerequisites

| Tool | Minimum Version |
|------|------------------|
| Python | 3.10 |
| pip | latest |
| Docker* (optional) | 20.10 |

*Docker is only required for containerised deployment.

---

## Installation

```bash
# Clone the repository (already done in your workspace)
# cd into the project directory
cd c:/Users/503376/NK/SAP_Assignment/sap-cxii-tech-ex-01

# Create a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\activate  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Run the one‑time pre‑computation to generate embeddings and metadata
python precompute.py
```


---

## Running Locally

```bash
# Ensure the FAISS index is built (first run will generate it)
# The pre‑computation step above already creates the embeddings and metadata.
# Start the API
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open your browser at <http://localhost:8000/docs> to explore the interactive Swagger UI.

## Quick Test (optional)

```bash
# Health check
curl http://localhost:8000/health

# Find similar products (example)
curl "http://localhost:8000/find_similar_products?product_id=26d41bdc1495de290bc8e6062d927729&num_similar=5"

# Retrieve a product's details (replace with an ID from the previous result)
curl "http://localhost:8000/products/<product_id>"
```

---

## API Reference

### GET `/`
Returns basic service info and health status.

### GET `/health`
Kubernetes liveness/readiness probe. Returns `{"status":"UP"}` when the index is ready.

### GET `/products/{product_id}`
Fetches detailed metadata for a given product.

### GET `/find_similar_products`
**Parameters** (all via query string):
- `product_id` (required) – target product `uniq_id`.
- `num_similar` (required, 1‑100) – how many results.
- `w_text`, `w_brand`, `w_image`, `w_price`, `w_rating`, `w_weight` – optional weight floats (0‑1). They are normalized automatically.

**Response** – JSON array of product `uniq_id`s ordered by similarity.

---

## Testing

```bash
# Run the provided test suite
pytest -q test_app.py
```
The tests cover endpoint availability, error handling and basic similarity logic.

---

## Docker & Deployment

### Build the image
```bash
docker build -t product-similarity-service .
```
### Run the container
```bash
docker run -p 8000:8000 product-similarity-service
```
### Kubernetes (example snippet)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: similarity-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: similarity-service
  template:
    metadata:
      labels:
        app: similarity-service
    spec:
      containers:
      - name: similarity
        image: product-similarity-service:latest
        ports:
        - containerPort: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          periodSeconds: 30
```

---

## FAQ & Troubleshooting

- **"Index not initialized"** – Ensure `similarity.load_search_index()` runs before the API starts. The startup event is designed to do this automatically.
- **Memory errors** – Reduce the number of indexed rows or increase the container memory limit.
- **Slow response** – Verify that the FAISS index is stored on SSD/NVMe; avoid re‑creating the index on every request.

---

## License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.
