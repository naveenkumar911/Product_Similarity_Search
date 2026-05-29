# Use official Python image
FROM python:3.10-slim

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV NAME=ProductSimilarityApp

# Working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install CPU-only PyTorch first
RUN pip install --no-cache-dir --default-timeout=1000 \
    torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir --default-timeout=1000 \
    -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]