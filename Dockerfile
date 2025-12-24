FROM python:3.11-slim
# Use official Python runtime

RUN apt-get update && apt-get install -y build-essential python3-dev && rm -rf /var/lib/apt/lists/*
# Install build dependencies for packages like hdbscan

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
# Install uv for faster dependency installation (optional but recommended)

WORKDIR /app
# Set working directory

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy requirements first for Docker layer caching

# ============================================================================
# CRITICAL: Pre-download sentence-transformers model to avoid runtime timeout
# ============================================================================
ENV TRANSFORMERS_CACHE=/app/.cache
RUN mkdir -p /app/.cache && \
    python -c "from sentence_transformers import SentenceTransformer; \
    print('Downloading sentence-transformers model...'); \
    model = SentenceTransformer('all-mpnet-base-v2'); \
    print(f'Model downloaded to {model.get_sentence_embedding_dimension()} dimensions')"
# This downloads the ~200MB model once at build time, not at runtime

# ============================================================================

COPY mcp_server.py .
COPY tool.py .
COPY dynamic_tools_framework.py .
COPY dynamic_tool_registry.py .
# Copy all Python files

RUN mkdir -p /app/dynamictools_storage
# Create directory for dynamic tools storage

ENV PYTHONUNBUFFERED=1
# Cloud Run expects the app to listen on PORT

CMD ["python", "mcp_server.py"]
# Start the server
