# STAGE 1: Dependencies (Stable, cacheable)
FROM python:3.11-slim AS deps
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (changes → ONLY this layer rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# STAGE 2: Model Cache (Stable, rarely changes)
FROM deps AS model-cache
RUN pip install sentence-transformers torch
ENV TRANSFORMERS_CACHE=/app.cache
RUN mkdir -p /app.cache && \
    python -c "
import os
from sentence_transformers import SentenceTransformer
print('Downloading sentence-transformers model...')
model = SentenceTransformer('all-mpnet-base-v2')
model.save_pretrained('/app.cache/all-mpnet-base-v2')
print('Model cached:', model.get_sentence_embedding_dimension(), 'dimensions')
"

# STAGE 3: Runtime (ONLY your code rebuilds on changes)
FROM python:3.11-slim AS runtime
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY deps + models (stable layers)
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY --from=model-cache /app.cache /app.cache

# Copy ONLY your code (changes → ONLY this layer rebuilds)
WORKDIR /app
COPY mcp_server.py .
COPY tool.py .
COPY dynamictoolsframework.py .
COPY dynamictoolregistry.py .
RUN mkdir -p app/dynamictoolsstorage

# Runtime config
ENV TRANSFORMERS_CACHE=/app.cache
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080
CMD ["python", "mcp_server.py"]
