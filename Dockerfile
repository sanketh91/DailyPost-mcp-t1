# Use official Python runtime
FROM python:3.11-slim

# Install uv for faster dependency installation (optional but recommended)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all Python files
COPY mcp_server.py .
COPY tool.py .
COPY dynamic_tools_framework.py .
COPY dynamic_tool_registry.py .

# Create directory for dynamic tools storage
RUN mkdir -p /app/dynamic_tools_storage

# Cloud Run expects the app to listen on $PORT
ENV PYTHONUNBUFFERED=1

# Start the server
CMD ["python", "mcp_server.py"]
