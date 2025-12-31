# Runtime image (fast rebuilds)
FROM gcr.io/<PROJECT_ID>/dailypost-mcp-base:latest

WORKDIR /app

# Only your code changes → only this layer rebuilds
COPY mcp_server.py .
COPY tool.py .
COPY dynamic_tools_framework.py .
COPY dynamic_tool_registry.py .
RUN mkdir -p app/dynamictoolsstorage

EXPOSE 8080
CMD ["python", "mcp_server.py"]
