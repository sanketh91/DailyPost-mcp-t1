#!/usr/bin/env python3
import sys
import os
from fastmcp import FastMCP

import tool

# Initialize MCP server
mcp = FastMCP("weaviate-dailypost")

# Register tools from tool.py
tool.mcp = mcp
tool.register_tools()

if __name__ == "__main__":
    # Cloud Run sets PORT environment variable
    port = int(os.getenv("PORT", 8000))

    # Check if running in cloud (Cloud Run or any env that sets PORT/K_SERVICE)
    is_cloud = os.getenv("PORT") or os.getenv("K_SERVICE")

    if is_cloud:
        print(f"🚀 Starting MCP Server (HTTP) on 0.0.0.0:{port}...", file=sys.stderr)
        # Run with HTTP (streamable) transport for remote / Cloud Run access
        # Endpoint will be: http://0.0.0.0:{port}/mcp
        mcp.run(
            transport="http",
            host="0.0.0.0",
            port=port,
            path="/mcp",
            log_level="info",
        )
    else:
        print("🚀 Starting MCP Server in STDIO mode (local)...", file=sys.stderr)
        # Default stdio transport for local Claude/Desktop integration
        mcp.run()
