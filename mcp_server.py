#!/usr/bin/env python3
import sys
import os
from fastmcp import FastMCP

mcp = FastMCP("weaviate-dailypost")

import tool
tool.mcp = mcp
tool.register_tools()

if __name__ == "__main__":
    # Cloud Run sets PORT environment variable
    port = int(os.getenv("PORT", 8000))
    
    # Check if running in cloud (Cloud Run, Railway, or any container)
    is_cloud = os.getenv("PORT") or os.getenv("K_SERVICE")  # K_SERVICE is Cloud Run specific
    
    if is_cloud:
        print(f"🚀 Starting MCP Server in SSE mode on port {port}...", file=sys.stderr)
        mcp.run(transport="sse", host="0.0.0.0", port=port)
    else:
        print("🚀 Starting MCP Server in STDIO mode (local)...", file=sys.stderr)
        mcp.run()
