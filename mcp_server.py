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
    
    # Check if running in cloud
    is_cloud = os.getenv("PORT") or os.getenv("K_SERVICE")
    
    if is_cloud:
        print(f"🚀 Starting MCP Server on 0.0.0.0:{port}...", file=sys.stderr)
        # Run with SSE transport for HTTP access
        mcp.run(transport="sse", host="0.0.0.0", port=port, log_level="info")
    else:
        print("🚀 Starting MCP Server in STDIO mode (local)...", file=sys.stderr)
        mcp.run()
