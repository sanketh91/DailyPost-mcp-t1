#!/usr/bin/env python3
import sys
import os
from fastmcp import FastMCP

# SAFEGUARD: Lazy-load tools with error isolation (prevents 500 crashes)
def safe_register_tools(mcp_instance):
    try:
        import tool
        tool.mcp = mcp_instance
        tool.register_tools()
        print("✅ Tools registered successfully", file=sys.stderr)
    except Exception as e:
        print(f"❌ Tool registration failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        # Continue without tools - MCP still works for prompts/resources

# Initialize MCP server WITH STATELESS HTTP (CRITICAL for Cloud Run)
mcp = FastMCP("weaviate-dailypost", stateless_http=True)

# Register tools SAFELY
safe_register_tools(mcp)

if __name__ == "__main__":
    # Cloud Run sets PORT environment variable
    port = int(os.getenv("PORT", 8000))

    # Check if running in cloud (Cloud Run or any env that sets PORT/K_SERVICE)
    is_cloud = os.getenv("PORT") or os.getenv("K_SERVICE")

    if is_cloud:
        print(f"🚀 Starting MCP Server (Streamable HTTP + Stateless) on 0.0.0.0:{port}...", file=sys.stderr)
        mcp.run(
            transport="streamable-http",
            host="0.0.0.0",
            port=port,
            path="/mcp",
            log_level="debug",  # More visibility into FastMCP internals
        )
    else:
        print("🚀 Starting MCP Server in STDIO mode (local)...", file=sys.stderr)
        mcp.run()
