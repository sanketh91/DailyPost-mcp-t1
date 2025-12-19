"""
Dynamic Tool Registry with Lazy Loading
=======================================
Runtime registry for dynamically generated MCP tools.
"""

import mcp.types as types
import json
import sys
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import os


class DynamicToolRegistry:
    """Registry for dynamically generated tools with lazy loading."""
    
    def __init__(self, storage_path: str = "dynamic_tools_storage"):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tool_handlers: Dict[str, Callable] = {}
        self.storage_path = storage_path
        self._persisted_loaded = False  # Track if we've loaded persisted tools
        self._ensure_storage_dir()
        # Don't load persisted tools yet - defer until first use
    
    def _ensure_storage_dir(self):
        """Create storage directory if it doesn't exist"""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
    
    def _load_persisted_tools_lazy(self):
        """Lazy load persisted tools on first access"""
        if self._persisted_loaded:
            return
        
        self._persisted_loaded = True
        tools_file = os.path.join(self.storage_path, "tools.json")
        
        if not os.path.exists(tools_file):
            return
        
        try:
            print("📂 Loading persisted dynamic tools...", file=sys.stderr)
            with open(tools_file, "r") as f:
                persisted = json.load(f)
                for tool_name, tool_data in persisted.items():
                    self.tools[tool_name] = tool_data["definition"]
                    # Handler will be recreated on first use via fix_missing_handlers
            
            print(f"✅ Loaded {len(self.tools)} persisted tool(s)", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  Warning: Could not load persisted tools: {e}", file=sys.stderr)
    
    def _persist_tools(self):
        """Save tools to disk for persistence across restarts"""
        tools_file = os.path.join(self.storage_path, "tools.json")
        try:
            persisted = {}
            for tool_name, tool_data in self.tools.items():
                persisted[tool_name] = {
                    "definition": tool_data,
                    "created_at": tool_data.get("created_at", datetime.now().isoformat()),
                    "tool_code": tool_data.get("tool_code")
                }
            with open(tools_file, "w") as f:
                json.dump(persisted, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not persist tools: {e}", file=sys.stderr)
    
    def register_tool(
        self,
        tool_name: str,
        tool_definition: types.Tool,
        handler_function: Callable,
        tool_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """Register a new tool at runtime."""
        # Ensure persisted tools are loaded
        self._load_persisted_tools_lazy()
        
        if tool_name in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' already exists"
            }
        
        if handler_function is None or not callable(handler_function):
            return {
                "success": False,
                "error": f"Invalid handler function for tool '{tool_name}'"
            }
        
        tool_data = {
            "name": tool_definition.name,
            "description": tool_definition.description,
            "inputSchema": tool_definition.inputSchema,
            "created_at": datetime.now().isoformat(),
            "usage_count": 0
        }
        
        if tool_code:
            tool_data["tool_code"] = tool_code
        
        self.tools[tool_name] = tool_data
        self.tool_handlers[tool_name] = handler_function
        
        self._persist_tools()
        
        return {
            "success": True,
            "tool_name": tool_name,
            "message": f"Tool '{tool_name}' registered successfully",
            "available_immediately": True,
            "handler_stored": True
        }
    
    def get_tool_definitions(self) -> List[types.Tool]:
        """Get all registered dynamic tools as MCP Tool objects."""
        self._load_persisted_tools_lazy()
        
        dynamic_tools = []
        for tool_name, tool_data in self.tools.items():
            dynamic_tools.append(
                types.Tool(
                    name=tool_name,
                    description=tool_data["description"],
                    inputSchema=tool_data["inputSchema"]
                )
            )
        return dynamic_tools
    
    def get_handler(self, tool_name: str) -> Optional[Callable]:
        """Get the handler function for a tool"""
        self._load_persisted_tools_lazy()
        return self.tool_handlers.get(tool_name)
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered"""
        self._load_persisted_tools_lazy()
        return tool_name in self.tools
    
    def list_tools(self) -> Dict[str, Any]:
        """List all registered dynamic tools"""
        self._load_persisted_tools_lazy()
        
        return {
            "success": True,
            "total_tools": len(self.tools),
            "tools": [
                {
                    "name": name,
                    "description": data["description"],
                    "created_at": data["created_at"],
                    "usage_count": data["usage_count"]
                }
                for name, data in self.tools.items()
            ]
        }
    
    def increment_usage(self, tool_name: str):
        """Increment usage counter for a tool"""
        if tool_name in self.tools:
            self.tools[tool_name]["usage_count"] += 1
            self._persist_tools()
    
    def unregister_tool(self, tool_name: str) -> Dict[str, Any]:
        """Remove a tool from the registry"""
        self._load_persisted_tools_lazy()
        
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        del self.tools[tool_name]
        if tool_name in self.tool_handlers:
            del self.tool_handlers[tool_name]
        
        self._persist_tools()
        
        return {
            "success": True,
            "message": f"Tool '{tool_name}' unregistered"
        }
    
    def fix_missing_handlers(self) -> Dict[str, Any]:
        """Recreate handlers for tools that are missing them"""
        self._load_persisted_tools_lazy()
        
        fixed = []
        failed = []
        
        for tool_name, tool_data in self.tools.items():
            if tool_name not in self.tool_handlers or self.tool_handlers[tool_name] is None:
                tool_code = tool_data.get("tool_code")
                if tool_code:
                    try:
                        from dynamic_tools_framework import DynamicToolGenerator
                        generator = DynamicToolGenerator()
                        handler = generator._create_handler_from_code(tool_code, tool_name)
                        if handler:
                            self.tool_handlers[tool_name] = handler
                            fixed.append(tool_name)
                        else:
                            failed.append(f"{tool_name}: Could not create handler")
                    except Exception as e:
                        failed.append(f"{tool_name}: {str(e)}")
                else:
                    failed.append(f"{tool_name}: No tool code available")
        
        return {
            "success": True,
            "fixed": fixed,
            "failed": failed,
            "message": f"Fixed {len(fixed)} handler(s), {len(failed)} failed"
        }


# Global registry instance
_dynamic_registry = None


def get_dynamic_registry() -> DynamicToolRegistry:
    """Get the global dynamic tool registry (lazy initialization)"""
    global _dynamic_registry
    if _dynamic_registry is None:
        _dynamic_registry = DynamicToolRegistry()
    return _dynamic_registry