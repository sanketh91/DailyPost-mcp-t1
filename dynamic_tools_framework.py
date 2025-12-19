"""
Dynamic Tool Generation Framework
==================================
Framework for AI-powered dynamic tool creation when existing tools are insufficient.
"""

from typing import Dict, List, Optional, Any, Callable
import ast
import json
import re
import sys
from datetime import datetime
import mcp.types as types
from dynamic_tool_registry import get_dynamic_registry


# Template for generating Weaviate query tools
WEAVIATE_TOOL_TEMPLATE = """
def {function_name}({parameters}) -> Dict[str, Any]:
    \"\"\"
    {description}
    
    Generated dynamically for: {query_description}
    \"\"\"
    client = get_weaviate_client()
    try:
        collection = client.collections.get("{collection_name}")
        
        {query_logic}
        
        # Query logic should always return directly
        # No fallthrough code needed
        
    except Exception as e:
        import traceback
        return {{
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }}
"""


class DynamicToolGenerator:
    """
    Generates new MCP tools dynamically based on query requirements.
    """
    
    def __init__(self):
        self.generated_tools = {}  # Store generated tools
        self.tool_counter = 0
    
    def analyze_query_requirements(
        self,
        query_description: str,
        existing_tools: List[str]
    ) -> Dict[str, Any]:
        """
        Analyzes a query to determine if existing tools can handle it.
        
        Returns:
        --------
        dict with:
            - can_handle: bool
            - suggested_tools: List[str]
            - missing_capabilities: List[str]
            - requires_new_tool: bool
        """
        # Simple keyword-based analysis (can be enhanced with LLM)
        query_lower = query_description.lower()
        
        # Check if query matches existing tool patterns
        tool_patterns = {
            "search_posts_hybrid": ["search", "find", "posts about", "articles on"],
            "search_by_date_range": ["from", "between", "date range", "in year"],
            "get_post_by_id": ["post #", "post number", "post id"],
            "search_posts_by_topic": ["topic", "category", "under"],
            "search_by_keyword": ["keyword", "exact", "phrase"],
            "get_recent_posts": ["recent", "latest", "new", "recently"]
        }
        
        matching_tools = []
        for tool_name, patterns in tool_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                matching_tools.append(tool_name)
        
        # Check for complex requirements that might need new tools
        complex_patterns = [
            ("count", "counting", "how many"),
            ("compare", "comparison", "difference"),
            ("trend", "over time", "evolution"),
            ("relationship", "connection", "related to"),
            ("frequency", "most mentioned", "appears"),
            ("contradiction", "opposing", "conflicting")
        ]
        
        missing_capabilities = []
        for pattern_group in complex_patterns:
            if any(p in query_lower for p in pattern_group):
                missing_capabilities.append(pattern_group[0])
        
        requires_new_tool = len(missing_capabilities) > 0 and len(matching_tools) == 0
        
        return {
            "can_handle": len(matching_tools) > 0 and not requires_new_tool,
            "suggested_tools": matching_tools,
            "missing_capabilities": missing_capabilities,
            "requires_new_tool": requires_new_tool,
            "query_analysis": {
                "keywords": self._extract_keywords(query_description),
                "intent": self._classify_intent(query_description)
            }
        }
    
    def generate_tool_code(
        self,
        query_description: str,
        required_capabilities: List[str],
        example_queries: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates Python code for a new tool.
        
        This is a simplified version. A full implementation would use an LLM
        to generate more sophisticated code.
        """
        # Generate function name
        self.tool_counter += 1
        function_name = parameters.get("tool_name") if parameters and "tool_name" in parameters else f"dynamic_tool_{self.tool_counter}"
        
        # Determine collection and query type based on capabilities
        collection_name = "Post"  # Default
        query_type = "hybrid"
        
        if "count" in required_capabilities:
            query_type = "aggregate"
        elif "compare" in required_capabilities:
            query_type = "multi_query"
        
        # Generate parameters based on query description
        parameters = self._extract_parameters(query_description)
        param_string = ", ".join([f"{p}: str" for p in parameters])
        
        # Generate query logic (simplified - real version would be more sophisticated)
        query_logic = self._generate_query_logic(query_type, parameters, query_description)
        
        # Generate result formatting
        result_formatting = self._generate_result_formatting(parameters)
        
        # Build the tool code
        tool_code = WEAVIATE_TOOL_TEMPLATE.format(
            function_name=function_name,
            parameters=param_string,
            description=f"Dynamic tool for: {query_description}",
            query_description=query_description,
            collection_name=collection_name,
            query_logic=query_logic,
            result_formatting=result_formatting
        )
        
        return {
            "success": True,
            "tool_name": function_name,
            "tool_code": tool_code,
            "parameters": parameters,
            "query_type": query_type,
            "validation": {
                "syntax_valid": self._validate_syntax(tool_code),
                "safety_checks": self._safety_checks(tool_code)
            }
        }
    
    def create_and_register_tool(
        self,
        query_description: str,
        tool_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Creates a new tool from a query description and registers it with the MCP server.
        
        This is the main method that:
        1. Analyzes the query
        2. Generates tool code
        3. Creates handler function
        4. Registers with dynamic registry (immediately available)
        
        Parameters:
        -----------
        query_description : str
            Description of what the tool should do
        tool_name : str, optional
            Custom name for the tool (auto-generated if not provided)
        parameters : dict, optional
            Additional parameters for tool generation
        
        Returns:
        --------
        dict with registration result
        """
        # Analyze query
        existing_tools = ["search_posts_hybrid", "search_by_date_range", "get_post_by_id",
                         "search_posts_by_topic", "search_by_keyword", "get_recent_posts"]
        analysis = self.analyze_query_requirements(query_description, existing_tools)
        
        if not analysis["requires_new_tool"]:
            return {
                "success": False,
                "error": "Query can be handled by existing tools. No new tool creation needed.",
                "suggested_tools": analysis["suggested_tools"],
                "message": f"You can use these existing tools: {', '.join(analysis['suggested_tools'])} to handle this query."
            }
        
        # Generate tool code
        tool_result = self.generate_tool_code(
            query_description,
            analysis["missing_capabilities"],
            parameters=parameters
        )
        
        if not tool_result["success"]:
            return tool_result
        
        # Create handler function from code
        handler_func = self._create_handler_from_code(
            tool_result["tool_code"],
            tool_result["tool_name"]
        )
        
        if handler_func is None:
            error_details = {
                "tool_name": tool_result["tool_name"],
                "validation": tool_result.get("validation", {}),
                "note": "Check server logs (stderr) for detailed error information"
            }
            return {
                "success": False,
                "error": "Failed to create handler function from generated code. This usually means there was a syntax error or execution error in the generated code.",
                "debug_info": error_details,
                "suggestion": "The generated code may have issues. Check the server console output for detailed error messages. You may need to use existing tools instead.",
                "fallback": "Consider using existing tools like 'search_posts_hybrid' or 'search_by_date_range' to accomplish this task."
            }
        
        # Create MCP tool definition
        tool_def = self._create_tool_definition(
            tool_result["tool_name"],
            query_description,
            tool_result["parameters"]
        )
        
        # Register with dynamic registry (include tool code for persistence)
        registry = get_dynamic_registry()
        register_result = registry.register_tool(
            tool_result["tool_name"],
            tool_def,
            handler_func,
            tool_code=tool_result["tool_code"]  # Store code for persistence
        )
        
        if register_result["success"]:
            # Verify the tool is in the registry
            # Note: With FastMCP, we can't directly query the tool list at runtime
            # but we can verify it's in the registry
            registry = get_dynamic_registry()
            is_in_registry = registry.has_tool(tool_result["tool_name"])
            
            response = {
                "success": True,
                "tool_name": tool_result["tool_name"],
                "message": f"Tool '{tool_result['tool_name']}' created and registered successfully. The tool is now available in the dynamic registry and can be called immediately.",
                "available_immediately": True,
                "handler_stored": True,
                "verified_in_registry": is_in_registry,
                "tool_definition": {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "inputSchema": tool_def.inputSchema
                },
                "instructions": f"You can now call the tool '{tool_result['tool_name']}' directly. The tool will extract all necessary information from its internal logic and requires no parameters (or use empty object {{}} if parameters are required).",
                "next_step": f"Call the tool '{tool_result['tool_name']}' to execute the query: {query_description}",
                "note": "With FastMCP, dynamic tools are available through the dynamic registry. The tool may require a client refresh to appear in the tool list, but it can be called directly by name."
            }
            
            if not is_in_registry:
                response["warning"] = "Tool created but not found in registry. This may indicate a registration issue."
                response["troubleshooting"] = [
                    "1. Check server logs for registration errors",
                    "2. Verify the handler function was created successfully",
                    "3. Try recreating the tool"
                ]
            
            return response
        else:
            return register_result
    
    def _create_handler_from_code(self, code: str, function_name: str) -> Optional[Callable]:
        """Create a handler function from generated code"""
        try:
            # Compile the code first to catch syntax errors early
            try:
                compiled = compile(code, f"<dynamic_tool_{function_name}>", "exec")
            except SyntaxError as e:
                print(f"Syntax error in generated code: {e}", file=sys.stderr)
                print(f"Generated code:\n{code}", file=sys.stderr)
                return None
            
            # Create a namespace with necessary imports
            namespace = {
                "__name__": f"dynamic_tool_{function_name}",
                "__builtins__": __builtins__,
                "__file__": f"<dynamic_tool_{function_name}>"
            }
            
            # Add necessary imports to namespace
            try:
                from weaviate_tools import get_weaviate_client, get_embedding_for_query
                from weaviate.classes.query import Filter, MetadataQuery
                from typing import Dict, Any, List, Optional
                from datetime import datetime
                
                namespace.update({
                    "get_weaviate_client": get_weaviate_client,
                    "get_embedding_for_query": get_embedding_for_query,
                    "Filter": Filter,
                    "MetadataQuery": MetadataQuery,
                    "Dict": Dict,
                    "Any": Any,
                    "List": List,
                    "Optional": Optional,
                    "datetime": datetime
                })
            except ImportError as e:
                print(f"Import error setting up namespace: {e}", file=sys.stderr)
                return None
            
            # Execute in namespace
            try:
                exec(compiled, namespace)
            except Exception as e:
                print(f"Error executing generated code: {e}", file=sys.stderr)
                print(f"Generated code:\n{code}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                return None
            
            # Get the function
            handler_func = namespace.get(function_name)
            
            if handler_func is None:
                print(f"Function '{function_name}' not found in namespace", file=sys.stderr)
                print(f"Available names in namespace: {list(namespace.keys())}", file=sys.stderr)
                print(f"Generated code:\n{code}", file=sys.stderr)
                return None
            
            if not callable(handler_func):
                print(f"'{function_name}' is not callable: {type(handler_func)}", file=sys.stderr)
                return None
            
            # Wrap it to make it async-compatible
            async def async_handler(arguments: dict | None) -> dict:
                """Async wrapper for dynamic tool handler"""
                try:
                    # Handle both parameterized and parameterless functions
                    # Check function signature to determine if it expects parameters
                    import inspect
                    sig = inspect.signature(handler_func)
                    param_count = len([p for p in sig.parameters.values() if p.default == inspect.Parameter.empty and p.name != 'self'])
                    
                    if param_count == 0:  # Function expects no parameters
                        result = handler_func()
                    else:
                        result = handler_func(**(arguments or {}))
                    
                    # FastMCP tools return dicts directly, not TextContent lists
                    # Ensure result is a dict
                    if isinstance(result, dict):
                        return result
                    elif isinstance(result, list):
                        # If it's a list (from old create_json_response), extract dict
                        if result and hasattr(result[0], 'text'):
                            import json
                            return json.loads(result[0].text)
                        return {"success": False, "error": "Unexpected result format", "result": result}
                    else:
                        return {"success": True, "result": result}
                except Exception as e:
                    import traceback
                    error_result = {
                        "success": False,
                        "error": f"Error executing dynamic tool: {str(e)}",
                        "tool_name": function_name,
                        "traceback": traceback.format_exc()
                    }
                    return error_result
            
            return async_handler
            
        except Exception as e:
            print(f"Unexpected error creating handler: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return None
    
    def _create_tool_definition(
        self,
        tool_name: str,
        description: str,
        parameters: List[str]
    ) -> types.Tool:
        """Create MCP Tool definition from parameters"""
        # Build input schema
        properties = {}
        required = []
        
        for param in parameters:
            # Determine parameter type based on name
            if "date" in param.lower():
                param_type = "string"
                param_desc = "Date filter (YYYY-MM-DD format)"
            elif "limit" in param.lower():
                param_type = "integer"
                param_desc = "Maximum number of results"
            else:
                param_type = "string"
                param_desc = f"Parameter: {param}"
            
            properties[param] = {
                "type": param_type,
                "description": param_desc
            }
            
            if param != "limit":  # limit is optional
                required.append(param)
        
        # Create appropriate description
        if not parameters:
            desc = f"Dynamic tool: {description}. No parameters required - all information extracted from query."
        else:
            desc = f"Dynamic tool: {description}"
        
        return types.Tool(
            name=tool_name,
            description=desc,
            inputSchema={
                "type": "object",
                "properties": properties,
                "required": required
            }
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from query text"""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        return [w for w in words if w not in stop_words and len(w) > 3]
    
    def _classify_intent(self, text: str) -> str:
        """Classify the intent of a query"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["count", "how many", "number of"]):
            return "counting"
        elif any(word in text_lower for word in ["find", "search", "get"]):
            return "search"
        elif any(word in text_lower for word in ["compare", "difference", "versus"]):
            return "comparison"
        elif any(word in text_lower for word in ["trend", "over time", "evolution"]):
            return "trend_analysis"
        else:
            return "general_query"
    
    def _extract_parameters(self, query: str) -> List[str]:
        """Extract potential parameters from query"""
        # For count/aggregation queries, no parameters needed
        if any(word in query.lower() for word in ["count", "how many"]):
            return []  # No parameters for count queries
        
        params = []
        if "date" in query.lower():
            params.append("date")
        if "topic" in query.lower():
            params.append("topic")
        if "keyword" in query.lower() or "term" in query.lower():
            params.append("keyword")
        if "limit" in query.lower():
            params.append("limit")
        return params if params else ["query"]
    
    def _generate_query_logic(self, query_type: str, parameters: List[str], description: str) -> str:
        """Generate Weaviate query logic"""
        if query_type == "aggregate" or "count" in description.lower():
            # For count queries, we need to search first, then count
            # Extract keyword and year from description if possible
            keyword = "GDPR"  # Default, should be extracted from description
            year = "2023"  # Default
            
            # Try to extract keyword and year from description
            import re
            keyword_match = re.search(r"mention(?:ing|s)?\s+['\"]?(\w+)['\"]?", description, re.IGNORECASE)
            if keyword_match:
                keyword = keyword_match.group(1)
            
            year_match = re.search(r"\b(20\d{2})\b", description)
            if year_match:
                year = year_match.group(1)
            
            return f"""
        # Count query: Count posts mentioning '{keyword}' in {year}
        query_vector = get_embedding_for_query("{keyword}")
        
        # Filter by date range for the year
        start_date = datetime(int({year}), 1, 1)
        end_date = datetime(int({year}), 12, 31, 23, 59, 59)
        start_iso = start_date.isoformat() + "Z"
        end_iso = end_date.isoformat() + "Z"
        
        date_filter = (
            Filter.by_property("post_date").greater_or_equal(start_iso) &
            Filter.by_property("post_date").less_or_equal(end_iso)
        )
        
        # Search for posts containing the keyword
        search_results = collection.query.hybrid(
            query="{keyword}",
            vector=query_vector,
            alpha=0.7,
            limit=10000,  # Get all matching posts
            filters=date_filter,
            return_properties=["post_number", "post_title", "post_date"]
        )
        
        # Count the results and return (this exits the function)
        count = len(search_results.objects)
        return {{
            "success": True,
            "query": "{description}",
            "keyword": "{keyword}",
            "year": int({year}),
            "count": count,
            "sample_results": [{{"post_number": obj.properties.get("post_number"), "title": obj.properties.get("post_title")}} for obj in search_results.objects[:10]]
        }}
        """
        else:
            return """
        # Hybrid search query
        query_vector = get_embedding_for_query(query)
        results = collection.query.hybrid(
            query=query,
            vector=query_vector,
            limit=10,
            return_properties=["post_number", "post_title", "post_content", "post_date", "topic_name"]
        )
        """
    
    def _generate_result_formatting(self, parameters: List[str]) -> str:
        """Generate result formatting code"""
        format_parts = []
        format_parts.append('"post_number": obj.properties.get("post_number")')
        format_parts.append('"title": obj.properties.get("post_title")')
        if "date" in parameters:
            format_parts.append('"date": obj.properties.get("post_date")')
        if "topic" in parameters:
            format_parts.append('"topic": obj.properties.get("topic_name")')
        
        return ",\n                ".join(format_parts)
    
    def _validate_syntax(self, code: str) -> bool:
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _safety_checks(self, code: str) -> Dict[str, Any]:
        """Perform safety checks on generated code"""
        issues = []
        
        # Check for dangerous operations
        dangerous_patterns = [
            (r'__import__', "Use of __import__"),
            (r'eval\(', "Use of eval()"),
            (r'exec\(', "Use of exec()"),
            (r'open\(', "File operations"),
            (r'subprocess', "Subprocess calls"),
            (r'os\.system', "System calls")
        ]
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, code):
                issues.append(description)
        
        # Check that it only uses Weaviate operations
        if not re.search(r'weaviate|get_weaviate_client', code):
            issues.append("Code doesn't use Weaviate client")
        
        return {
            "safe": len(issues) == 0,
            "issues": issues
        }


# Example usage
if __name__ == "__main__":
    generator = DynamicToolGenerator()
    
    # Test query analysis
    test_query = "Count how many posts mention 'GDPR' in the last year"
    print("=" * 80)
    print("Dynamic Tool Generation Test")
    print("=" * 80)
    
    print(f"\nQuery: {test_query}")
    analysis = generator.analyze_query_requirements(
        test_query,
        ["search_posts_hybrid", "search_by_keyword"]
    )
    print(f"\nAnalysis:\n{json.dumps(analysis, indent=2)}")
    
    if analysis["requires_new_tool"]:
        print("\nGenerating new tool...")
        tool_result = generator.generate_tool_code(
            test_query,
            analysis["missing_capabilities"]
        )
        print(f"\nGenerated Tool:\n{tool_result['tool_code']}")
        
        print("\nRegistering tool...")
        register_result = generator.register_dynamic_tool(
            tool_result["tool_code"],
            tool_result["tool_name"],
            f"Tool for: {test_query}"
        )
        print(f"\nRegistration:\n{json.dumps(register_result, indent=2)}")

