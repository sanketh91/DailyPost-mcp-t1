# Weaviate MCP Tools for Blog Post Database
# Version 2.0.3 - Complete reconnection logic with query-level error handling
# Changes:
#   - Fixed printf → print throughout
#   - Reset weaviateconnectiontested on failed initial connection
#   - Added invalidate_client() helper for query-level errors
#   - Shorter health check interval (60 seconds)
#   - Query-level retry logic
#   - Central client invalidation mechanism

import json
import uuid
import sys
import time
import os
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable

import weaviate
from weaviate.classes.backup import BackupStorage
from sentencetransformers import SentenceTransformer
from weaviate.classes.query import Filter, MetadataQuery, Sort

from dynamictoolregistry import getdynamicregistry
from dynamictoolsframework import DynamicToolGenerator

# ============================================================================
# CONSTANTS AND GLOBALS
# ============================================================================

TITLE = "Weaviate MCP Tools"
STYLEGUIDEPATH = ".sanjaysahaystyle.txt"
MINPOSTSFORPATTERN = 5

mcp = None
EMBEDDINGMODEL = None
MODELDIMENSION = 768
modelloadattempted = False

weaviateclient = None
weaviateconnectiontested = False
weaviatelasthealthcheck = 0.0
WEAVIATEHEALTHCHECKINTERVALSECONDS = 60  # 1 minute (shorter for faster recovery)
WEAVIATEMAXRETRIES = 3
WEAVIATERETRYBACKOFFSECONDS = 2.0

# Retry tracking for connection failures
WEAVIATE_CONSECUTIVE_FAILURES = 0
WEAVIATE_MAX_CONSECUTIVE_FAILURES = 5

weaviateclientlock = threading.Lock()


def json_safe_default(obj):
    """Custom JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def register_tools():
    """Register all tools with the mcp instance."""
    if mcp is None:
        raise RuntimeError("mcp instance not set. Cannot register tools.")
    
    print("Registering MCP tools...", file=sys.stderr)
    
    # Register static tools
    mcp.tool(search_posts_hybrid)
    mcp.tool(search_by_daterange)
    mcp.tool(get_post_by_id)
    mcp.tool(get_posts_batch)
    mcp.tool(search_posts_by_topic)
    mcp.tool(get_topic_statistics)
    mcp.tool(find_similar_posts)
    mcp.tool(search_by_keyword)
    mcp.tool(list_all_topics)
    mcp.tool(get_recent_posts)
    mcp.tool(aggregate_posts)
    mcp.tool(search_chunks)
    mcp.tool(get_posts_for_daily)
    mcp.tool(add_writing_pattern)
    mcp.tool(get_style_guide)
    mcp.tool(create_dynamic_tool)
    mcp.tool(list_dynamic_tools)
    mcp.tool(insert_object)
    mcp.tool(update_object)
    mcp.tool(delete_object)
    mcp.tool(backup_weaviate)
    mcp.tool(restore_backup)
    mcp.tool(export_all_data)
    
    print("Static tools registered successfully", file=sys.stderr)


def load_embedding_model():
    """Lazy load the embedding model on first use."""
    global EMBEDDINGMODEL, modelloadattempted
    
    if EMBEDDINGMODEL is not None:
        return EMBEDDINGMODEL
    
    if modelloadattempted:
        raise RuntimeError("Failed to load embedding model previously.")
    
    modelloadattempted = True
    try:
        print("Loading SentenceTransformer model all-mpnet-base-v2...", file=sys.stderr)
        EMBEDDINGMODEL = SentenceTransformer("all-mpnet-base-v2")
        print("Embedding model loaded successfully.", file=sys.stderr)
        return EMBEDDINGMODEL
    except Exception as e:
        print(f"ERROR: Could not load SentenceTransformer model. Error: {e}", file=sys.stderr)
        EMBEDDINGMODEL = None
        raise RuntimeError(f"Failed to load embedding model: {e}") from e


def get_embedding_for_query(text: str) -> List[float]:
    """Generates an embedding vector for a given text query.
    Lazily loads the model."""
    model = load_embedding_model()
    vector = model.encode(text)
    return vector.tolist()


def create_weaviate_client():
    """Create a new Weaviate client instance using environment credentials."""
    wcd_url = os.getenv("WEAVIATE_URL")
    wcd_api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not wcd_url or not wcd_api_key:
        raise RuntimeError(
            "WEAVIATE_URL and WEAVIATE_API_KEY environment variables must be set. "
            "Check your Cloud Run environment variables."
        )
    
    print("Reconnecting to Weaviate Cloud...", file=sys.stderr)
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=wcd_url,
        auth_credentials=weaviate.auth.AuthApiKey(wcd_api_key),
        skip_init_checks=True,
        additional_config=weaviate.config.AdditionalConfig(
            timeout=weaviate.config.Timeout(
                init=30,
                query=60,
                insert=120
            )
        )
    )
    return client


def is_client_alive(client) -> bool:
    """Check whether the existing Weaviate client connection is still healthy."""
    try:
        client.is_ready()
        return True
    except Exception as exc:
        print(f"Detected stale Weaviate connection: {exc}", file=sys.stderr)
        return False


def connect_with_retries(max_attempts: int = WEAVIATEMAXRETRIES):
    """Attempt to create a Weaviate client with retries and exponential backoff."""
    last_error = None
    delay = 1.0
    
    for attempt in range(1, max_attempts + 1):
        try:
            client = create_weaviate_client()
            client.is_ready()
            print("Weaviate connection established.", file=sys.stderr)
            return client
        except Exception as exc:
            last_error = exc
            print(f"Weaviate connection attempt {attempt}/{max_attempts} failed: {exc}", file=sys.stderr)
            if attempt < max_attempts:
                print(f"Retrying in {delay:.1f}s...", file=sys.stderr)
                time.sleep(delay)
                delay *= WEAVIATERETRYBACKOFFSECONDS
    
    raise RuntimeError(
        f"Unable to connect to Weaviate after {max_attempts} attempts: {last_error}"
    )


def should_run_healthcheck() -> bool:
    """Determine whether a healthcheck should run based on the last check timestamp."""
    now = time.time()
    return now - weaviatelasthealthcheck >= WEAVIATEHEALTHCHECKINTERVALSECONDS


def invalidate_client():
    """Invalidate and close the current Weaviate client.
    Called when connection errors occur during queries (query-level errors).
    Allows automatic reconnection on next tool call."""
    global weaviateclient, weaviateconnectiontested
    
    print("Invalidating Weaviate client due to query-level error.", file=sys.stderr)
    try:
        if weaviateclient:
            weaviateclient.close()
    except Exception as e:
        print(f"Error closing client: {e}", file=sys.stderr)
    
    weaviateclient = None
    weaviateconnectiontested = False


def execute_with_retry(func: Callable, max_retries: int = 2):
    """Execute a query with automatic retry on connection failure.
    On failure, invalidates client and retries once."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Query failed on attempt {attempt + 1}, retrying after client invalidation...", file=sys.stderr)
                invalidate_client()
                time.sleep(1)  # Brief delay before retry
            else:
                raise


def get_weaviate_client():
    """Initialize and return a Weaviate client with connection validation and retries.
    Ensures the connection is healthy before handing it to tools.
    
    CRITICAL FIXES:
    - Resets weaviateconnectiontested flag when connection dies (allows reconnection)
    - Resets flag on failed initial connection (allows retry on next call)
    - Tracks consecutive failures (gives up after 5 attempts)"""
    
    global weaviateclient, weaviateconnectiontested, weaviatelasthealthcheck, WEAVIATE_CONSECUTIVE_FAILURES
    
    with weaviateclientlock:
        if weaviateclient is not None:
            # Periodically validate the client is still alive
            if should_run_healthcheck():
                if is_client_alive(weaviateclient):
                    weaviatelasthealthcheck = time.time()
                    WEAVIATE_CONSECUTIVE_FAILURES = 0  # Reset on success
                    return weaviateclient
                else:
                    # Connection is dead - reset flag to allow reconnection
                    print("Weaviate client connection lost. Reconnecting...", file=sys.stderr)
                    try:
                        weaviateclient.close()
                    except Exception as e:
                        print(f"Error closing dead client: {e}", file=sys.stderr)
                    weaviateclient = None
                    weaviateconnectiontested = False  # ← CRITICAL: Allow reconnection
            else:
                return weaviateclient
        
        # Check if we've failed too many times
        if WEAVIATE_CONSECUTIVE_FAILURES >= WEAVIATE_MAX_CONSECUTIVE_FAILURES:
            raise RuntimeError(
                f"Weaviate connection failed {WEAVIATE_CONSECUTIVE_FAILURES} times. "
                f"Service appears to be down. Please check Weaviate status."
            )
        
        # Need to create a new client either first time or after failure
        if not weaviateconnectiontested:
            weaviateconnectiontested = True
            try:
                weaviateclient = connect_with_retries()
                weaviatelasthealthcheck = time.time()
                WEAVIATE_CONSECUTIVE_FAILURES = 0  # Reset on success
                print("Weaviate connection re-established.", file=sys.stderr)
                return weaviateclient
            except Exception as exc:
                # ← CRITICAL FIX: Reset flag on failed initial connection
                weaviateconnectiontested = False
                WEAVIATE_CONSECUTIVE_FAILURES += 1
                print(
                    f"Weaviate connection failed ({WEAVIATE_CONSECUTIVE_FAILURES}/"
                    f"{WEAVIATE_MAX_CONSECUTIVE_FAILURES})",
                    file=sys.stderr
                )
                raise


def format_date(date_input: Any) -> Optional[str]:
    """Safely converts a datetime object OR ISO date string to YYYY-MM-DD format."""
    dt = None
    
    if isinstance(date_input, datetime):
        dt = date_input
    elif isinstance(date_input, str):
        try:
            dt = datetime.fromisoformat(date_input.replace("Z", "+00:00"))
        except ValueError:
            return date_input
    
    if dt:
        return dt.strftime("%Y-%m-%d")
    return None


def parse_date_input(date_str: str) -> datetime:
    """Parses a YYYY-MM-DD string into a datetime object."""
    return datetime.strptime(date_str, "%Y-%m-%d")


# ============================================================================
# SEARCH AND RETRIEVAL TOOLS
# ============================================================================

def search_posts_hybrid(
    query: str,
    limit: int = 10,
    alpha: float = 0.7,
    startdate: Optional[str] = None,
    enddate: Optional[str] = None,
    topicfilter: Optional[str] = None,
    includescores: bool = True,
) -> Dict[str, Any]:
    """Advanced hybrid search combining semantic vector and keyword BM25 matching.
    Use alpha parameter to balance between meaning-based (1.0) and exact matches (0.0).
    Supports optional date range and topic filtering."""
    
    try:
        try:
            query_vector = get_embedding_for_query(query)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate query vector: {e}",
                "query": query
            }
        
        def do_search():
            client = get_weaviate_client()
            post_collection = client.collections.get("Post")
            filters = []
            
            if startdate and enddate:
                start_dt = parse_date_input(startdate).replace(hour=0, minute=0, second=0)
                end_dt = parse_date_input(enddate).replace(hour=23, minute=59, second=59)
                start_iso = start_dt.isoformat() + "Z"
                end_iso = end_dt.isoformat() + "Z"
                filters.append(Filter.by_property("postdate").greater_or_equal(start_iso))
                filters.append(Filter.by_property("postdate").less_or_equal(end_iso))
            
            if topicfilter:
                filters.append(Filter.by_property("finaltopic").equal(topicfilter))
            
            combined_filter = filters[0] if len(filters) == 1 else (filters[0] & filters[1] if len(filters) == 2 else None)
            
            results = post_collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=alpha,
                limit=limit,
                filters=combined_filter,
                query_properties=["postcontent", "posttitle", "finaltopic"],
                return_metadata=MetadataQuery(score=True) if includescores else None,
                return_properties=["postnumber", "posttitle", "postcontent", "finaltopic", "topicconfidence", "postdate", "secondarytopics"]
            )
            return results
        
        results = execute_with_retry(do_search, max_retries=2)
        
        formatted_results = []
        for obj in results.objects:
            result = {
                "postnumber": obj.properties.get("postnumber"),
                "title": obj.properties.get("posttitle"),
                "content": obj.properties.get("postcontent", "")[:300] + "...",
                "primarytopic": obj.properties.get("finaltopic"),
                "topicconfidence": obj.properties.get("topicconfidence"),
                "date": format_date(obj.properties.get("postdate")),
                "secondarytopics": obj.
