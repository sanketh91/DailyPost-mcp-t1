"""
Weaviate MCP Tools for Blog Post Database
==========================================
Version: 2.0.1 (FastMCP 2 with lazy initialization)
"""

import json
import uuid
import sys
import time
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import weaviate
from weaviate.classes.backup import BackupStorage
from sentence_transformers import SentenceTransformer
from weaviate.classes.query import Filter, MetadataQuery

# Import dynamic tool framework
from dynamic_tool_registry import get_dynamic_registry
from dynamic_tools_framework import DynamicToolGenerator

# mcp will be set by mcp_server.py before tools are registered
mcp = None

# ============================================
# LAZY INITIALIZATION - DO NOT CONNECT ON IMPORT
# ============================================

EMBEDDING_MODEL = None
MODEL_DIMENSION = 768
_model_load_attempted = False
_weaviate_client = None
_weaviate_connection_tested = False


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
    
    # Apply decorators to all static tool functions
    mcp.tool()(search_posts_hybrid)
    mcp.tool()(search_by_date_range)
    mcp.tool()(get_post_by_id)
    mcp.tool()(get_posts_batch)
    mcp.tool()(search_posts_by_topic)
    mcp.tool()(get_topic_statistics)
    mcp.tool()(find_similar_posts)
    mcp.tool()(search_by_keyword)
    mcp.tool()(list_all_topics)
    mcp.tool()(get_recent_posts)
    mcp.tool()(aggregate_posts)
    mcp.tool()(search_chunks)
    mcp.tool()(create_dynamic_tool)
    mcp.tool()(list_dynamic_tools)
    mcp.tool()(insert_object)
    mcp.tool()(update_object)
    mcp.tool()(delete_object)
    mcp.tool()(backup_weaviate)
    mcp.tool()(restore_backup)
    mcp.tool()(export_all_data)
    
    print("Static tools registered successfully", file=sys.stderr)
    
    # Register any persisted dynamic tools (deferred to first use)
    # _register_persisted_dynamic_tools()


def _load_embedding_model():
    """Lazy load the embedding model on first use"""
    global EMBEDDING_MODEL, _model_load_attempted
    
    if EMBEDDING_MODEL is not None:
        return EMBEDDING_MODEL
    
    if _model_load_attempted:
        raise RuntimeError("Failed to load embedding model previously.")
    
    _model_load_attempted = True
    
    try:
        print("📦 Loading SentenceTransformer model 'all-mpnet-base-v2'...", file=sys.stderr)
        EMBEDDING_MODEL = SentenceTransformer("all-mpnet-base-v2")
        print("✅ Embedding model loaded successfully.", file=sys.stderr)
        return EMBEDDING_MODEL
    except Exception as e:
        print(f"🚨 ERROR: Could not load SentenceTransformer model. Error: {e}", file=sys.stderr)
        EMBEDDING_MODEL = None
        raise RuntimeError(f"Failed to load embedding model: {e}")


def get_embedding_for_query(text: str) -> List[float]:
    """Generates an embedding vector for a given text query. Lazily loads the model."""
    model = _load_embedding_model()
    vector = model.encode(text)
    return vector.tolist()


def get_weaviate_client():
    """Initialize and return Weaviate client (lazy connection)"""
    global _weaviate_client, _weaviate_connection_tested
    
    # Test connection only on first call
    if not _weaviate_connection_tested:
        _weaviate_connection_tested = True
        print("🔌 Connecting to Weaviate...", file=sys.stderr)
        try:
            # Test connection with a short timeout
            test_client = weaviate.connect_to_local(
                skip_init_checks=True,
                additional_config=weaviate.config.AdditionalConfig(
                    timeout=weaviate.config.Timeout(init=5, query=30, insert=30)
                )
            )
            test_client.close()
            print("✅ Weaviate connection successful", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to Weaviate: {e}", file=sys.stderr)
            print("   Tools will attempt connection on first use", file=sys.stderr)
    
    # Return a new client for each request (connection pooling handled by weaviate-client)
    return weaviate.connect_to_local(skip_init_checks=True)


def _format_date(date_input: Any) -> Optional[str]:
    """Safely converts a datetime object OR ISO date string to 'YYYY-MM-DD' format."""
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


def _parse_date_input(date_str: str) -> datetime:
    """Parses a 'YYYY-MM-DD' string into a datetime object."""
    return datetime.strptime(date_str, "%Y-%m-%d")


# =============================
# MCP TOOL 1: HYBRID SEARCH
# =============================


def search_posts_hybrid(
    query: str,
    limit: int = 10,
    alpha: float = 0.7,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    topic_filter: Optional[str] = None,
    include_scores: bool = True,
) -> Dict[str, Any]:
    """Advanced hybrid search combining semantic (vector) and keyword (BM25) matching.
    Use alpha parameter to balance between meaning-based (1.0) and exact matches (0.0).
    Supports optional date range and topic filtering."""
    client = get_weaviate_client()
    try:
        try:
            query_vector = get_embedding_for_query(query)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate query vector: {e}",
                "query": query,
            }

        post_collection = client.collections.get("Post")

        filters = []
        if start_date and end_date:
            start_dt = _parse_date_input(start_date).replace(hour=0, minute=0, second=0)
            end_dt = _parse_date_input(end_date).replace(hour=23, minute=59, second=59)
            start_iso = start_dt.isoformat() + "Z"
            end_iso = end_dt.isoformat() + "Z"

            filters.append(
                Filter.by_property("post_date").greater_or_equal(start_iso)
                & Filter.by_property("post_date").less_or_equal(end_iso)
            )

        if topic_filter:
            filters.append(Filter.by_property("topic_name").equal(topic_filter))

        combined_filter = (
            filters[0]
            if len(filters) == 1
            else (filters[0] & filters[1] if len(filters) == 2 else None)
        )

        results = post_collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=alpha,
            limit=limit,
            filters=combined_filter,
            query_properties=["post_content", "post_title", "topic_name"],
            return_metadata=MetadataQuery(score=True) if include_scores else None,
            return_properties=[
                "post_number",
                "post_title",
                "post_content",
                "topic_name",
                "post_date",
                "all_topic_names",
            ],
        )

        formatted_results = []
        for obj in results.objects:
            result = {
                "post_number": obj.properties.get("post_number"),
                "title": obj.properties.get("post_title"),
                "content": obj.properties.get("post_content", "")[:300] + "...",
                "topic": obj.properties.get("topic_name"),
                "date": _format_date(obj.properties.get("post_date")),
                "all_topics": obj.properties.get("all_topic_names"),
            }
            if include_scores and hasattr(obj.metadata, "score"):
                result["relevance_score"] = obj.metadata.score
            formatted_results.append(result)

        return {
            "success": True,
            "query": query,
            "total_results": len(formatted_results),
            "search_params": {
                "alpha": alpha,
                "limit": limit,
                "filters_applied": bool(filters),
            },
            "results": formatted_results,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "query": query}
    finally:
        if client:
            client.close()


# =============================
# MCP TOOL 2: DATE RANGE SEARCH
# =============================


def search_by_date_range(
    start_date: str,
    end_date: str,
    limit: int = 20,
    topic_filter: Optional[str] = None,
    sort_order: str = "desc",
) -> Dict[str, Any]:
    """Search for posts published within a specific date range (YYYY-MM-DD format).
    Results can be sorted by newest first (desc) or oldest first (asc).
    Optional topic filter available."""
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")

        start_dt = _parse_date_input(start_date).replace(hour=0, minute=0, second=0)
        end_dt = _parse_date_input(end_date).replace(hour=23, minute=59, second=59)
        start_iso = start_dt.isoformat() + "Z"
        end_iso = end_dt.isoformat() + "Z"

        date_filter = Filter.by_property("post_date").greater_or_equal(
            start_iso
        ) & Filter.by_property("post_date").less_or_equal(end_iso)

        if topic_filter:
            date_filter = date_filter & Filter.by_property("topic_name").equal(
                topic_filter
            )

        results = post_collection.query.fetch_objects(
            filters=date_filter,
            limit=limit,
            return_properties=[
                "post_number",
                "post_title",
                "post_date",
                "topic_name",
                "post_content",
            ],
        )

        temp_results = []
        for obj in results.objects:
            raw_date = obj.properties.get("post_date")
            temp_results.append(
                {
                    "data": {
                        "post_number": obj.properties.get("post_number"),
                        "title": obj.properties.get("post_title"),
                        "date": _format_date(raw_date),
                        "topic": obj.properties.get("topic_name"),
                        "preview": obj.properties.get("post_content", "")[:150] + "...",
                    },
                    "sort_key": raw_date or "",
                }
            )

        if sort_order == "desc":
            temp_results.sort(key=lambda x: x["sort_key"], reverse=True)
        else:
            temp_results.sort(key=lambda x: x["sort_key"])

        formatted_results = [item["data"] for item in temp_results]

        return {
            "success": True,
            "date_range": f"{start_date} to {end_date}",
            "total_results": len(formatted_results),
            "topic_filter": topic_filter,
            "results": formatted_results,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()


# =============================
# MCP TOOL 3: GET POST BY ID
# =============================


def get_post_by_id(post_number: int) -> Dict[str, Any]:
    """Retrieve a complete single post by its unique post number (ID).
    Includes full content, metadata, and all associated topic information.
    Fast direct lookup method."""
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")
        results = post_collection.query.fetch_objects(
            filters=Filter.by_property("post_number").equal(post_number),
            limit=1,
            return_properties=[
                "post_number",
                "post_title",
                "post_content",
                "post_date",
                "topic_name",
                "all_topic_names",
                "secondary_topic_names",
                "secondary_topic_similarities",
            ],
        )
        if not results.objects:
            return {"success": False, "error": f"Post #{post_number} not found"}
        obj = results.objects[0]
        return {
            "success": True,
            "post": {
                "post_number": obj.properties.get("post_number"),
                "title": obj.properties.get("post_title"),
                "content": obj.properties.get("post_content"),
                "date": _format_date(obj.properties.get("post_date")),
                "primary_topic": obj.properties.get("topic_name"),
                "all_topics": obj.properties.get("all_topic_names"),
                "secondary_topics": obj.properties.get("secondary_topic_names", []),
                "topic_similarities": obj.properties.get(
                    "secondary_topic_similarities", []
                ),
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()


# =============================
# MCP TOOL 4: BATCH POST RETRIEVAL
# =============================


def get_posts_batch(
    post_numbers: List[int], include_content: bool = True
) -> Dict[str, Any]:
    """Batch retrieve multiple posts by their IDs efficiently.
    Can optionally exclude post content for metadata-only queries.
    Returns list of found posts and any missing post IDs."""
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")

        properties_list = ["post_number", "post_title", "post_date", "topic_name"]
        if include_content:
            properties_list.append("post_content")

        results = post_collection.query.fetch_objects(
            filters=Filter.by_property("post_number").contains_any(post_numbers),
            limit=len(post_numbers),
            return_properties=properties_list,
        )

        found_posts, found_ids = [], set()
        for obj in results.objects:
            post_id = obj.properties.get("post_number")
            found_ids.add(post_id)
            post_data = {
                "post_number": post_id,
                "title": obj.properties.get("post_title"),
                "date": _format_date(obj.properties.get("post_date")),
                "topic": obj.properties.get("topic_name"),
            }
            if include_content and "post_content" in obj.properties:
                post_data["content"] = obj.properties.get("post_content")
            found_posts.append(post_data)
        missing = [pid for pid in post_numbers if pid not in found_ids]
        return {
            "success": True,
            "requested_count": len(post_numbers),
            "found_count": len(found_posts),
            "missing_posts": missing,
            "posts": found_posts,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()


# =============================
# MCP TOOL 5: TOPIC SEARCH
# =============================


def search_posts_by_topic(
    topic_name: str,
    limit: int = 15,
    fuzzy: bool = True,
    include_secondary: bool = False,
) -> Dict[str, Any]:
    """Search posts by topic name with optional fuzzy matching for partial names.
    Can include posts where the topic is secondary.
    Returns matched topic names and associated posts."""
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")
        if fuzzy:
            all_posts = post_collection.query.fetch_objects(
                limit=5000,
                return_properties=["post_topic", "topic_name", "topic_keywords"],
            )
            matching_topic_ids, matched_topic_names = set(), set()
            search_lower = topic_name.lower()
            for post in all_posts.objects:
                topic_id, t_name, t_keywords = (
                    post.properties.get("post_topic"),
                    post.properties.get("topic_name", "") or "",
                    post.properties.get("topic_keywords", "") or "",
                )
                if search_lower in t_name.lower() or search_lower in t_keywords.lower():
                    matching_topic_ids.add(topic_id)
                    matched_topic_names.add(t_name)
            if not matching_topic_ids:
                return {"success": False, "error": "No matching topics found"}
            filter_prop = "all_topic_ids" if include_secondary else "post_topic"
            results = post_collection.query.fetch_objects(
                filters=Filter.by_property(filter_prop).contains_any(
                    list(matching_topic_ids)
                ),
                limit=limit,
                return_properties=[
                    "post_number",
                    "post_title",
                    "post_date",
                    "topic_name",
                    "post_content",
                    "all_topic_ids",
                ],
            )
            matched_topics_list = list(matched_topic_names)
        else:
            if include_secondary:
                results = post_collection.query.bm25(
                    query=topic_name,
                    query_properties=["topic_name", "all_topic_names"],
                    limit=limit,
                    return_properties=[
                        "post_number",
                        "post_title",
                        "post_date",
                        "topic_name",
                        "post_content",
                        "all_topic_ids",
                    ],
                )
            else:
                results = post_collection.query.fetch_objects(
                    filters=Filter.by_property("topic_name").equal(topic_name),
                    limit=limit,
                    return_properties=[
                        "post_number",
                        "post_title",
                        "post_date",
                        "topic_name",
                        "post_content",
                    ],
                )
            matched_topics_list = [topic_name]

        formatted_results = []
        for obj in results.objects:
            result_data = {
                "post_number": obj.properties.get("post_number"),
                "title": obj.properties.get("post_title"),
                "date": _format_date(obj.properties.get("post_date")),
                "topic": obj.properties.get("topic_name"),
                "preview": obj.properties.get("post_content", "")[:150] + "...",
            }
            if include_secondary:
                result_data["is_secondary"] = (
                    obj.properties.get("topic_name") != topic_name
                )
            formatted_results.append(result_data)

        response = {
            "success": True,
            "topic_search": topic_name,
            "fuzzy_match": fuzzy,
            "total_results": len(formatted_results),
            "results": formatted_results,
        }
        if fuzzy:
            response["matched_topics"] = matched_topics_list
        return response
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()


# =============================
# MCP TOOL 6: TOPIC STATISTICS
# =============================


def get_topic_statistics(
    top_n: int = 15, include_distribution: bool = True
) -> Dict[str, Any]:
    """Get comprehensive statistics on topic distribution across the database.
    Includes unique topic count, multi-label post percentage, and top N topics.
    Optional breakdown of posts grouped by number of topics."""
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")
        all_posts = post_collection.query.fetch_objects(
            limit=10000,
            return_properties=[
                "post_topic",
                "topic_name",
                "secondary_topic_names",
                "all_topic_ids",
            ],
        )
        topic_counts, topic_name_to_ids = defaultdict(int), defaultdict(set)
        multi_label_count, topic_count_distribution = 0, defaultdict(int)

        for item in all_posts.objects:
            topic_id, topic_name = (
                item.properties.get("post_topic"),
                item.properties.get("topic_name", "Unknown"),
            )
            if topic_id is not None and topic_name:
                topic_counts[topic_name] += 1
                topic_name_to_ids[topic_name].add(topic_id)
            if item.properties.get("secondary_topic_names"):
                multi_label_count += 1
            if include_distribution:
                all_topic_ids = item.properties.get("all_topic_ids", [])
                num_topics = (
                    len(all_topic_ids) if all_topic_ids else (1 if topic_id else 0)
                )
                topic_count_distribution[
                    f"{num_topics}_topic{'s' if num_topics != 1 else ''}"
                    if num_topics < 4
                    else "4+_topics"
                ] += 1

        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]
        total_posts_count = len(all_posts.objects)
        topic_breakdown = [
            {
                "topic_name": name,
                "post_count": count,
                "percentage": round((count / total_posts_count) * 100, 2)
                if total_posts_count > 0
                else 0,
                "cluster_count": len(topic_name_to_ids[name]),
            }
            for name, count in sorted_topics
        ]

        response = {
            "success": True,
            "statistics": {
                "total_posts": total_posts_count,
                "unique_topics": len(topic_counts),
                "multi_label_posts": multi_label_count,
                "multi_label_percentage": round(
                    (multi_label_count / total_posts_count) * 100, 2
                )
                if total_posts_count > 0
                else 0,
                "avg_posts_per_topic": round(total_posts_count / len(topic_counts), 2)
                if topic_counts
                else 0,
            },
            "top_topics": topic_breakdown,
        }
        if include_distribution:
            response["distribution"] = dict(topic_count_distribution)
        return response
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()


# =============================
# MCP TOOL 7: SEMANTIC SIMILARITY
# =============================


def find_similar_posts(
    post_number: int, limit: int = 5, min_similarity: float = 0.7
) -> Dict[str, Any]:
    """Find semantically similar posts to a reference post using vector similarity.
    Set min_similarity threshold to filter results (0.0 to 1.0 range).
    Ideal for 'more like this' features."""
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")
        reference = post_collection.query.fetch_objects(
            filters=Filter.by_property("post_number").equal(post_number),
            limit=1,
            return_properties=[
                "post_number",
                "post_title",
                "topic_name",
                "avg_embedding",
            ],
        )
        if not reference.objects:
            return {
                "success": False,
                "error": f"Reference post #{post_number} not found",
            }
        ref_obj = reference.objects[0]
        ref_embedding = ref_obj.properties.get("avg_embedding")
        if not ref_embedding:
            return {
                "success": False,
                "error": f"Post #{post_number} has no embedding available",
            }

        similar = post_collection.query.near_vector(
            near_vector=ref_embedding,
            limit=limit + 1,
            return_metadata=MetadataQuery(distance=True),
            return_properties=[
                "post_number",
                "post_title",
                "topic_name",
                "post_content",
            ],
        )
        similar_posts = []
        for obj in similar.objects:
            if obj.properties.get("post_number") == post_number:
                continue
            similarity = (
                1 - obj.metadata.distance if hasattr(obj.metadata, "distance") else 0
            )
            if similarity >= min_similarity:
                similar_posts.append(
                    {
                        "post_number": obj.properties.get("post_number"),
                        "title": obj.properties.get("post_title"),
                        "topic": obj.properties.get("topic_name"),
                        "similarity_score": round(similarity, 4),
                        "preview": obj.properties.get("post_content", "")[:150] + "...",
                    }
                )
        return {
            "success": True,
            "reference_post": {
                "post_number": ref_obj.properties.get("post_number"),
                "title": ref_obj.properties.get("post_title"),
                "topic": ref_obj.properties.get("topic_name"),
            },
            "similar_posts": similar_posts[:limit],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()


# =============================
# MCP TOOL 8: KEYWORD SEARCH
# =============================


def search_by_keyword(
    keyword: str,
    search_in: List[str] = None,
    limit: int = 10,
    exact_match: bool = False,
) -> Dict[str, Any]:
    """Pure keyword search using BM25 algorithm for exact term matching.
    Search specific fields: 'content', 'title', or 'topic'.
    Shows match context and surrounding text."""
    if search_in is None:
        search_in = ["content", "title"]

    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")
        field_map = {
            "content": "post_content",
            "title": "post_title",
            "topic": "topic_name",
        }
        query_properties = [
            field_map[field] for field in search_in if field in field_map
        ]

        results = post_collection.query.bm25(
            query=keyword,
            query_properties=query_properties,
            limit=limit,
            return_properties=[
                "post_number",
                "post_title",
                "post_content",
                "topic_name",
                "post_date",
            ],
        )

        formatted_results = []
        for obj in results.objects:
            content = obj.properties.get("post_content", "")
            match_idx = content.lower().find(keyword.lower())
            match_context = (
                "..."
                + content[
                    max(0, match_idx - 50) : min(
                        len(content), match_idx + len(keyword) + 50
                    )
                ]
                + "..."
                if match_idx != -1
                else content[:150] + "..."
            )
            formatted_results.append(
                {
                    "post_number": obj.properties.get("post_number"),
                    "title": obj.properties.get("post_title"),
                    "topic": obj.properties.get("topic_name"),
                    "date": _format_date(obj.properties.get("post_date")),
                    "preview": content[:200] + "...",
                    "match_context": match_context,
                }
            )
        return {
            "success": True,
            "keyword": keyword,
            "search_fields": search_in,
            "total_results": len(formatted_results),
            "results": formatted_results,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()


# =============================
# MCP TOOL 9: LIST ALL TOPICS
# =============================


def list_all_topics(sort_by: str = "count", min_posts: int = 1) -> Dict[str, Any]:
    """List all available topics in the database with post counts and percentages.
    Sort by post count (descending) or alphabetically by name.
    Filter by minimum post threshold."""
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")
        all_posts = post_collection.query.fetch_objects(
            limit=10000, return_properties=["post_topic", "topic_name"]
        )
        topic_counts, topic_name_to_ids = defaultdict(int), defaultdict(set)

        for item in all_posts.objects:
            topic_id, topic_name = (
                item.properties.get("post_topic"),
                item.properties.get("topic_name", "Unknown"),
            )
            if topic_id is not None and topic_name:
                topic_counts[topic_name] += 1
                topic_name_to_ids[topic_name].add(topic_id)

        filtered_topics = [
            (name, count) for name, count in topic_counts.items() if count >= min_posts
        ]
        filtered_topics.sort(
            key=lambda x: x[0] if sort_by == "name" else x[1],
            reverse=(sort_by == "count"),
        )

        total_posts = len(all_posts.objects)
        topic_list = [
            {
                "topic_name": name,
                "post_count": count,
                "percentage": round((count / total_posts) * 100, 2)
                if total_posts > 0
                else 0,
                "cluster_count": len(topic_name_to_ids[name]),
            }
            for name, count in filtered_topics
        ]
        return {"success": True, "total_topics": len(topic_list), "topics": topic_list}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()


# =============================
# MCP TOOL 10: RECENT POSTS
# =============================


def get_recent_posts(
    days: int = 30, limit: int = 20, topic_filter: Optional[str] = None
) -> Dict[str, Any]:
    """Get the most recently published posts within a specified number of days.
    Shows how many days ago each post was published.
    Optional topic filter available."""
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")
        threshold_date = (datetime.now() - timedelta(days=days)).replace(
            hour=0, minute=0, second=0
        ).isoformat() + "Z"
        date_filter = Filter.by_property("post_date").greater_or_equal(threshold_date)
        if topic_filter:
            date_filter = date_filter & Filter.by_property("topic_name").equal(
                topic_filter
            )

        results = post_collection.query.fetch_objects(
            filters=date_filter,
            limit=limit,
            return_properties=[
                "post_number",
                "post_title",
                "post_date",
                "topic_name",
                "post_content",
            ],
        )

        formatted_results = []
        now = datetime.now()
        for obj in results.objects:
            post_date_iso = obj.properties.get("post_date")
            post_date_formatted = _format_date(post_date_iso)
            days_ago = None
            if post_date_iso:
                try:
                    post_date = datetime.fromisoformat(
                        post_date_iso.replace("Z", "+00:00")
                    )
                    days_ago = (now - post_date.replace(tzinfo=None)).days
                except:
                    days_ago = None

            formatted_results.append(
                {
                    "post_number": obj.properties.get("post_number"),
                    "title": obj.properties.get("post_title"),
                    "date": post_date_formatted,
                    "topic": obj.properties.get("topic_name"),
                    "preview": obj.properties.get("post_content", "")[:150] + "...",
                    "days_ago": days_ago,
                }
            )

        formatted_results.sort(key=lambda x: x["date"] or "", reverse=True)

        return {
            "success": True,
            "period": f"Last {days} days",
            "total_results": len(formatted_results),
            "topic_filter": topic_filter,
            "results": formatted_results,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()


# =============================
# MCP TOOL 11: AGGREGATE QUERY
# =============================


def aggregate_posts(
    group_by: str = "topic", date_range: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Aggregate posts by topic, month, or year for analytics.
    Optional date range filter (YYYY-MM-DD format).
    Returns grouped counts and percentages."""
    client = get_weaviate_client()
    try:
        post_collection = client.collections.get("Post")
        filters = None
        if date_range:
            start_dt = _parse_date_input(date_range["start"]).replace(
                hour=0, minute=0, second=0
            )
            end_dt = _parse_date_input(date_range["end"]).replace(
                hour=23, minute=59, second=59
            )
            start_iso = start_dt.isoformat() + "Z"
            end_iso = end_dt.isoformat() + "Z"
            filters = Filter.by_property("post_date").greater_or_equal(
                start_iso
            ) & Filter.by_property("post_date").less_or_equal(end_iso)

        results = post_collection.query.fetch_objects(
            filters=filters, limit=10000, return_properties=["topic_name", "post_date"]
        )

        aggregations = defaultdict(int)
        for obj in results.objects:
            key = "Unknown"
            if group_by == "topic":
                key = obj.properties.get("topic_name", "Unknown")
            elif group_by in ["month", "year"]:
                date_iso = obj.properties.get("post_date")
                if date_iso:
                    try:
                        dt = datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
                        if group_by == "month":
                            key = dt.strftime("%Y-%m")
                        else:
                            key = str(dt.year)
                    except:
                        key = "Unknown Date"
                else:
                    key = "Unknown Date"
            aggregations[key] += 1

        total = len(results.objects)
        formatted_aggs = [
            {
                "group": group,
                "count": count,
                "percentage": round((count / total) * 100, 2) if total > 0 else 0,
            }
            for group, count in sorted(
                aggregations.items(), key=lambda x: x[1], reverse=True
            )
        ]

        return {
            "success": True,
            "grouped_by": group_by,
            "total_posts": total,
            "aggregations": formatted_aggs,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if client:
            client.close()


# =============================
# MCP TOOL 12: SEARCH CHUNKS
# =============================


def search_chunks(
    query: str, limit: int = 10, post_title: Optional[str] = None
) -> Dict[str, Any]:
    """Search for specific passages within posts using semantic search.
    Returns chunk text, chunk number, and associated post information.
    Optional filter to search within a specific post title."""
    client = get_weaviate_client()
    try:
        chunk_collection = client.collections.get("Chunk")

        try:
            query_vector = get_embedding_for_query(query)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate query vector: {e}",
                "query": query,
            }

        filters = None
        if post_title:
            filters = Filter.by_property("post_title").equal(post_title)

        results = chunk_collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            filters=filters,
            return_properties=[
                "post_number",
                "post_title",
                "chunk_number",
                "chunk_text",
                "chunk_topic",
            ],
        )

        formatted_results = []
        for obj in results.objects:
            props = obj.properties
            formatted_results.append(
                {
                    "post_number": props.get("post_number"),
                    "post_title": props.get("post_title"),
                    "chunk_number": props.get("chunk_number"),
                    "chunk_text": props.get("chunk_text"),
                    "topic": props.get("chunk_topic"),
                }
            )

        return {
            "success": True,
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "query": query}
    finally:
        if client:
            client.close()


# =============================
# MCP TOOL 13: CREATE DYNAMIC TOOL
# =============================


def create_dynamic_tool(
    query_description: str,
    tool_name: Optional[str] = None
) -> Dict[str, Any]:
    """DYNAMIC TOOL CREATOR. Creates new MCP tools on-the-fly when existing tools cannot fulfill a query.
    Use this when a user's request requires capabilities not available in existing tools. 
    The new tool will be immediately available without server restart.
    DO NOT use for queries that can be handled by existing tools."""
    generator = DynamicToolGenerator()
    result = generator.create_and_register_tool(
        query_description=query_description,
        tool_name=tool_name,
        parameters={"tool_name": tool_name} if tool_name else None
    )
    
    # If tool was successfully created, try to register it with FastMCP
    if result.get("success") and mcp is not None:
        tool_name_created = result.get("tool_name")
        if tool_name_created:
            registry = get_dynamic_registry()
            handler = registry.get_handler(tool_name_created)
            tool_def = None
            
            # Get the tool definition from registry
            for dyn_tool in registry.get_tool_definitions():
                if dyn_tool.name == tool_name_created:
                    tool_def = dyn_tool
                    break
            
            if handler and tool_def:
                try:
                    # Create a wrapper function that calls the handler
                    async def tool_wrapper(**kwargs):
                        """Wrapper for dynamic tool handler"""
                        result = await handler(kwargs)
                        # Convert result to dict if it's a list (from create_json_response)
                        if isinstance(result, list):
                            # Extract text from TextContent if present
                            if result and hasattr(result[0], 'text'):
                                import json
                                return json.loads(result[0].text)
                            return result
                        return result
                    
                    # Update the wrapper's metadata for FastMCP
                    tool_wrapper.__name__ = tool_name_created
                    tool_wrapper.__doc__ = tool_def.description
                    
                    # Try to add tool to FastMCP using add_tool method
                    # FastMCP may support runtime tool registration through its tool manager
                    try:
                        # Try different ways to access FastMCP's tool manager
                        tool_manager = None
                        if hasattr(mcp, 'tool_manager'):
                            tool_manager = mcp.tool_manager
                        elif hasattr(mcp, '_tool_manager'):
                            tool_manager = mcp._tool_manager
                        elif hasattr(mcp, 'tools'):
                            tool_manager = mcp.tools
                        
                        if tool_manager and hasattr(tool_manager, 'add_tool'):
                            # Use FastMCP's add_tool method
                            tool_manager.add_tool(
                                tool_wrapper,
                                name=tool_name_created,
                                description=tool_def.description
                            )
                            result["registered_with_fastmcp"] = True
                            result["note"] = f"Tool '{tool_name_created}' registered with FastMCP and is immediately available."
                        else:
                            # FastMCP doesn't expose tool manager directly or doesn't support runtime addition
                            # Tools will be available through dynamic registry
                            result["registered_with_fastmcp"] = False
                            result["note"] = "Tool registered in dynamic registry. FastMCP doesn't support runtime tool registration. Tool may require server restart to appear in tool list, but handler is available."
                    except Exception as e:
                        result["registered_with_fastmcp"] = False
                        result["registration_warning"] = f"Could not register with FastMCP: {str(e)}"
                        result["note"] = "Tool is available through dynamic registry handler. May require server restart for full integration."
                except Exception as e:
                    result["registration_error"] = str(e)
                    result["note"] = "Tool created but registration with FastMCP failed. Tool is still available through dynamic registry."
    
    return result


# =============================
# MCP TOOL 14: LIST DYNAMIC TOOLS
# =============================


def list_dynamic_tools() -> Dict[str, Any]:
    """LIST DYNAMIC TOOLS. Lists all dynamically created tools that are currently registered.
    Use to see what custom tools have been created and are available."""
    registry = get_dynamic_registry()
    result = registry.list_tools()
    
    # Also check if tools are in the main tool list
    # Note: In FastMCP, we can't directly query the tool list, but we can verify registry
    all_tool_names = [
        "search_posts_hybrid", "search_by_date_range", "get_post_by_id",
        "get_posts_batch", "search_posts_by_topic", "get_topic_statistics",
        "find_similar_posts", "search_by_keyword", "list_all_topics",
        "get_recent_posts", "aggregate_posts", "search_chunks",
        "create_dynamic_tool", "list_dynamic_tools"
    ]
    dynamic_tool_names = [t["name"] for t in result.get("tools", [])]
    
    # Check which dynamic tools are actually available in the registry
    available_in_registry = [name for name in dynamic_tool_names if registry.has_tool(name)]
    not_in_registry = [name for name in dynamic_tool_names if not registry.has_tool(name)]
    
    result["tools_in_registry"] = available_in_registry
    result["tools_not_in_registry"] = not_in_registry
    result["total_static_tools"] = len(all_tool_names)
    
    if not_in_registry:
        result["warning"] = f"Some dynamic tools ({not_in_registry}) are listed but not in registry. This may indicate a persistence issue."
        result["suggestion"] = "Try recreating the tools or check the dynamic_tools_storage directory."
    
    return result
# These tools provide CRUD operations, backup/restore, and export functionality for your Weaviate database.

# ================================
# MCP TOOL: INSERT OBJECT (ROW)
# ================================
def insert_object(class_name: str, properties: dict) -> dict:
    """
    Inserts a new object (row) into a Weaviate class.

    Parameters:
        class_name (str): Name of the Weaviate class.
        properties (dict): Properties of the object to insert. Example: {"title": "Inception", "year": 2010}

    Returns:
        dict: Success/failure status, message, and new object ID.
    """
    client = get_weaviate_client()
    try:
        coll = client.collections.get(class_name)
        obj_id = coll.data.insert(properties)
        return {
            "success": True,
            "message": "Object inserted",
            "id": str(obj_id)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        client.close()

# ================================
# MCP TOOL: UPDATE OBJECT
# ================================
def update_object(class_name: str, object_id: str, properties: dict) -> dict:
    """
    Updates an existing object's properties (PUT) in a given class.

    Parameters:
        class_name (str): Name of the Weaviate class.
        object_id (str): UUID of the object to update.
        properties (dict): Dictionary of updated properties.

    Returns:
        dict: Success/failure status, message, and updated properties.
    """
    client = get_weaviate_client()
    try:
        coll = client.collections.get(class_name)
        coll.data.update(object_id, properties)
        return {
            "success": True,
            "message": "Object updated",
            "id": object_id,
            "updated_properties": properties
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        client.close()

# ================================
# MCP TOOL: DELETE OBJECT
# ================================
def delete_object(class_name: str, object_id: str) -> dict:
    """
    Deletes an object by UUID from a specified class in Weaviate.

    Parameters:
        class_name (str): Name of the Weaviate class.
        object_id (str): UUID of the object to delete.

    Returns:
        dict: Success/failure status, message, class name, and deleted object ID.
    """
    client = get_weaviate_client()
    try:
        collection = client.collections.get(class_name)
        collection.data.delete_by_id(object_id)
        return {
            "success": True,
            "message": "Object deleted successfully",
            "class": class_name,
            "id": object_id
        }
    except Exception as e:
        return {"success": False, "error": f"Deletion failed: {str(e)}"}
    finally:
        try:
            client.close()
        except:
            pass

# ================================
# MCP TOOL: BACKUP WEAVIATE DATABASE
# ================================
def backup_weaviate(backup_id: str) -> dict:
    """
    Creates a filesystem backup of the entire local Weaviate database.

    Parameters:
        backup_id (str): Identifier for the backup (required).

    Returns:
        dict: Success/failure status, message, backup details.
    """
    client = get_weaviate_client()
    try:
        if not backup_id or not isinstance(backup_id, str):
            return {"success": False, "error": "backup_id must be a non-empty string"}
        backup_id = backup_id.replace(" ", "_")
        result = client.backup.create(
            backup_id=backup_id,
            backend=BackupStorage.FILESYSTEM,
            wait_for_completion=True
        )
        return {
            "success": True,
            "message": "Backup created successfully",
            "backup_id": backup_id,
            "backend": "filesystem",
            "timestamp": time.time(),
            "status": getattr(result, "status", "SUCCESS")
        }
    except Exception as e:
        error_msg = str(e).lower()
        if "backup" in error_msg or "not" in error_msg:
            return {
                "success": False,
                "error": "Backup failed. Ensure Weaviate has backup module enabled with BACKUP_FILESYSTEM_PATH in docker-compose.yml",
                "details": str(e)
            }
        return {"success": False, "error": f"Backup failed: {str(e)}"}
    finally:
        try:
            client.close()
        except:
            pass

# ================================
# MCP TOOL: RESTORE WEAVIATE BACKUP
# ================================
def restore_backup(backup_id: str) -> dict:
    """
    Restores a filesystem backup of local Weaviate database by backup ID.

    Parameters:
        backup_id (str): Identifier for the backup to restore.

    Returns:
        dict: Success/failure status, message, restore details.
    """
    client = get_weaviate_client()
    try:
        if not backup_id or not isinstance(backup_id, str):
            return {"success": False, "error": "backup_id must be a non-empty string"}
        backup_id = backup_id.replace(" ", "_")
        result = client.backup.restore(
            backup_id=backup_id,
            backend=BackupStorage.FILESYSTEM,
            wait_for_completion=True
        )
        return {
            "success": True,
            "message": "Backup restored successfully",
            "backup_id": backup_id,
            "backend": "filesystem",
            "timestamp": time.time(),
            "status": getattr(result, "status", "SUCCESS")
        }
    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg or "does not exist" in error_msg:
            return {
                "success": False,
                "error": f"Backup '{backup_id}' not found. Verify backup exists before restoring.",
                "details": str(e)
            }
        if "no backup backend" in error_msg or ("not found" in error_msg and "backup" in error_msg):
            return {
                "success": False,
                "error": "Restore failed. Ensure Weaviate has backup module enabled.",
                "details": str(e)
            }
        return {"success": False, "error": f"Restore failed: {str(e)}", "backup_id": backup_id}
    finally:
        try:
            client.close()
        except:
            pass

# ================================
# MCP TOOL: EXPORT ALL DATA
# ================================
def export_all_data(output_dir: str = "./weaviate_export") -> dict:
    """
    Exports the entire Weaviate database schema and all objects to JSON files.

    Parameters:
        output_dir (str): Directory for exported JSON files (default: './weaviate_export').

    Returns:
        dict: Success/failure status, message, export details, exported collections, total objects.
    """
    def _json_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (bytes, bytearray)):
            return obj.hex()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        raise TypeError(f"Type {type(obj)} not serializable")
    client = get_weaviate_client()
    try:
        # Ensure writable directory
        output_dir = os.path.abspath(output_dir)
        try:
            os.makedirs(output_dir, exist_ok=True)
            test_file = os.path.join(output_dir, ".test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except (OSError, PermissionError):
            output_dir = "/tmp/weaviate_export"
            os.makedirs(output_dir, exist_ok=True)
        # Get all collections
        try:
            collections_list = client.collections.list_all()
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to retrieve collections: {str(e)}"
            }
        # Build schema from collections
        schema = {"classes": []}
        for col in collections_list:
            col_name = col if isinstance(col, str) else (col.name if hasattr(col, 'name') else str(col))
            schema["classes"].append({"class": col_name})
        schema_path = os.path.join(output_dir, "schema.json")
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)
        exported_collections = []
        total_objects = 0
        for col in collections_list:
            collection_name = col if isinstance(col, str) else (col.name if hasattr(col, 'name') else str(col))
            try:
                collection = client.collections.get(collection_name)
                all_data = []
                objects_count = 0
                try:
                    for obj in collection.iterator():
                        obj_id = str(getattr(obj, 'uuid', getattr(obj, 'id', obj)))
                        obj_props = obj.properties if hasattr(obj, 'properties') else {}
                        all_data.append({"id": obj_id, "properties": obj_props})
                        objects_count += 1
                except Exception:
                    response = collection.query.fetch_all()
                    objects_list = response.objects if hasattr(response, 'objects') else list(response)
                    for obj in objects_list:
                        obj_id = str(getattr(obj, 'uuid', getattr(obj, 'id', obj)))
                        obj_props = obj.properties if hasattr(obj, 'properties') else {}
                        all_data.append({"id": obj_id, "properties": obj_props})
                        objects_count += 1
                collection_path = os.path.join(output_dir, f"{collection_name}.json")
                with open(collection_path, "w") as f:
                    json.dump(all_data, f, indent=2, default=_json_serializer)
                total_objects += objects_count
                exported_collections.append({"collection": collection_name, "object_count": objects_count, "file": f"{collection_name}.json"})
            except Exception as class_error:
                return {
                    "success": False,
                    "error": f"Error exporting collection {collection_name}: {str(class_error)}",
                    "partially_exported": exported_collections
                }
        return {
            "success": True,
            "message": "Export completed successfully",
            "directory": output_dir,
            "schema_file": "schema.json",
            "exported_collections": exported_collections,
            "total_collections": len(exported_collections),
            "total_objects": total_objects,
            "timestamp": time.time()
        }
    except Exception as e:
        return {"success": False, "error": f"Export failed: {str(e)}", "attempted_directory": output_dir}
    finally:
        try:
            client.close()
        except:
            pass

# =============================
# Helper: Register Persisted Dynamic Tools
# =============================


def _register_persisted_dynamic_tools():
    """Register any dynamic tools that were persisted to disk.
    This is called during server startup to restore dynamic tools."""
    if mcp is None:
        return
    
    registry = get_dynamic_registry()
    
    # Fix any missing handlers for persisted tools
    fix_result = registry.fix_missing_handlers()
    
    # Get all dynamic tools and try to register them with FastMCP
    # Note: FastMCP doesn't support runtime registration, but tools are available
    # through the dynamic registry's handler lookup mechanism
    dynamic_tools = registry.get_tool_definitions()
    
    if fix_result.get("fixed"):
        print(f"✅ Restored {len(fix_result.get('fixed', []))} dynamic tool handler(s) from persistence", file=__import__('sys').stderr)
    
    if fix_result.get("failed"):
        print(f"⚠️  Failed to restore {len(fix_result.get('failed', []))} dynamic tool handler(s)", file=__import__('sys').stderr)
