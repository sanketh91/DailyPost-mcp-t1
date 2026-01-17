"""
Weaviate MCP Tools for DailyPost Database

Version: 3.0.0 (Cloud Run hardened)
- Persistent Weaviate client singleton with healthchecks + reconnect-on-failure
- Embedding TTL cache (in-memory) + graceful fallback to BM25 keyword/hybrid-without-vector
- Hard request timeouts to prevent Claude Desktop hangs
- Structured JSON logging for Cloud Run observability
- Input validation + safe limits

NOTE:
- Do NOT close the Weaviate client per tool call. That defeats pooling and causes reconnect churn.
- Close only on process shutdown (optional).
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
import threading
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Tuple
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import weaviate
from weaviate.classes.query import Filter, MetadataQuery, Sort
from weaviate.classes.backup import BackupStorage
from sentence_transformers import SentenceTransformer

# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class Config:
    # Validation / limits
    MAX_QUERY_CHARS: int = int(os.getenv("MAX_QUERY_CHARS", "500"))
    MAX_LIMIT: int = int(os.getenv("MAX_LIMIT", "50"))
    DEFAULT_LIMIT: int = int(os.getenv("DEFAULT_LIMIT", "10"))

    # Timeouts (seconds)
    TOOL_TIMEOUT_S: float = float(os.getenv("TOOL_TIMEOUT_S", "30"))
    EMBED_TIMEOUT_S: float = float(os.getenv("EMBED_TIMEOUT_S", "8"))
    WEAVIATE_QUERY_TIMEOUT_S: float = float(os.getenv("WEAVIATE_QUERY_TIMEOUT_S", "20"))

    # Weaviate health / reconnect
    HEALTHCHECK_INTERVAL_S: float = float(os.getenv("WEAVIATE_HEALTHCHECK_INTERVAL_S", "300"))
    CONNECT_MAX_RETRIES: int = int(os.getenv("WEAVIATE_CONNECT_MAX_RETRIES", "3"))
    CONNECT_BACKOFF_BASE_S: float = float(os.getenv("WEAVIATE_CONNECT_BACKOFF_BASE_S", "1.0"))
    CONNECT_BACKOFF_MULT: float = float(os.getenv("WEAVIATE_CONNECT_BACKOFF_MULT", "2.0"))
    RECONNECT_ON_FAILURE: bool = os.getenv("WEAVIATE_RECONNECT_ON_FAILURE", "1") == "1"

    # Weaviate client timeouts (driver-level)
    WEAVIATE_INIT_TIMEOUT_S: int = int(os.getenv("WEAVIATE_INIT_TIMEOUT_S", "30"))
    WEAVIATE_QUERY_TIMEOUT_DRIVER_S: int = int(os.getenv("WEAVIATE_QUERY_TIMEOUT_DRIVER_S", "60"))
    WEAVIATE_INSERT_TIMEOUT_S: int = int(os.getenv("WEAVIATE_INSERT_TIMEOUT_S", "120"))

    # Embeddings
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")
    MODEL_DIMENSION: int = int(os.getenv("MODEL_DIMENSION", "768"))
    EAGER_LOAD_EMBEDDINGS: bool = os.getenv("EAGER_LOAD_EMBEDDINGS", "1") == "1"

    # Embedding cache
    EMB_CACHE_MAXSIZE: int = int(os.getenv("EMB_CACHE_MAXSIZE", "2048"))
    EMB_CACHE_TTL_S: float = float(os.getenv("EMB_CACHE_TTL_S", "3600"))

    # Ranking / filtering
    DYNAMIC_SCORE_FILTER: bool = os.getenv("DYNAMIC_SCORE_FILTER", "1") == "1"
    DYNAMIC_SCORE_RATIO: float = float(os.getenv("DYNAMIC_SCORE_RATIO", "0.83"))  # keep >= top_score * ratio
    MIN_SCORE_FLOOR: float = float(os.getenv("MIN_SCORE_FLOOR", "0.55"))  # never keep below this if score exists

    # Collections
    POST_COLLECTION: str = os.getenv("POST_COLLECTION", "Post")
    CHUNK_COLLECTION: str = os.getenv("CHUNK_COLLECTION", "Chunk")


CFG = Config()


# ============================================================
# Structured logging (Cloud Run friendly)
# ============================================================

class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "time": datetime.utcnow().isoformat() + "Z",
        }
        # Attach structured extras if present
        if hasattr(record, "props") and isinstance(record.props, dict):
            payload.update(record.props)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=json_safe_default)


logger = logging.getLogger("mcp.tools")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


def log_event(level: str, msg: str, **props: Any) -> None:
    level = level.upper()
    if level == "DEBUG":
        logger.debug(msg, extra={"props": props})
    elif level == "WARNING":
        logger.warning(msg, extra={"props": props})
    elif level == "ERROR":
        logger.error(msg, extra={"props": props})
    else:
        logger.info(msg, extra={"props": props})


# ============================================================
# Common helpers
# ============================================================

def json_safe_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, uuid.UUID):
        return str(obj)
    return str(obj)


def _now_s() -> float:
    return time.time()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _clamp_limit(limit: int) -> int:
    try:
        limit_int = int(limit)
    except Exception:
        limit_int = CFG.DEFAULT_LIMIT
    return max(1, min(limit_int, CFG.MAX_LIMIT))


def _validate_query(query: str) -> str:
    if query is None:
        raise ValueError("query must not be null")
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    q = query.strip()
    if not q:
        raise ValueError("query must not be empty")
    if len(q) > CFG.MAX_QUERY_CHARS:
        raise ValueError(f"query too long (max {CFG.MAX_QUERY_CHARS} chars)")
    return q


def _format_date(date_input: Any) -> Optional[str]:
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
    return datetime.strptime(date_str, "%Y-%m-%d")


# ============================================================
# Hard timeouts (prevents Claude Desktop hangs)
# ============================================================

_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("TOOL_EXECUTOR_WORKERS", "8")))


class ToolTimeout(Exception):
    pass


def run_with_timeout(fn: Callable[[], Any], timeout_s: float, op: str, request_id: str) -> Any:
    future = _EXECUTOR.submit(fn)
    try:
        return future.result(timeout=timeout_s)
    except FuturesTimeoutError:
        log_event("ERROR", "Operation timed out", request_id=request_id, op=op, timeout_s=timeout_s)
        raise ToolTimeout(f"{op} timed out after {timeout_s}s")


# ============================================================
# Embedding cache (TTL + maxsize)
# ============================================================

class TTLCache:
    def __init__(self, maxsize: int, ttl_s: float):
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        now = _now_s()
        with self._lock:
            item = self._data.get(key)
            if not item:
                self.misses += 1
                return None
            expires_at, value = item
            if expires_at <= now:
                self._data.pop(key, None)
                self.misses += 1
                return None
            # LRU bump
            self._data.move_to_end(key, last=True)
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        now = _now_s()
        with self._lock:
            self._data[key] = (now + self.ttl_s, value)
            self._data.move_to_end(key, last=True)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total else 0.0
            return {"size": len(self._data), "hits": self.hits, "misses": self.misses, "hit_rate": round(hit_rate, 4)}


_EMB_CACHE = TTLCache(maxsize=CFG.EMB_CACHE_MAXSIZE, ttl_s=CFG.EMB_CACHE_TTL_S)


# ============================================================
# Embedding model manager (eager-load optional, graceful failure)
# ============================================================

class EmbeddingManager:
    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
        self._lock = threading.Lock()
        self._load_attempted = False
        self._load_error: Optional[str] = None

    def ensure_loaded(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_attempted and self._model is None:
                raise RuntimeError(self._load_error or "Embedding model failed to load previously")
            self._load_attempted = True
            t0 = _now_s()
            try:
                log_event("INFO", "Loading embedding model", model=CFG.EMBEDDING_MODEL_NAME)
                self._model = SentenceTransformer(CFG.EMBEDDING_MODEL_NAME)
                log_event("INFO", "Embedding model loaded", model=CFG.EMBEDDING_MODEL_NAME, load_ms=int((_now_s() - t0) * 1000))
                return self._model
            except Exception as e:
                self._load_error = str(e)
                self._model = None
                log_event("ERROR", "Embedding model load failed", model=CFG.EMBEDDING_MODEL_NAME, error=str(e))
                raise

    def embed(self, text: str, request_id: str) -> Optional[List[float]]:
        """
        Returns embedding or None if embedding fails (caller can fallback to keyword/BM25).
        """
        normalized = text.strip()
        cache_key = f"{CFG.EMBEDDING_MODEL_NAME}:{_sha(normalized.lower())}"
        cached = _EMB_CACHE.get(cache_key)
        if cached is not None:
            log_event("DEBUG", "Embedding cache hit", request_id=request_id, cache_key=cache_key, cache=_EMB_CACHE.stats())
            return cached

        def _do_embed():
            model = self.ensure_loaded()
            vec = model.encode(normalized)
            return vec.tolist()

        t0 = _now_s()
        try:
            vec_list = run_with_timeout(_do_embed, CFG.EMBED_TIMEOUT_S, op="embed", request_id=request_id)
            _EMB_CACHE.set(cache_key, vec_list)
            log_event("INFO", "Embedding computed", request_id=request_id, embed_ms=int((_now_s() - t0) * 1000), cache=_EMB_CACHE.stats())
            return vec_list
        except Exception as e:
            log_event("WARNING", "Embedding failed; will fallback", request_id=request_id, error=str(e))
            return None


EMBEDDINGS = EmbeddingManager()


# Optional eager load (startup warm)
if CFG.EAGER_LOAD_EMBEDDINGS:
    def _warm():
        try:
            EMBEDDINGS.ensure_loaded()
        except Exception:
            # already logged; do not crash container
            pass

    threading.Thread(target=_warm, daemon=True).start()


# ============================================================
# Weaviate client singleton + reconnect logic
# ============================================================

_weaviate_client: Optional[Any] = None
_weaviate_lock = threading.Lock()
_weaviate_last_healthcheck_s: float = 0.0


def _create_weaviate_client() -> Any:
    wcd_url = os.getenv("WEAVIATE_URL")
    wcd_api_key = os.getenv("WEAVIATE_API_KEY")
    if not wcd_url or not wcd_api_key:
        raise RuntimeError("WEAVIATE_URL and WEAVIATE_API_KEY environment variables must be set")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=wcd_url,
        auth_credentials=weaviate.auth.AuthApiKey(wcd_api_key),
        skip_init_checks=True,
        additional_config=weaviate.config.AdditionalConfig(
            timeout=weaviate.config.Timeout(
                init=CFG.WEAVIATE_INIT_TIMEOUT_S,
                query=CFG.WEAVIATE_QUERY_TIMEOUT_DRIVER_S,
                insert=CFG.WEAVIATE_INSERT_TIMEOUT_S,
            )
        ),
    )
    return client


def _connect_with_retries(request_id: str) -> Any:
    last_error = None
    delay = CFG.CONNECT_BACKOFF_BASE_S
    for attempt in range(1, CFG.CONNECT_MAX_RETRIES + 1):
        try:
            t0 = _now_s()
            client = _create_weaviate_client()
            client.is_ready()
            log_event("INFO", "Weaviate connected", request_id=request_id, attempt=attempt, connect_ms=int((_now_s() - t0) * 1000))
            return client
        except Exception as exc:
            last_error = exc
            log_event("ERROR", "Weaviate connect attempt failed", request_id=request_id, attempt=attempt, error=str(exc))
            if attempt < CFG.CONNECT_MAX_RETRIES:
                time.sleep(delay)
                delay *= CFG.CONNECT_BACKOFF_MULT
    raise RuntimeError(f"Unable to connect to Weaviate after {CFG.CONNECT_MAX_RETRIES} attempts: {last_error}")


def _is_client_alive(client: Any) -> bool:
    try:
        client.is_ready()
        return True
    except Exception:
        return False


def get_weaviate_client(request_id: str) -> Any:
    """
    Persistent singleton client. Reconnects if client is None/stale.
    """
    global _weaviate_client, _weaviate_last_healthcheck_s
    with _weaviate_lock:
        # If client exists, optionally healthcheck
        if _weaviate_client is not None:
            now = _now_s()
            if (now - _weaviate_last_healthcheck_s) >= CFG.HEALTHCHECK_INTERVAL_S:
                alive = _is_client_alive(_weaviate_client)
                _weaviate_last_healthcheck_s = now
                if alive:
                    return _weaviate_client
                # stale -> drop it and reconnect
                try:
                    _weaviate_client.close()
                except Exception:
                    pass
                _weaviate_client = None

            return _weaviate_client

        # No client -> connect
        _weaviate_client = _connect_with_retries(request_id=request_id)
        _weaviate_last_healthcheck_s = _now_s()
        return _weaviate_client


def weaviate_call(op_name: str, request_id: str, fn: Callable[[Any], Any]) -> Any:
    """
    Executes a Weaviate operation with:
    - tool-level timeout
    - reconnect-on-failure (one retry)
    """
    def _attempt() -> Any:
        client = get_weaviate_client(request_id=request_id)
        return fn(client)

    t0 = _now_s()
    try:
        result = run_with_timeout(_attempt, CFG.WEAVIATE_QUERY_TIMEOUT_S, op=op_name, request_id=request_id)
        log_event("INFO", "Weaviate op ok", request_id=request_id, op=op_name, weaviate_ms=int((_now_s() - t0) * 1000))
        return result
    except Exception as e:
        log_event("WARNING", "Weaviate op failed", request_id=request_id, op=op_name, error=str(e))
        if not CFG.RECONNECT_ON_FAILURE:
            raise
        # Force reconnect once
        global _weaviate_client
        with _weaviate_lock:
            try:
                if _weaviate_client is not None:
                    _weaviate_client.close()
            except Exception:
                pass
            _weaviate_client = None
        # Retry once
        result = run_with_timeout(_attempt, CFG.WEAVIATE_QUERY_TIMEOUT_S, op=f"{op_name}.retry", request_id=request_id)
        log_event("INFO", "Weaviate op ok after retry", request_id=request_id, op=op_name, weaviate_ms=int((_now_s() - t0) * 1000))
        return result


# ============================================================
# Ranking / filtering helpers
# ============================================================

def _dedupe_results(results: List[Dict[str, Any]], key: str = "post_number") -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in results:
        k = r.get(key)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _apply_dynamic_score_filter(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not results:
        return results
    # Find top score if present
    scores = [r.get("relevance_score") for r in results if isinstance(r.get("relevance_score"), (int, float))]
    if not scores:
        return results
    top = max(scores)
    threshold = max(CFG.MIN_SCORE_FLOOR, top * CFG.DYNAMIC_SCORE_RATIO) if CFG.DYNAMIC_SCORE_FILTER else CFG.MIN_SCORE_FLOOR
    filtered = [r for r in results if not isinstance(r.get("relevance_score"), (int, float)) or r["relevance_score"] >= threshold]
    return filtered


# ============================================================
# MCP registration
# ============================================================

# mcp will be set by mcp_server.py before tools are registered
mcp = None


def register_tools():
    if mcp is None:
        raise RuntimeError("mcp instance not set. Cannot register tools.")

    # Static tools
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
    mcp.tool()(get_style_guide)
    mcp.tool()(vectorize_and_insert_post)
    mcp.tool()(delete_post)

# ============================================================
# Tools
# ============================================================

def search_posts_hybrid(
    query: str,
    limit: int = 10,
    alpha: float = 0.7,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    topic_filter: Optional[str] = None,
    include_scores: bool = True,
) -> Dict[str, Any]:
    """
    Hybrid search combining semantic and keyword matching.
    Alpha controls the balance: 0.0=pure keyword, 1.0=pure vector, 0.7=favors semantic.
    """
    request_id = f"hybrid_search_{int(time.time() * 1000)}"
    client = get_weaviate_client(request_id)
    
    try:
        try:
            query_vector = get_embedding_for_query(query)
            embedding_used = True
        except Exception as e:
            logger.warning(f"Failed to generate query vector, falling back to keyword-only: {e}")
            query_vector = None
            embedding_used = False

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
            filters.append(Filter.by_property("final_topic").equal(topic_filter))

        combined_filter = (
            filters[0] if len(filters) == 1 
            else (filters[0] & filters[1] if len(filters) == 2 else None)
        )

        # Use hybrid search if we have embeddings, otherwise fall back to keyword search
        if query_vector is not None:
            results = post_collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=alpha,
                limit=limit,
                filters=combined_filter,
                query_properties=["post_content", "post_title", "final_topic"],
                return_metadata=MetadataQuery(score=True) if include_scores else None,
                return_properties=[
                    "post_number", "post_title", "post_content", "final_topic",
                    "topic_confidence", "post_date", "secondary_topics",
                ],
            )
        else:
            # Fallback to BM25 keyword search
            results = post_collection.query.bm25(
                query=query,
                limit=limit,
                filters=combined_filter,
                query_properties=["post_content", "post_title", "final_topic"],
                return_metadata=MetadataQuery(score=True) if include_scores else None,
                return_properties=[
                    "post_number", "post_title", "post_content", "final_topic",
                    "topic_confidence", "post_date", "secondary_topics",
                ],
            )

        formatted_results = []
        for obj in results.objects:
            result = {
                "post_number": obj.properties.get("post_number"),
                "title": obj.properties.get("post_title"),
                "content": obj.properties.get("post_content", "")[:300] + "...",
                "primary_topic": obj.properties.get("final_topic"),
                "topic_confidence": obj.properties.get("topic_confidence"),
                "date": _format_date(obj.properties.get("post_date")),
                "secondary_topics": obj.properties.get("secondary_topics"),
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
                "embedding_used": embedding_used
            },
            "results": formatted_results,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "query": query}
    finally:
        if client:
            client.close()


def search_by_date_range(
    start_date: str,
    end_date: str,
    limit: int = 20,
    topic_filter: Optional[str] = None,
    sort_order: str = "desc",
) -> Dict[str, Any]:
    """
    Fetches posts whose post_date falls within a given start/end date range, optionally filtered by final topic, returning sorted metadata and short content previews.
    """
    request_id = str(uuid.uuid4())
    t0 = _now_s()
    try:
        limit = _clamp_limit(limit)
        start_dt = _parse_date_input(start_date).replace(hour=0, minute=0, second=0)
        end_dt = _parse_date_input(end_date).replace(hour=23, minute=59, second=59)
        start_iso = start_dt.isoformat() + "Z"
        end_iso = end_dt.isoformat() + "Z"

        def _op(client):
            post_collection = client.collections.get(CFG.POST_COLLECTION)
            date_filter = (
                Filter.by_property("post_date").greater_or_equal(start_iso)
                & Filter.by_property("post_date").less_or_equal(end_iso)
            )
            if topic_filter:
                date_filter = date_filter & Filter.by_property("final_topic").equal(topic_filter)

            return post_collection.query.fetch_objects(
                filters=date_filter,
                limit=limit,
                return_properties=[
                    "post_number", "post_title", "post_date", "final_topic",
                    "topic_confidence", "post_content", "secondary_topics",
                ],
            )

        results = weaviate_call("search_by_date_range", request_id=request_id, fn=_op)
        out = []
        for obj in results.objects:
            props = obj.properties or {}
            out.append({
                "post_number": props.get("post_number"),
                "title": props.get("post_title"),
                "date": _format_date(props.get("post_date")),
                "primary_topic": props.get("final_topic"),
                "topic_confidence": props.get("topic_confidence"),
                "preview": (props.get("post_content") or "")[:150] + "...",
                "secondary_topics": props.get("secondary_topics"),
            })

        out.sort(key=lambda x: (x.get("post_number") or 0), reverse=(sort_order == "desc"))

        log_event("INFO", "Tool ok", request_id=request_id, tool="search_by_date_range", total_results=len(out), total_ms=int((_now_s() - t0) * 1000))
        return {"success": True, "date_range": f"{start_date} to {end_date}", "total_results": len(out), "topic_filter": topic_filter, "results": out}
    except Exception as e:
        log_event("ERROR", "Tool error", request_id=request_id, tool="search_by_date_range", error=str(e))
        return {"success": False, "error": str(e)}


def get_post_by_id(post_number: int) -> Dict[str, Any]:
    """
    Retrieve a single post by post_number with full content, topic metadata (primary topic, confidence score, secondary topics), and all available metadata. Fast single-post lookup.
    """
    request_id = str(uuid.uuid4())
    try:
        pn = int(post_number)

        def _op(client):
            post_collection = client.collections.get(CFG.POST_COLLECTION)
            return post_collection.query.fetch_objects(
                filters=Filter.by_property("post_number").equal(pn),
                limit=1,
                return_properties=[
                    "post_number", "post_title", "post_content", "post_date",
                    "final_topic", "topic_confidence", "secondary_topics",
                    "sentence_level_explanation", "word_level_explanation",
                ],
            )

        results = weaviate_call("get_post_by_id", request_id=request_id, fn=_op)
        if not results.objects:
            return {"success": False, "error": f"Post #{pn} not found"}

        obj = results.objects[0]
        props = obj.properties or {}
        return {
            "success": True,
            "post": {
                "post_number": props.get("post_number"),
                "title": props.get("post_title"),
                "content": props.get("post_content"),
                "date": _format_date(props.get("post_date")),
                "primary_topic": props.get("final_topic"),
                "topic_confidence": props.get("topic_confidence"),
                "secondary_topics": props.get("secondary_topics"),
                "sentence_level_explanation": props.get("sentence_level_explanation"),
                "word_level_explanation": props.get("word_level_explanation"),
            },
        }
    except Exception as e:
        log_event("ERROR", "Tool error", request_id=request_id, tool="get_post_by_id", error=str(e))
        return {"success": False, "error": str(e)}


def get_posts_batch(post_numbers: List[int], include_content: bool = True) -> Dict[str, Any]:
    """
    Retrieves multiple posts in one call by a list of post_number values, with optional inclusion of full content and an explicit list of any missing IDs.
    """
    request_id = str(uuid.uuid4())
    try:
        if not isinstance(post_numbers, list) or not post_numbers:
            raise ValueError("post_numbers must be a non-empty list")

        ids = [int(x) for x in post_numbers]
        limit = _clamp_limit(len(ids))  # clamp to MAX_LIMIT (defensive)

        props = ["post_number", "post_title", "post_date", "final_topic", "topic_confidence", "secondary_topics"]
        if include_content:
            props.append("post_content")

        def _op(client):
            post_collection = client.collections.get(CFG.POST_COLLECTION)
            return post_collection.query.fetch_objects(
                filters=Filter.by_property("post_number").contains_any(ids),
                limit=limit,
                return_properties=props,
            )

        results = weaviate_call("get_posts_batch", request_id=request_id, fn=_op)
        found, found_ids = [], set()
        for obj in results.objects:
            p = obj.properties or {}
            pid = p.get("post_number")
            found_ids.add(pid)
            row = {
                "post_number": pid,
                "title": p.get("post_title"),
                "date": _format_date(p.get("post_date")),
                "primary_topic": p.get("final_topic"),
                "topic_confidence": p.get("topic_confidence"),
                "secondary_topics": p.get("secondary_topics"),
            }
            if include_content:
                row["content"] = p.get("post_content")
            found.append(row)

        missing = [pid for pid in ids if pid not in found_ids]
        return {"success": True, "requested_count": len(ids), "found_count": len(found), "missing_posts": missing, "posts": found}
    except Exception as e:
        log_event("ERROR", "Tool error", request_id=request_id, tool="get_posts_batch", error=str(e))
        return {"success": False, "error": str(e)}


def search_posts_by_topic(
    topic_name: str,
    limit: int = 15,
    fuzzy: bool = True,
    include_secondary: bool = False,
) -> Dict[str, Any]:
    """
    Finds posts matching a topic string using fuzzy or exact matching over final_topic and optionally secondary_topics , returning de‑duplicated posts and, for fuzzy mode, the set of matched topic names
    """
    request_id = str(uuid.uuid4())
    try:
        topic_name = _validate_query(topic_name)
        limit = _clamp_limit(limit)

        def _op(client):
            post_collection = client.collections.get(CFG.POST_COLLECTION)

            if fuzzy:
                # NOTE: still expensive; kept for compatibility. Consider indexing topics separately later.
                all_posts = post_collection.query.fetch_objects(
                    limit=5000,
                    return_properties=["final_topic", "secondary_topics"],
                )
                search_lower = topic_name.lower()
                matching_topics = set()
                for post in all_posts.objects:
                    props = post.properties or {}
                    final_topic = (props.get("final_topic") or "")
                    secondary_topics = (props.get("secondary_topics") or "")
                    if (search_lower in final_topic.lower()) or (search_lower in secondary_topics.lower()):
                        if final_topic:
                            matching_topics.add(final_topic)

                if not matching_topics:
                    return ("NO_MATCH", [], [])

                filters = None
                for t in matching_topics:
                    topic_filter = (
                        (Filter.by_property("final_topic").equal(t) | Filter.by_property("secondary_topics").like(t))
                        if include_secondary
                        else Filter.by_property("final_topic").equal(t)
                    )
                    filters = topic_filter if filters is None else (filters | topic_filter)

                res = post_collection.query.fetch_objects(
                    filters=filters,
                    limit=limit,
                    return_properties=[
                        "post_number", "post_title", "post_date", "final_topic",
                        "topic_confidence", "post_content", "secondary_topics",
                    ],
                )
                return ("OK", list(matching_topics), res)

            # Non-fuzzy path
            if include_secondary:
                res = post_collection.query.bm25(
                    query=topic_name,
                    query_properties=["final_topic", "secondary_topics"],
                    limit=limit,
                    return_properties=[
                        "post_number", "post_title", "post_date", "final_topic",
                        "topic_confidence", "post_content", "secondary_topics",
                    ],
                )
                return ("OK", [topic_name], res)

            res = post_collection.query.fetch_objects(
                filters=Filter.by_property("final_topic").equal(topic_name),
                limit=limit,
                return_properties=[
                    "post_number", "post_title", "post_date", "final_topic",
                    "topic_confidence", "post_content", "secondary_topics",
                ],
            )
            return ("OK", [topic_name], res)

        status, matched_topics, results = weaviate_call("search_posts_by_topic", request_id=request_id, fn=_op)
        if status == "NO_MATCH":
            return {"success": False, "error": "No matching topics found"}

        formatted = []
        for obj in results.objects:
            p = obj.properties or {}
            formatted.append({
                "post_number": p.get("post_number"),
                "title": p.get("post_title"),
                "date": _format_date(p.get("post_date")),
                "primary_topic": p.get("final_topic"),
                "topic_confidence": p.get("topic_confidence"),
                "preview": (p.get("post_content") or "")[:150] + "...",
                "secondary_topics": p.get("secondary_topics"),
                "is_secondary": include_secondary and (p.get("final_topic") != topic_name),
            })

        formatted = _dedupe_results(formatted, key="post_number")
        resp = {
            "success": True,
            "topic_search": topic_name,
            "fuzzy_match": fuzzy,
            "total_results": len(formatted),
            "results": formatted,
        }
        if fuzzy:
            resp["matched_topics"] = matched_topics
        return resp
    except Exception as e:
        log_event("ERROR", "Tool error", request_id=request_id, tool="search_posts_by_topic", error=str(e))
        return {"success": False, "error": str(e)}


def get_topic_statistics(top_n: int = 15, include_distribution: bool = True) -> Dict[str, Any]:
    """
    Analyze topic distribution across posts including: unique topic counts, top N topics (default 15) with counts/percentages, and optional distribution analysis showing how many topics are tagged per post.
    """
    request_id = str(uuid.uuid4())
    # Entry log to confirm FastMCP actually invoked this tool
    log_event(
        "INFO",
        "Tool entered",
        request_id=request_id,
        tool="get_topic_statistics",
        raw_args={"top_n": top_n, "include_distribution": include_distribution},
    )
    try:
        top_n = max(1, min(int(top_n), 100))

        def _op(client):
            post_collection = client.collections.get(CFG.POST_COLLECTION)
            return post_collection.query.fetch_objects(
                limit=10000,
                return_properties=["final_topic", "secondary_topics"],
            )

        results = weaviate_call("get_topic_statistics", request_id=request_id, fn=_op)

        topic_counts = defaultdict(int)
        multi_label_count = 0
        dist = defaultdict(int)

        for item in results.objects:
            p = item.properties or {}
            topic = p.get("final_topic") or "Unknown"
            topic_counts[topic] += 1
            secondary = p.get("secondary_topics") or ""
            if secondary:
                multi_label_count += 1
            if include_distribution:
                num_topics = 1 + (len([x for x in secondary.split(",") if x.strip()]) if secondary else 0)
                label = f"{num_topics}_topic{'s' if num_topics != 1 else ''}" if num_topics < 4 else "4+_topics"
                dist[label] += 1

        total_posts = sum(topic_counts.values())
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_topics = []
        for name, count in sorted_topics:
            top_topics.append({
                "topic_name": name,
                "post_count": count,
                "percentage": round((count / total_posts) * 100, 2) if total_posts else 0,
            })

        resp = {
            "success": True,
            "statistics": {
                "total_posts": total_posts,
                "unique_topics": len(topic_counts),
                "multi_label_posts": multi_label_count,
                "multi_label_percentage": round((multi_label_count / total_posts) * 100, 2) if total_posts else 0,
                "avg_posts_per_topic": round(total_posts / len(topic_counts), 2) if topic_counts else 0,
            },
            "top_topics": top_topics,
        }
        if include_distribution:
            resp["distribution"] = dict(dist)
        return resp
    except Exception as e:
        log_event("ERROR", "Tool error", request_id=request_id, tool="get_topic_statistics", error=str(e))
        return {"success": False, "error": str(e)}


def find_similar_posts(post_number: int, limit: int = 5, min_similarity: float = 0.7) -> Dict[str, Any]:
    """
    Given a reference post_number , uses its stored vector to run a near_vector search and return the most similar posts with similarity scores, excluding the reference itself.
    """
    request_id = str(uuid.uuid4())
    # Entry log to confirm FastMCP actually invoked this tool
    log_event(
        "INFO",
        "Tool entered",
        request_id=request_id,
        tool="find_similar_posts",
        raw_args={"post_number": post_number, "limit": limit, "min_similarity": min_similarity},
    )
    try:
        pn = int(post_number)
        limit = _clamp_limit(limit)
        min_similarity = float(min_similarity)

        def _op(client):
            post_collection = client.collections.get(CFG.POST_COLLECTION)
            query_opts = {"filters": Filter.by_property("post_number").equal(pn), "limit": 1, "return_properties": ["post_number", "post_title", "final_topic", "topic_confidence"], "include_vector": True}
            ref = post_collection.query.fetch_objects(**query_opts)
            return post_collection, ref

        post_collection, ref = weaviate_call("find_similar_posts.ref", request_id=request_id, fn=_op)
        if not ref.objects:
            return {"success": False, "error": f"Reference post #{pn} not found"}

        ref_obj = ref.objects[0]
        ref_props = ref_obj.properties or {}
        # V4 client: vector is attribute of object, not in properties
        ref_embedding = ref_obj.vector.get("default") if isinstance(ref_obj.vector, dict) else ref_obj.vector
        
        if not ref_embedding:
             return {"success": False, "error": f"Post #{pn} has no embedding"}

        def _op2(client):
            pc = client.collections.get(CFG.POST_COLLECTION)
            return pc.query.near_vector(
                near_vector=ref_embedding,
                limit=limit + 1,
                return_metadata=MetadataQuery(distance=True),
                return_properties=["post_number", "post_title", "final_topic", "topic_confidence", "post_content"],
            )

        similar = weaviate_call("find_similar_posts.near_vector", request_id=request_id, fn=_op2)

        similar_posts = []
        for obj in similar.objects:
            p = obj.properties or {}
            if p.get("post_number") == pn:
                continue
            distance = getattr(obj.metadata, "distance", None)
            similarity = (1 - distance) if isinstance(distance, (int, float)) else 0.0
            if similarity >= min_similarity:
                similar_posts.append({
                    "post_number": p.get("post_number"),
                    "title": p.get("post_title"),
                    "primary_topic": p.get("final_topic"),
                    "topic_confidence": p.get("topic_confidence"),
                    "similarity_score": round(similarity, 4),
                    "preview": (p.get("post_content") or "")[:150] + "...",
                })

        similar_posts = _dedupe_results(similar_posts, key="post_number")
        return {
            "success": True,
            "reference_post": {
                "post_number": ref_props.get("post_number"),
                "title": ref_props.get("post_title"),
                "primary_topic": ref_props.get("final_topic"),
                "topic_confidence": ref_props.get("topic_confidence"),
            },
            "similar_posts": similar_posts[:limit],
        }
    except Exception as e:
        log_event("ERROR", "Tool error", request_id=request_id, tool="find_similar_posts", error=str(e))
        return {"success": False, "error": str(e)}


def search_by_keyword(
    keyword: str,
    search_in: List[str] = None,
    limit: int = 10,
    exact_match: bool = False,
) -> Dict[str, Any]:
    """
    Runs a pure BM25 keyword search over selected fields (content, title, topic), returning de‑duplicated hits with previews and a short match context snippet around the first occurrence of the keyword.
    """
    request_id = str(uuid.uuid4())
    try:
        keyword = _validate_query(keyword)
        limit = _clamp_limit(limit)
        if search_in is None:
            search_in = ["content", "title"]

        field_map = {"content": "post_content", "title": "post_title", "topic": "final_topic"}
        query_properties = [field_map[f] for f in search_in if f in field_map]
        if not query_properties:
            query_properties = ["post_content", "post_title"]

        def _op(client):
            post_collection = client.collections.get(CFG.POST_COLLECTION)
            return post_collection.query.bm25(
                query=keyword,
                query_properties=query_properties,
                limit=limit,
                return_properties=["post_number", "post_title", "post_content", "final_topic", "topic_confidence", "post_date"],
            )

        results = weaviate_call("search_by_keyword", request_id=request_id, fn=_op)

        formatted = []
        for obj in results.objects:
            p = obj.properties or {}
            content = p.get("post_content") or ""
            idx = content.lower().find(keyword.lower())
            context = (
                "..." + content[max(0, idx - 50): min(len(content), idx + len(keyword) + 50)] + "..."
                if idx != -1 else (content[:150] + "...")
            )
            formatted.append({
                "post_number": p.get("post_number"),
                "title": p.get("post_title"),
                "primary_topic": p.get("final_topic"),
                "topic_confidence": p.get("topic_confidence"),
                "date": _format_date(p.get("post_date")),
                "preview": content[:200] + "...",
                "match_context": context,
            })

        formatted = _dedupe_results(formatted, key="post_number")
        return {"success": True, "keyword": keyword, "search_fields": search_in, "total_results": len(formatted), "results": formatted}
    except Exception as e:
        log_event("ERROR", "Tool error", request_id=request_id, tool="search_by_keyword", error=str(e))
        return {"success": False, "error": str(e)}


def list_all_topics(sort_by: str = "count", min_posts: int = 1) -> Dict[str, Any]:
    """
    Scans all posts to build a frequency table of final_topic values, then returns topics meeting a minimum post threshold sorted by count or name, including per‑topic percentages.
    """
    request_id = str(uuid.uuid4())
    try:
        min_posts = max(1, int(min_posts))
        sort_by = sort_by if sort_by in ("count", "name") else "count"

        def _op(client):
            post_collection = client.collections.get(CFG.POST_COLLECTION)
            return post_collection.query.fetch_objects(limit=10000, return_properties=["final_topic"])

        results = weaviate_call("list_all_topics", request_id=request_id, fn=_op)

        counts = defaultdict(int)
        for item in results.objects:
            p = item.properties or {}
            t = p.get("final_topic") or "Unknown"
            counts[t] += 1

        filtered = [(n, c) for n, c in counts.items() if c >= min_posts]
        filtered.sort(key=lambda x: x[0] if sort_by == "name" else x[1], reverse=(sort_by == "count"))

        total_posts = sum(counts.values())
        topics = [{"topic_name": n, "post_count": c, "percentage": round((c / total_posts) * 100, 2) if total_posts else 0} for n, c in filtered]
        return {"success": True, "total_topics": len(topics), "topics": topics}
    except Exception as e:
        log_event("ERROR", "Tool error", request_id=request_id, tool="list_all_topics", error=str(e))
        return {"success": False, "error": str(e)}


def get_recent_posts(days: Optional[int] = None, limit: int = 20, topic_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the most recent posts (default 20, configurable via limit), optionally filtered to a time window in days and/or specific topic. Results include "days_ago" field and content previews, sorted by date.
    """
    request_id = str(uuid.uuid4())
    try:
        limit = _clamp_limit(limit)
        combined_filter = None

        if days is not None:
            days = int(days)
            threshold_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            threshold_iso = threshold_date + "T00:00:00Z"
            combined_filter = Filter.by_property("post_date").greater_or_equal(threshold_iso)

        if topic_filter:
            f2 = Filter.by_property("final_topic").equal(topic_filter)
            combined_filter = (combined_filter & f2) if combined_filter is not None else f2

        def _op(client):
            post_collection = client.collections.get(CFG.POST_COLLECTION)
            kwargs = {
                "limit": limit,
                "return_properties": [
                    "post_number", "post_title", "post_date", "final_topic",
                    "topic_confidence", "post_content", "secondary_topics",
                ],
                "sort": Sort.by_property(name="post_number", ascending=False),
            }
            if combined_filter is not None:
                kwargs["filters"] = combined_filter
            return post_collection.query.fetch_objects(**kwargs)

        results = weaviate_call("get_recent_posts", request_id=request_id, fn=_op)

        formatted = []
        now = datetime.now()
        for obj in results.objects:
            p = obj.properties or {}
            post_date_iso = p.get("post_date")
            days_ago = None
            if post_date_iso:
                try:
                    post_date = datetime.fromisoformat(post_date_iso.replace("Z", "+00:00"))
                    days_ago = (now - post_date.replace(tzinfo=None)).days
                except Exception:
                    pass
            formatted.append({
                "post_number": p.get("post_number"),
                "title": p.get("post_title"),
                "date": post_date_iso,
                "primary_topic": p.get("final_topic"),
                "topic_confidence": p.get("topic_confidence"),
                "preview": (p.get("post_content") or "")[:150] + "...",
                "secondary_topics": p.get("secondary_topics"),
                "days_ago": days_ago,
            })

        period = "All time" if days is None else f"Last {days} days"
        return {"success": True, "period": period, "total_results": len(formatted), "topic_filter": topic_filter, "results": formatted}
    except Exception as e:
        log_event("ERROR", "Tool error", request_id=request_id, tool="get_recent_posts", error=str(e))
        return {"success": False, "error": str(e)}


def aggregate_posts(group_by: str = "topic", date_range: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Aggregates posts by final_topic , month, or year (based on post_date ), with an optional date range filter, returning grouped counts and percentages for each bucket.
    """
    request_id = str(uuid.uuid4())
    try:
        if group_by not in ("topic", "month", "year"):
            raise ValueError("group_by must be one of: topic, month, year")

        filters = None
        if date_range:
            start_dt = _parse_date_input(date_range["start"]).replace(hour=0, minute=0, second=0)
            end_dt = _parse_date_input(date_range["end"]).replace(hour=23, minute=59, second=59)
            filters = (
                Filter.by_property("post_date").greater_or_equal(start_dt.isoformat() + "Z")
                & Filter.by_property("post_date").less_or_equal(end_dt.isoformat() + "Z")
            )

        def _op(client):
            post_collection = client.collections.get(CFG.POST_COLLECTION)
            return post_collection.query.fetch_objects(filters=filters, limit=10000, return_properties=["final_topic", "post_date"])

        results = weaviate_call("aggregate_posts", request_id=request_id, fn=_op)

        aggs = defaultdict(int)
        for obj in results.objects:
            p = obj.properties or {}
            if group_by == "topic":
                key = p.get("final_topic") or "Unknown"
            else:
                iso = p.get("post_date")
                key = "Unknown Date"
                if iso:
                    try:
                        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                        key = dt.strftime("%Y-%m") if group_by == "month" else str(dt.year)
                    except Exception:
                        pass
            aggs[key] += 1

        total = sum(aggs.values())
        formatted = [{"group": g, "count": c, "percentage": round((c / total) * 100, 2) if total else 0} for g, c in sorted(aggs.items(), key=lambda x: x[1], reverse=True)]
        return {"success": True, "grouped_by": group_by, "total_posts": total, "aggregations": formatted}
    except Exception as e:
        log_event("ERROR", "Tool error", request_id=request_id, tool="aggregate_posts", error=str(e))
        return {"success": False, "error": str(e)}


def search_chunks(query: str, limit: int = 10, post_title: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs semantic or BM25 search over the Chunk collection’s chunk_text , returning matching chunks with post_number , chunk_number , and truncated text, and falling back to BM25 if embeddings are unavailable.
    """
    request_id = str(uuid.uuid4())
    try:
        query = _validate_query(query)
        limit = _clamp_limit(limit)

        query_vector = EMBEDDINGS.embed(query, request_id=request_id)

        def _op(client):
            chunk_collection = client.collections.get(CFG.CHUNK_COLLECTION)
            # Check if post_title is in schema before filtering (it is not in current schema)
            # filters = Filter.by_property("post_title").equal(post_title) if post_title else None
            # Schema inspection shows Chunk has: [post_number, chunk_number, chunk_text]
            # No post_title or chunk_topic.
            filters = None

            # If embedding unavailable, fallback to BM25 on chunk_text
            if query_vector is None:
                return chunk_collection.query.bm25(
                    query=query,
                    query_properties=["chunk_text"],
                    limit=limit,
                    filters=filters,
                    return_properties=["post_number", "chunk_number", "chunk_text"],
                )

            return chunk_collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                filters=filters,
                return_properties=["post_number", "chunk_number", "chunk_text"],
            )

        results = weaviate_call("search_chunks", request_id=request_id, fn=_op)

        formatted = []
        for obj in results.objects:
            p = obj.properties or {}
            formatted.append({
                "post_number": p.get("post_number"),
                # "post_title": p.get("post_title"), # not in schema
                "chunk_number": p.get("chunk_number"),
                "chunk_text": (p.get("chunk_text") or "")[:300] + "...",
                # "topic": p.get("chunk_topic"), # not in schema
            })

        formatted = _dedupe_results(formatted, key="chunk_number")
        return {"success": True, "query": query, "total_results": len(formatted), "embedding_used": query_vector is not None, "results": formatted}
    except Exception as e:
        log_event("ERROR", "Tool error", request_id=request_id, tool="search_chunks", error=str(e))
        return {"success": False, "error": str(e), "query": query}

def get_style_guide(refresh_context: bool = False) -> str:
    """
    Retrieves the Sanjay Sahay DailyPost writing style guide using a Hybrid Approach.
    """
    # --- PATH FIX START ---
    # Get the directory where THIS script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Join it with the filename to get a full, absolute path
    STYLE_GUIDE_PATH = os.path.join(current_dir, "sanjay_sahay_style.txt")
    # --- PATH FIX END ---

    request_id = str(uuid.uuid4())
    
    # Setup minimal logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("sanjay_style_tool")

    try:
        # 1. Base Layer: Read the static file
        base_content = ""
        if os.path.exists(STYLE_GUIDE_PATH):
            with open(STYLE_GUIDE_PATH, "r", encoding="utf-8") as f:
                base_content = f.read()
        else:
            # Returns the absolute path in the error so you can debug exactly where it looked
            return f"Error: Style guide file not found at {STYLE_GUIDE_PATH}"

        # 2. Hybrid Layer: Optional Dynamic Refresh
        dynamic_content = ""
        if refresh_context:
            try:
                logger.info(f"Request {request_id}: Refreshing context...")
                # Placeholder logic
                dynamic_content = "\n\n[DYNAMIC UPDATE]: Recent trends show a focus on 'AI Ethics'."
            except Exception as e:
                logger.warning(f"Refresh failed: {e}")
                dynamic_content = "\n\n[WARNING]: Could not fetch updates."

        return base_content + dynamic_content

    except Exception as e:
        logger.error(f"Error in get_style_guide: {e}")
        return f"Error processing request: {str(e)}"

# ============================================================
# NEW TOOL: Vectorize & Insert (Helpers + Main Function)
# ============================================================

def _parse_date_to_rfc3339(date_str: str) -> str:
    """
    Ensures date is in Weaviate-compliant RFC3339 format.
    Accepts: '2023-01-01', '2023/01/01', 'Jan 1 2023'.
    Returns: '2023-01-01T00:00:00Z'
    """
    if not date_str:
        # Default to now if missing to prevent error
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
    # If it already looks like RFC3339 (has T and Z), trust it
    if "T" in str(date_str) and "Z" in str(date_str):
        return str(date_str)

    try:
        # Clean up string and try basic ISO parsing (YYYY-MM-DD)
        clean_str = str(date_str).strip().split(" ")[0]
        dt = datetime.strptime(clean_str, "%Y-%m-%d")
    except ValueError:
        try:
            # Final fallback: Try pandas to_datetime if available, or just use now
            # (Note: importing pandas locally to avoid dependency issues if not at top)
            import pandas as pd
            dt = pd.to_datetime(date_str).to_pydatetime()
        except Exception:
            # Absolute failsafe: Return today's date so the insert doesn't crash
            log_event("WARNING", f"Date parsing failed for '{date_str}', using current date")
            dt = datetime.utcnow()

    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def _chunk_text(text: str, chunk_size: int = 700, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    if not text: 
        return []
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        # Move forward, ensuring we don't get stuck if overlap >= chunk_size
        step = max(1, chunk_size - overlap)
        start += step
        
    return chunks

def vectorize_and_insert_post(
    post_number: int,
    post_title: str,
    post_content: str,
    post_date: str,
    final_topic: str,
    topic_confidence: float,
    secondary_topics: str = "",
    sentence_level_explanation: str = "",
    word_level_explanation: str = "",
) -> Dict[str, Any]:
    """
    Vectorize a DailyPost (all fields combined) and insert into Weaviate.
    Combines title, content, and all explanations into a single vector.
    Strictly follows Post and Chunk schema (Weaviate v4).
    Before assigning the topics, first use list_all_topics to verify they exist, and ONLY USE THOSE TOPICS FOR THE POSTS.
    """
    # Import UUID generator locally
    from weaviate.util import generate_uuid5
    
    request_id = f"vectorize_insert_{int(time.time() * 1000)}"
    t0 = _now_s()
    
    try:
        # 1. Validate inputs
        post_number = int(post_number)
        topic_confidence = float(topic_confidence)
        
        # 2. Date Safety (Uses the helper defined ABOVE)
        formatted_date = _parse_date_to_rfc3339(post_date)

        # 3. Combine text for Post embedding
        combined_text = (
            f"{post_title} {post_content} {secondary_topics} "
            f"{sentence_level_explanation} {word_level_explanation}"
        ).strip()
        
        if not combined_text:
            raise ValueError("All text fields are empty")
        
        # 4. Generate Post Embedding
        post_vector = EMBEDDINGS.embed(combined_text, request_id=request_id)
        if post_vector is None:
            raise RuntimeError("Failed to generate embedding for post (model error)")
        
        # 5. Prepare Properties (EXCLUDING vector)
        post_properties = {
            "post_number": post_number,
            "post_title": post_title,
            "post_content": post_content,
            "post_date": formatted_date,
            "final_topic": final_topic,
            "topic_confidence": topic_confidence,
            "secondary_topics": secondary_topics,
            "sentence_level_explanation": sentence_level_explanation,
            "word_level_explanation": word_level_explanation,
        }
        
        # 6. Insert Post (Vector passed separately)
        def _insert_post(client):
            post_collection = client.collections.get(CFG.POST_COLLECTION)
            return post_collection.data.insert(
                properties=post_properties, 
                vector=post_vector
            )
        
        post_id = weaviate_call("vectorize_insert.post", request_id=request_id, fn=_insert_post)
        
        # 7. Chunking Strategy
        chunks = _chunk_text(post_content)
        chunk_ids = []
        
        for chunk_num, chunk_text in enumerate(chunks):
            # Embed the individual chunk
            chunk_vector = EMBEDDINGS.embed(chunk_text, request_id=request_id)
            if chunk_vector is None:
                log_event("WARNING", "Chunk embedding failed", request_id=request_id, chunk_num=chunk_num)
                continue
            
            chunk_properties = {
                "post_number": post_number,
                "chunk_number": chunk_num,
                "chunk_text": chunk_text,
            }
            
            # Deterministic UUID prevents duplicate chunks
            chunk_uuid = generate_uuid5(f"{post_number}_{chunk_num}")
            
            def _insert_chunk(client):
                chunk_collection = client.collections.get(CFG.CHUNK_COLLECTION)
                return chunk_collection.data.insert(
                    properties=chunk_properties,
                    vector=chunk_vector,
                    uuid=chunk_uuid
                )
            
            chunk_id = weaviate_call(f"vectorize_insert.chunk.{chunk_num}", request_id=request_id, fn=_insert_chunk)
            chunk_ids.append(chunk_id)
        
        log_event(
            "INFO",
            "Vectorize and insert completed",
            request_id=request_id,
            post_number=post_number,
            chunks_created=len(chunk_ids),
            total_ms=int((_now_s() - t0) * 1000),
        )
        
        return {
            "success": True,
            "post_number": post_number,
            "post_id": str(post_id),
            "date_stored": formatted_date,
            "chunks_created": len(chunk_ids),
            "message": f"Successfully inserted post #{post_number} with {len(chunk_ids)} chunks",
        }
        
    except Exception as e:
        log_event("ERROR", "Vectorize insert failed", request_id=request_id, post_number=post_number, error=str(e))
        return {
            "success": False,
            "error": str(e),
            "post_number": post_number,
        }

def delete_post(post_number: int) -> Dict[str, Any]:
    """
    Deletes a post AND all its associated chunks from Weaviate by post_number.
    """
    request_id = f"delete_post_{int(time.time())}"
    # Reuse the existing persistent client from tool.py
    client = get_weaviate_client(request_id)
    
    try:
        post_number = int(post_number)
        
        # 1. Delete from 'Post' Collection
        post_collection = client.collections.get(CFG.POST_COLLECTION)
        post_res = post_collection.data.delete_many(
            where=Filter.by_property("post_number").equal(post_number)
        )

        # 2. Delete from 'Chunk' Collection
        chunk_collection = client.collections.get(CFG.CHUNK_COLLECTION)
        chunk_res = chunk_collection.data.delete_many(
            where=Filter.by_property("post_number").equal(post_number)
        )

        return {
            "success": True,
            "post_number": post_number,
            "posts_deleted": post_res.successful,
            "chunks_deleted": chunk_res.successful,
            "failed_count": post_res.failed + chunk_res.failed
        }
        
    except Exception as e:
        log_event("ERROR", "Delete post failed", request_id=request_id, error=str(e))
        return {"success": False, "error": str(e), "post_number": post_number}

# ============================================================
# Optional: controlled shutdown hook
# ============================================================

def close_weaviate_client() -> None:
    global _weaviate_client
    with _weaviate_lock:
        if _weaviate_client is not None:
            try:
                _weaviate_client.close()
            except Exception:
                pass
            _weaviate_client = None
