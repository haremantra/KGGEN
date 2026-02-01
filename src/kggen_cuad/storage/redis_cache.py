"""
Redis caching adapter.
"""

import hashlib
import json
from functools import lru_cache
from typing import Any

import structlog
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from kggen_cuad.config import get_settings

logger = structlog.get_logger(__name__)


class RedisCache:
    """
    Redis caching adapter.

    Provides caching for query results, embeddings, and other data.
    """

    def __init__(self, url: str | None = None):
        settings = get_settings()
        self.url = url or settings.redis_url
        self._client: Redis | None = None

        # Default TTLs (in seconds)
        self.default_ttl = 3600  # 1 hour
        self.query_ttl = 1800  # 30 minutes
        self.embedding_ttl = 86400  # 24 hours

    def connect(self) -> Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = Redis.from_url(
                self.url,
                decode_responses=True,
            )
            logger.info("redis_connected", url=self.url)
        return self._client

    @property
    def client(self) -> Redis:
        """Get the Redis client."""
        return self.connect()

    def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            self._client.close()
            self._client = None

    def health_check(self) -> bool:
        """Check Redis connectivity."""
        try:
            self.client.ping()
            return True
        except RedisConnectionError as e:
            logger.error("redis_health_check_failed", error=str(e))
            return False

    # =========================================================================
    # Basic Operations
    # =========================================================================

    def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except (RedisConnectionError, json.JSONDecodeError) as e:
            logger.warning("cache_get_failed", key=key, error=str(e))
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Set a value in cache."""
        try:
            serialized = json.dumps(value, default=str)
            self.client.setex(
                key,
                ttl or self.default_ttl,
                serialized,
            )
            return True
        except (RedisConnectionError, TypeError) as e:
            logger.warning("cache_set_failed", key=key, error=str(e))
            return False

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        try:
            self.client.delete(key)
            return True
        except RedisConnectionError as e:
            logger.warning("cache_delete_failed", key=key, error=str(e))
            return False

    def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        try:
            return bool(self.client.exists(key))
        except RedisConnectionError:
            return False

    # =========================================================================
    # Query Caching
    # =========================================================================

    @staticmethod
    def _make_query_key(query: str, contract_ids: list[str] | None = None) -> str:
        """Generate a cache key for a query."""
        key_parts = [query]
        if contract_ids:
            key_parts.extend(sorted(contract_ids))
        key_string = "|".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        return f"query:{key_hash}"

    def get_query_result(
        self,
        query: str,
        contract_ids: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Get cached query result."""
        key = self._make_query_key(query, contract_ids)
        result = self.get(key)
        if result:
            logger.debug("query_cache_hit", query=query[:50])
            # Increment hit counter
            self.client.incr(f"{key}:hits")
        return result

    def set_query_result(
        self,
        query: str,
        result: dict[str, Any],
        contract_ids: list[str] | None = None,
        ttl: int | None = None,
    ) -> bool:
        """Cache a query result."""
        key = self._make_query_key(query, contract_ids)
        return self.set(key, result, ttl or self.query_ttl)

    def invalidate_query_cache(self, pattern: str = "query:*") -> int:
        """Invalidate query cache entries matching pattern."""
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except RedisConnectionError as e:
            logger.warning("cache_invalidate_failed", pattern=pattern, error=str(e))
            return 0

    # =========================================================================
    # Embedding Caching
    # =========================================================================

    @staticmethod
    def _make_embedding_key(text: str, model: str) -> str:
        """Generate a cache key for an embedding."""
        key_string = f"{model}:{text}"
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        return f"emb:{key_hash}"

    def get_embedding(
        self,
        text: str,
        model: str,
    ) -> list[float] | None:
        """Get cached embedding."""
        key = self._make_embedding_key(text, model)
        return self.get(key)

    def set_embedding(
        self,
        text: str,
        model: str,
        embedding: list[float],
        ttl: int | None = None,
    ) -> bool:
        """Cache an embedding."""
        key = self._make_embedding_key(text, model)
        return self.set(key, embedding, ttl or self.embedding_ttl)

    # =========================================================================
    # Contract Status Caching
    # =========================================================================

    def get_contract_status(self, contract_id: str) -> str | None:
        """Get cached contract processing status."""
        return self.get(f"contract:{contract_id}:status")

    def set_contract_status(
        self,
        contract_id: str,
        status: str,
        ttl: int = 300,  # 5 minutes
    ) -> bool:
        """Cache contract processing status."""
        return self.set(f"contract:{contract_id}:status", status, ttl)

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """
        Check and update rate limit.

        Returns (allowed, current_count).
        """
        try:
            current = self.client.incr(key)
            if current == 1:
                self.client.expire(key, window_seconds)
            return current <= limit, current
        except RedisConnectionError:
            return True, 0  # Allow on error

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        try:
            info = self.client.info("stats")
            return {
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "N/A"),
            }
        except RedisConnectionError:
            return {"status": "disconnected"}


@lru_cache()
def get_redis_cache() -> RedisCache:
    """Get cached Redis cache instance."""
    return RedisCache()
