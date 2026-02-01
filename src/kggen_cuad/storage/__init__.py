"""
Storage adapters for KGGEN-CUAD.

Provides interfaces to various databases and storage systems.
"""

from kggen_cuad.storage.postgres import PostgresAdapter, get_postgres_adapter
from kggen_cuad.storage.neo4j_adapter import Neo4jAdapter, get_neo4j_adapter
from kggen_cuad.storage.qdrant import QdrantAdapter, get_qdrant_adapter
from kggen_cuad.storage.redis_cache import RedisCache, get_redis_cache

__all__ = [
    "PostgresAdapter",
    "get_postgres_adapter",
    "Neo4jAdapter",
    "get_neo4j_adapter",
    "QdrantAdapter",
    "get_qdrant_adapter",
    "RedisCache",
    "get_redis_cache",
]
