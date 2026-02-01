"""
Configuration management for KGGEN-CUAD.

Loads settings from environment variables with sensible defaults.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================================
    # Environment
    # ==========================================================================
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"

    # ==========================================================================
    # LLM API Keys
    # ==========================================================================
    anthropic_api_key: str = Field(default="", description="Anthropic API key for Claude")
    openai_api_key: str = Field(default="", description="OpenAI API key for GPT-4o")

    # ==========================================================================
    # LLM Configuration
    # ==========================================================================
    primary_llm_model: str = "claude-sonnet-4-20250514"
    primary_llm_provider: Literal["anthropic", "openai"] = "anthropic"
    fallback_llm_model: str = "gpt-4o"
    fallback_llm_provider: Literal["anthropic", "openai"] = "openai"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096
    llm_timeout: int = 60

    # ==========================================================================
    # PostgreSQL
    # ==========================================================================
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "kggen"
    postgres_password: str = "kggen_dev_password"
    postgres_db: str = "kggen_cuad"
    database_url: str | None = None

    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_sync_url(self) -> str:
        """Get synchronous PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ==========================================================================
    # Neo4j
    # ==========================================================================
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "kggen_dev_password"

    # ==========================================================================
    # Redis
    # ==========================================================================
    redis_url: str = "redis://localhost:6379/0"

    # ==========================================================================
    # Qdrant
    # ==========================================================================
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_collection: str = "kggen_embeddings"

    # ==========================================================================
    # Elasticsearch
    # ==========================================================================
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "kggen_triples"

    # ==========================================================================
    # API Configuration
    # ==========================================================================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # ==========================================================================
    # Embedding Configuration
    # ==========================================================================
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ==========================================================================
    # Pipeline Configuration
    # ==========================================================================
    # Extraction
    extraction_batch_size: int = 5
    extraction_max_retries: int = 3

    # Resolution (from KGGen paper)
    resolution_cluster_size: int = 128  # k=128 for clustering
    resolution_retrieval_k: int = 16  # k=16 for retrieval
    resolution_similarity_threshold: float = 0.85

    # Search
    search_bm25_weight: float = 0.5
    search_semantic_weight: float = 0.5
    search_top_k: int = 10
    search_expansion_hops: int = 2

    # ==========================================================================
    # Storage Paths
    # ==========================================================================
    contracts_dir: Path = Path("./cuad_contracts")
    cache_dir: Path = Path("./cache")
    logs_dir: Path = Path("./logs")

    @field_validator("contracts_dir", "cache_dir", "logs_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v

    # ==========================================================================
    # Entity Types (from PRD schema)
    # ==========================================================================
    @property
    def entity_types(self) -> list[str]:
        """Valid entity types for the knowledge graph."""
        return [
            "Party",
            "IPAsset",
            "Obligation",
            "Restriction",
            "LiabilityProvision",
            "Temporal",
            "Jurisdiction",
            "ContractClause",
        ]

    # ==========================================================================
    # Predicate Types (from PRD schema)
    # ==========================================================================
    @property
    def predicate_types(self) -> list[str]:
        """Valid predicate/relationship types for the knowledge graph."""
        return [
            "LICENSES_TO",
            "OWNS",
            "ASSIGNS",
            "HAS_OBLIGATION",
            "SUBJECT_TO_RESTRICTION",
            "HAS_LIABILITY",
            "GOVERNED_BY",
            "CONTAINS_CLAUSE",
            "EFFECTIVE_ON",
            "TERMINATES_ON",
        ]

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for dir_path in [self.contracts_dir, self.cache_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
