"""Configuration settings for KGGEN-CUAD."""

from pydantic_settings import BaseSettings
from pydantic import Field


# Default per-entity-type dedup thresholds for aggregation
DEFAULT_AGGREGATION_THRESHOLDS = {
    "Party": 0.85,
    "Obligation": 0.78,
    "Restriction": 0.78,
    "IPAsset": 0.82,
    "Temporal": 0.90,
    "Jurisdiction": 0.90,
    "LiabilityProvision": 0.80,
    "ContractClause": 0.75,
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM API Keys
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="kggen_password", alias="NEO4J_PASSWORD")

    # PostgreSQL Configuration
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="kggen", alias="POSTGRES_USER")
    postgres_password: str = Field(default="kggen_password", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="kggen", alias="POSTGRES_DB")

    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_enabled: bool = Field(default=False, alias="REDIS_ENABLED")

    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_enabled: bool = Field(default=False, alias="QDRANT_ENABLED")

    # Application Settings
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    debug: bool = Field(default=False, alias="DEBUG")

    # LLM Settings
    default_llm_model: str = Field(
        default="claude-sonnet-4-20250514", alias="DEFAULT_LLM_MODEL"
    )
    extraction_temperature: float = Field(default=0.0, alias="EXTRACTION_TEMPERATURE")

    # LLM Provider Settings
    primary_llm_provider: str = Field(default="anthropic", alias="PRIMARY_LLM_PROVIDER")
    primary_llm_model: str = Field(default="claude-sonnet-4-20250514", alias="PRIMARY_LLM_MODEL")
    fallback_llm_provider: str = Field(default="openai", alias="FALLBACK_LLM_PROVIDER")
    fallback_llm_model: str = Field(default="gpt-4o", alias="FALLBACK_LLM_MODEL")
    llm_max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")

    # Embedding Settings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")

    # Resolution Settings
    resolution_cluster_size: int = Field(default=128, alias="RESOLUTION_CLUSTER_SIZE")
    resolution_similarity_threshold: float = Field(
        default=0.80, alias="RESOLUTION_SIMILARITY_THRESHOLD"
    )

    # Search Settings
    search_bm25_weight: float = Field(default=0.5, alias="SEARCH_BM25_WEIGHT")
    search_semantic_weight: float = Field(default=0.5, alias="SEARCH_SEMANTIC_WEIGHT")
    search_inmemory_warn_threshold: int = Field(
        default=50000, alias="SEARCH_INMEMORY_WARN_THRESHOLD"
    )
    search_inmemory_max_vectors: int = Field(
        default=100000, alias="SEARCH_INMEMORY_MAX_VECTORS"
    )

    # Aggregation Settings — per-entity-type dedup thresholds
    aggregation_thresholds: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_AGGREGATION_THRESHOLDS),
        alias="AGGREGATION_THRESHOLDS",
    )

    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def async_postgres_url(self) -> str:
        """Get async PostgreSQL connection URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
    }


settings = Settings()


def get_settings() -> Settings:
    """Get the module-level settings singleton."""
    return settings
