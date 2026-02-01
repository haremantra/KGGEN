"""
PostgreSQL database adapter using SQLAlchemy async.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from typing import Any, AsyncGenerator
from uuid import UUID

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from kggen_cuad.config import get_settings
from kggen_cuad.models.contract import Contract, ContractStatus

logger = structlog.get_logger(__name__)


class PostgresAdapter:
    """
    PostgreSQL database adapter.

    Handles all database operations for contracts, extractions, triples, etc.
    """

    def __init__(self, database_url: str | None = None):
        settings = get_settings()
        self.database_url = database_url or settings.postgres_url

        self.engine = create_async_engine(
            self.database_url,
            echo=settings.debug,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )

        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session."""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def close(self) -> None:
        """Close database connections."""
        await self.engine.dispose()

    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error("postgres_health_check_failed", error=str(e))
            return False

    # =========================================================================
    # Contract Operations
    # =========================================================================

    async def create_contract(self, contract: Contract) -> Contract:
        """Create a new contract record."""
        async with self.session() as session:
            await session.execute(
                text("""
                    INSERT INTO contracts (
                        id, cuad_id, filename, contract_type, jurisdiction,
                        raw_text, page_count, word_count, status, created_at
                    ) VALUES (
                        :id, :cuad_id, :filename, :contract_type, :jurisdiction,
                        :raw_text, :page_count, :word_count, :status, :created_at
                    )
                """),
                {
                    "id": str(contract.id),
                    "cuad_id": contract.cuad_id,
                    "filename": contract.filename,
                    "contract_type": contract.contract_type,
                    "jurisdiction": contract.jurisdiction,
                    "raw_text": contract.raw_text,
                    "page_count": contract.page_count,
                    "word_count": contract.word_count,
                    "status": contract.status.value,
                    "created_at": contract.created_at,
                },
            )
        logger.info("contract_created", contract_id=str(contract.id))
        return contract

    async def get_contract(self, contract_id: UUID) -> Contract | None:
        """Get a contract by ID."""
        async with self.session() as session:
            result = await session.execute(
                text("SELECT * FROM contracts WHERE id = :id"),
                {"id": str(contract_id)},
            )
            row = result.mappings().fetchone()
            if row:
                return self._row_to_contract(row)
            return None

    async def get_contract_by_cuad_id(self, cuad_id: str) -> Contract | None:
        """Get a contract by CUAD ID."""
        async with self.session() as session:
            result = await session.execute(
                text("SELECT * FROM contracts WHERE cuad_id = :cuad_id"),
                {"cuad_id": cuad_id},
            )
            row = result.mappings().fetchone()
            if row:
                return self._row_to_contract(row)
            return None

    async def list_contracts(
        self,
        status: ContractStatus | None = None,
        contract_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Contract]:
        """List contracts with optional filters."""
        query = "SELECT * FROM contracts WHERE 1=1"
        params: dict[str, Any] = {}

        if status:
            query += " AND status = :status"
            params["status"] = status.value

        if contract_type:
            query += " AND contract_type = :contract_type"
            params["contract_type"] = contract_type

        query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
        params["limit"] = limit
        params["offset"] = offset

        async with self.session() as session:
            result = await session.execute(text(query), params)
            rows = result.mappings().fetchall()
            return [self._row_to_contract(row) for row in rows]

    async def update_contract_status(
        self,
        contract_id: UUID,
        status: ContractStatus,
        error_message: str | None = None,
    ) -> None:
        """Update contract processing status."""
        update_fields = ["status = :status", "updated_at = :updated_at"]
        params: dict[str, Any] = {
            "id": str(contract_id),
            "status": status.value,
            "updated_at": datetime.utcnow(),
        }

        if error_message:
            update_fields.append("error_message = :error_message")
            params["error_message"] = error_message

        # Set timestamp based on status
        if status == ContractStatus.EXTRACTED:
            update_fields.append("extracted_at = :extracted_at")
            params["extracted_at"] = datetime.utcnow()
        elif status == ContractStatus.AGGREGATED:
            update_fields.append("aggregated_at = :aggregated_at")
            params["aggregated_at"] = datetime.utcnow()
        elif status == ContractStatus.RESOLVED:
            update_fields.append("resolved_at = :resolved_at")
            params["resolved_at"] = datetime.utcnow()

        query = f"UPDATE contracts SET {', '.join(update_fields)} WHERE id = :id"

        async with self.session() as session:
            await session.execute(text(query), params)

        logger.info(
            "contract_status_updated",
            contract_id=str(contract_id),
            status=status.value,
        )

    async def count_contracts(self, status: ContractStatus | None = None) -> int:
        """Count contracts with optional status filter."""
        query = "SELECT COUNT(*) FROM contracts"
        params: dict[str, Any] = {}

        if status:
            query += " WHERE status = :status"
            params["status"] = status.value

        async with self.session() as session:
            result = await session.execute(text(query), params)
            return result.scalar() or 0

    # =========================================================================
    # Triple Operations
    # =========================================================================

    async def save_triples(
        self,
        triples: list[dict[str, Any]],
        contract_id: UUID,
    ) -> int:
        """Save multiple triples to the database."""
        if not triples:
            return 0

        async with self.session() as session:
            for triple in triples:
                await session.execute(
                    text("""
                        INSERT INTO triples (
                            id, contract_id, subject_text, predicate, object_text,
                            properties, cuad_label, confidence_score, source_text
                        ) VALUES (
                            :id, :contract_id, :subject_text, :predicate, :object_text,
                            :properties, :cuad_label, :confidence_score, :source_text
                        )
                    """),
                    {
                        "id": triple.get("id", str(UUID())),
                        "contract_id": str(contract_id),
                        "subject_text": triple["subject"],
                        "predicate": triple["predicate"],
                        "object_text": triple["object"],
                        "properties": triple.get("properties", {}),
                        "cuad_label": triple.get("cuad_label"),
                        "confidence_score": triple.get("confidence", 1.0),
                        "source_text": triple.get("source_text"),
                    },
                )

        logger.info(
            "triples_saved",
            contract_id=str(contract_id),
            count=len(triples),
        )
        return len(triples)

    async def get_triples(
        self,
        contract_id: UUID | None = None,
        predicate: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get triples with optional filters."""
        query = "SELECT * FROM triples WHERE 1=1"
        params: dict[str, Any] = {}

        if contract_id:
            query += " AND contract_id = :contract_id"
            params["contract_id"] = str(contract_id)

        if predicate:
            query += " AND predicate = :predicate"
            params["predicate"] = predicate

        query += " LIMIT :limit"
        params["limit"] = limit

        async with self.session() as session:
            result = await session.execute(text(query), params)
            rows = result.mappings().fetchall()
            return [dict(row) for row in rows]

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def save_entities(
        self,
        entities: list[dict[str, Any]],
        contract_id: UUID,
    ) -> int:
        """Save multiple entities to the database."""
        if not entities:
            return 0

        async with self.session() as session:
            for entity in entities:
                await session.execute(
                    text("""
                        INSERT INTO entities (
                            id, contract_id, name, entity_type, properties,
                            source_text, confidence_score
                        ) VALUES (
                            :id, :contract_id, :name, :entity_type, :properties,
                            :source_text, :confidence_score
                        )
                    """),
                    {
                        "id": entity.get("id", str(UUID())),
                        "contract_id": str(contract_id),
                        "name": entity["name"],
                        "entity_type": entity["type"],
                        "properties": entity.get("properties", {}),
                        "source_text": entity.get("source_text"),
                        "confidence_score": entity.get("confidence", 1.0),
                    },
                )

        logger.info(
            "entities_saved",
            contract_id=str(contract_id),
            count=len(entities),
        )
        return len(entities)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _row_to_contract(self, row: dict[str, Any]) -> Contract:
        """Convert database row to Contract model."""
        return Contract(
            id=UUID(row["id"]) if isinstance(row["id"], str) else row["id"],
            cuad_id=row["cuad_id"],
            filename=row["filename"],
            contract_type=row.get("contract_type"),
            jurisdiction=row.get("jurisdiction"),
            raw_text=row.get("raw_text"),
            page_count=row.get("page_count"),
            word_count=row.get("word_count"),
            status=ContractStatus(row["status"]),
            error_message=row.get("error_message"),
            created_at=row["created_at"],
            updated_at=row.get("updated_at", row["created_at"]),
            extracted_at=row.get("extracted_at"),
            aggregated_at=row.get("aggregated_at"),
            resolved_at=row.get("resolved_at"),
        )


@lru_cache()
def get_postgres_adapter() -> PostgresAdapter:
    """Get cached PostgreSQL adapter instance."""
    return PostgresAdapter()
