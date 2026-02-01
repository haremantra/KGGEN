"""
Command-line interface for KGGEN-CUAD.
"""

import asyncio
from pathlib import Path
from typing import Optional

import click
import structlog

from kggen_cuad.config import get_settings

logger = structlog.get_logger(__name__)


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """KGGEN-CUAD: Knowledge Graph Generator for Legal Contracts."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug

    if debug:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(10),
        )


# =========================================================================
# Server Commands
# =========================================================================


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool) -> None:
    """Start the API server."""
    import uvicorn

    click.echo(f"Starting KGGEN-CUAD API server on {host}:{port}")

    uvicorn.run(
        "kggen_cuad.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


# =========================================================================
# Pipeline Commands
# =========================================================================


@cli.command()
@click.argument("contract_paths", nargs=-1, type=click.Path(exists=True))
@click.option("--skip-resolution", is_flag=True, help="Skip entity resolution stage")
@click.option("--no-persist", is_flag=True, help="Don't persist to database")
@click.option("--output", "-o", type=click.Path(), help="Output file for results")
def extract(
    contract_paths: tuple[str, ...],
    skip_resolution: bool,
    no_persist: bool,
    output: Optional[str],
) -> None:
    """Extract knowledge graph from contracts."""
    if not contract_paths:
        click.echo("Error: No contract files specified", err=True)
        return

    from kggen_cuad.pipeline.orchestrator import get_pipeline_orchestrator

    async def run_extraction():
        orchestrator = get_pipeline_orchestrator()

        paths = [Path(p) for p in contract_paths]

        click.echo(f"Processing {len(paths)} contract(s)...")

        result = await orchestrator.run(
            contract_paths=paths,
            skip_resolution=skip_resolution,
            persist_graph=not no_persist,
        )

        click.echo(f"\nPipeline Status: {result.status.value}")
        click.echo(f"Entities extracted: {len(result.knowledge_graph.entities) if result.knowledge_graph else 0}")
        click.echo(f"Triples extracted: {len(result.knowledge_graph.triples) if result.knowledge_graph else 0}")

        if result.duration_seconds:
            click.echo(f"Duration: {result.duration_seconds:.2f}s")

        if result.error:
            click.echo(f"Error: {result.error}", err=True)

        if output and result.knowledge_graph:
            import json
            output_path = Path(output)
            output_data = {
                "entities": [
                    {
                        "id": str(e.id),
                        "name": e.name,
                        "type": e.entity_type.value,
                    }
                    for e in result.knowledge_graph.entities
                ],
                "triples": [
                    {
                        "id": str(t.id),
                        "subject_id": str(t.subject_id),
                        "predicate": t.predicate.value,
                        "object_id": str(t.object_id),
                    }
                    for t in result.knowledge_graph.triples
                ],
            }
            output_path.write_text(json.dumps(output_data, indent=2))
            click.echo(f"\nResults written to: {output}")

    asyncio.run(run_extraction())


@cli.command()
@click.argument("text", type=str)
@click.option("--cuad-id", required=True, help="CUAD identifier for the contract")
@click.option("--no-persist", is_flag=True, help="Don't persist to database")
def extract_text(text: str, cuad_id: str, no_persist: bool) -> None:
    """Extract knowledge graph from text input."""
    from kggen_cuad.pipeline.orchestrator import get_pipeline_orchestrator

    async def run_extraction():
        orchestrator = get_pipeline_orchestrator()

        result = await orchestrator.process_contract_text(
            text=text,
            cuad_id=cuad_id,
            persist=not no_persist,
        )

        click.echo(f"Status: {result.status.value}")
        click.echo(f"Entities: {len(result.knowledge_graph.entities) if result.knowledge_graph else 0}")
        click.echo(f"Triples: {len(result.knowledge_graph.triples) if result.knowledge_graph else 0}")

    asyncio.run(run_extraction())


# =========================================================================
# Query Commands
# =========================================================================


@cli.command()
@click.argument("question", type=str)
@click.option("--contract-id", "-c", multiple=True, help="Contract ID(s) to query")
def query(question: str, contract_id: tuple[str, ...]) -> None:
    """Ask a question about contracts."""
    from uuid import UUID
    from kggen_cuad.services.query_service import get_query_service

    async def run_query():
        service = get_query_service()

        contract_ids = [UUID(c) for c in contract_id] if contract_id else None

        response = await service.answer_query(
            query=question,
            contract_ids=contract_ids,
        )

        click.echo(f"\nQuestion: {question}")
        click.echo(f"\nAnswer: {response.answer}")
        click.echo(f"\nConfidence: {response.confidence:.2%}")

        if response.sources:
            click.echo(f"\nSources: {len(response.sources)} relevant triples")

    asyncio.run(run_query())


# =========================================================================
# Graph Commands
# =========================================================================


@cli.command()
def stats() -> None:
    """Show knowledge graph statistics."""
    from kggen_cuad.services.graph_service import get_graph_service

    async def get_stats():
        service = get_graph_service()
        stats = await service.get_statistics()

        click.echo("\n=== Knowledge Graph Statistics ===\n")
        click.echo(f"Total Entities: {stats.total_entities}")
        click.echo(f"Total Triples: {stats.total_triples}")
        click.echo(f"Contracts Processed: {stats.contracts_processed}")

        if stats.entities_by_type:
            click.echo("\nEntities by Type:")
            for entity_type, count in stats.entities_by_type.items():
                click.echo(f"  {entity_type}: {count}")

        if stats.triples_by_predicate:
            click.echo("\nTriples by Predicate:")
            for predicate, count in stats.triples_by_predicate.items():
                click.echo(f"  {predicate}: {count}")

    asyncio.run(get_stats())


@cli.command()
@click.argument("entity_id", type=str)
def entity(entity_id: str) -> None:
    """Show entity details."""
    from kggen_cuad.services.graph_service import get_graph_service

    async def get_entity():
        service = get_graph_service()
        entity = await service.get_entity(entity_id)

        if not entity:
            click.echo(f"Entity not found: {entity_id}", err=True)
            return

        click.echo(f"\n=== Entity: {entity.name} ===\n")
        click.echo(f"ID: {entity.id}")
        click.echo(f"Type: {entity.entity_type.value}")
        click.echo(f"Normalized Name: {entity.normalized_name}")

        if entity.properties:
            click.echo("\nProperties:")
            for key, value in entity.properties.items():
                click.echo(f"  {key}: {value}")

        if entity.aliases:
            click.echo(f"\nAliases: {', '.join(entity.aliases)}")

    asyncio.run(get_entity())


@cli.command()
@click.argument("search_query", type=str)
@click.option("--type", "entity_type", help="Filter by entity type")
@click.option("--limit", default=10, help="Maximum results")
def search(search_query: str, entity_type: Optional[str], limit: int) -> None:
    """Search for entities."""
    from kggen_cuad.services.search_service import get_search_service

    async def run_search():
        service = get_search_service()

        entity_types = [entity_type] if entity_type else None

        results = await service.search_entities(
            query=search_query,
            entity_types=entity_types,
            limit=limit,
        )

        click.echo(f"\n=== Search Results for '{search_query}' ===\n")

        if not results:
            click.echo("No results found.")
            return

        for entity, score in results:
            click.echo(f"  [{score:.3f}] {entity.name} ({entity.entity_type.value})")

    asyncio.run(run_search())


# =========================================================================
# Database Commands
# =========================================================================


@cli.command()
@click.confirmation_option(prompt="Are you sure you want to clear all data?")
def clear_db() -> None:
    """Clear all data from the knowledge graph."""
    from kggen_cuad.services.graph_service import get_graph_service

    async def clear():
        service = get_graph_service()
        await service.clear_all()
        click.echo("Knowledge graph cleared.")

    asyncio.run(clear())


@cli.command()
def health() -> None:
    """Check service health."""
    from kggen_cuad.services.llm_service import get_llm_service
    from kggen_cuad.storage.redis_cache import get_redis_cache

    click.echo("\n=== Service Health Check ===\n")

    # LLM
    llm = get_llm_service()
    llm_status = llm.health_check()
    click.echo(f"LLM Services:")
    for provider, status in llm_status.items():
        status_str = "✓" if status else "✗"
        click.echo(f"  {provider}: {status_str}")

    # Redis
    redis = get_redis_cache()
    redis_status = redis.health_check()
    status_str = "✓" if redis_status else "✗"
    click.echo(f"\nRedis: {status_str}")

    # Settings
    settings = get_settings()
    click.echo(f"\nEnvironment: {settings.environment}")
    click.echo(f"Debug: {settings.debug}")


# =========================================================================
# Config Commands
# =========================================================================


@cli.command()
def config() -> None:
    """Show current configuration."""
    settings = get_settings()

    click.echo("\n=== KGGEN-CUAD Configuration ===\n")
    click.echo(f"Environment: {settings.environment}")
    click.echo(f"Debug: {settings.debug}")
    click.echo(f"\nPrimary LLM: {settings.primary_llm_provider} ({settings.primary_llm_model})")
    click.echo(f"Fallback LLM: {settings.fallback_llm_provider} ({settings.fallback_llm_model})")
    click.echo(f"\nEmbedding Model: {settings.embedding_model}")
    click.echo(f"Embedding Dimension: {settings.embedding_dimension}")
    click.echo(f"\nEntity Types: {len(settings.entity_types)}")
    click.echo(f"Predicate Types: {len(settings.predicate_types)}")


@cli.command()
def init_db() -> None:
    """Initialize database schema."""
    click.echo("Initializing database schema...")
    click.echo("Run: docker-compose up -d")
    click.echo("Then: psql -h localhost -U kggen -d kggen < docker/postgres/init.sql")


def main() -> None:
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
