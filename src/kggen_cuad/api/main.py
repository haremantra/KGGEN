"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from kggen_cuad.config import get_settings

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("application_starting")

    settings = get_settings()
    logger.info(
        "configuration_loaded",
        environment=settings.environment,
        debug=settings.debug,
    )

    yield

    # Shutdown
    logger.info("application_shutting_down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="KGGEN-CUAD API",
        description="Knowledge Graph Generator for CUAD Legal Contracts",
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.debug else None,
            },
        )

    # Include routers
    from kggen_cuad.api.routes import contracts, edits, graph, query, pipeline

    app.include_router(contracts.router, prefix="/api/v1/contracts", tags=["contracts"])
    app.include_router(edits.router, prefix="/api/v1/edits", tags=["edits"])
    app.include_router(graph.router, prefix="/api/v1/graph", tags=["graph"])
    app.include_router(query.router, prefix="/api/v1/query", tags=["query"])
    app.include_router(pipeline.router, prefix="/api/v1/pipeline", tags=["pipeline"])

    # Health check
    @app.get("/health")
    async def health_check() -> dict:
        """Health check endpoint."""
        from kggen_cuad.services.llm_service import get_llm_service
        from kggen_cuad.storage.redis_cache import get_redis_cache

        llm_status = get_llm_service().health_check()
        redis_status = get_redis_cache().health_check()

        return {
            "status": "healthy",
            "services": {
                "llm": llm_status,
                "redis": redis_status,
            },
        }

    # Root endpoint
    @app.get("/")
    async def root() -> dict:
        """Root endpoint."""
        return {
            "name": "KGGEN-CUAD API",
            "version": "0.1.0",
            "docs": "/docs",
        }

    return app


# Create default app instance
app = create_app()
