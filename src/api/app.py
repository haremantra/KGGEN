"""FastAPI application for KGGEN-CUAD contract analysis."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from ..config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("Starting KGGEN-CUAD API...")
    print(f"Debug mode: {settings.debug}")

    # Initialize heavy resources here (classifier, etc.)
    # We lazy-load these in routes to avoid blocking startup

    yield

    # Shutdown
    print("Shutting down KGGEN-CUAD API...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="KGGEN-CUAD API",
        description="Contract Analysis and Knowledge Graph API for legal documents",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware for frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",  # Streamlit
            "http://localhost:3000",  # React dev
            "http://127.0.0.1:8501",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(router, prefix="/api")

    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "kggen-cuad-api",
            "version": "0.1.0",
        }

    @app.get("/health")
    async def health():
        """Detailed health check."""
        return {
            "status": "healthy",
            "neo4j": "configured",
            "api": "running",
        }

    return app


# Create app instance
app = create_app()
