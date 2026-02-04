# KGGEN-CUAD Application Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir pip --upgrade && \
    pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir \
    anthropic>=0.18.0 \
    openai>=1.12.0 \
    sentence-transformers>=2.2.0 \
    transformers>=4.36.0 \
    neo4j>=5.15.0 \
    networkx>=3.2.0 \
    rank-bm25>=0.2.2 \
    qdrant-client>=1.7.0 \
    pdfplumber>=0.10.0 \
    PyPDF2>=3.0.0 \
    fastapi>=0.109.0 \
    uvicorn>=0.27.0 \
    python-multipart>=0.0.6 \
    pandas>=2.1.0 \
    numpy>=1.26.0 \
    pydantic>=2.5.0 \
    pydantic-settings>=2.1.0 \
    asyncpg>=0.29.0 \
    sqlalchemy>=2.0.0 \
    python-dotenv>=1.0.0 \
    httpx>=0.26.0 \
    tenacity>=8.2.0 \
    structlog>=24.1.0 \
    streamlit>=1.30.0 \
    plotly>=5.18.0 \
    requests>=2.31.0

# Copy application code
COPY src/ ./src/
COPY streamlit_app.py .

# Set Python path
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
