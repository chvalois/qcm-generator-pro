# QCM Generator Pro - Docker Image  
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Create non-root user
RUN groupadd -r qcmuser && useradd -r -g qcmuser qcmuser

# Copy requirements first for better layer caching
COPY requirements*.txt ./
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit langchain openai chromadb python-multipart

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY main_app.py ./
COPY Makefile ./
COPY .streamlit/ ./.streamlit/

# Create data directories with proper permissions
RUN mkdir -p data/pdfs data/vectorstore data/database data/exports data/cache logs uploads && \
    chown -R qcmuser:qcmuser /app

# Switch to non-root user
USER qcmuser

# Expose ports
EXPOSE 8001 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/api/health || exit 1

# Default command
CMD ["python3", "scripts/docker_start.py"]