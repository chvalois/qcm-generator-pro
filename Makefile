# QCM Generator Pro - Development Makefile
# Based on specifications from CLAUDE.md

.PHONY: help install install-dev run run-ui test test-unit test-cov lint format db-init db-reset docker-build docker-run setup-models clean

# Default target
help:
	@echo "QCM Generator Pro - Available Commands"
	@echo "======================================"
	@echo ""
	@echo "Development Commands:"
	@echo "  install-dev     Install dependencies + pre-commit hooks"
	@echo "  run             Start FastAPI server"
	@echo "  run-ui          Start Gradio interface"
	@echo "  setup-models    Download local LLM models"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test            Run all tests"
	@echo "  test-unit       Run unit tests only"
	@echo "  test-cov        Run tests with coverage report"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint            Run linting (ruff + mypy)"
	@echo "  format          Format code (black + ruff)"
	@echo ""
	@echo "Database Commands:"
	@echo "  db-init         Initialize database"
	@echo "  db-reset        Reset database"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build    Build Docker container"
	@echo "  docker-run      Run containerized app"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean           Clean temporary files"
	@echo "  check           Run all quality checks"

# ============================================================================
# Installation & Setup
# ============================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install
	@echo "✅ Development environment installed successfully!"

setup-models:
	@echo "🔄 Setting up local LLM models..."
	python scripts/setup_local_models.py
	@echo "✅ Local models setup complete!"

# ============================================================================
# Development Server Commands
# ============================================================================

run:
	@echo "🚀 Starting FastAPI server..."
	uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

run-ui:
	@echo "🚀 Starting Gradio interface..."
	python -m src.ui.gradio_app

# ============================================================================
# Testing Commands
# ============================================================================

test:
	@echo "🧪 Running all tests..."
	pytest

test-unit:
	@echo "🧪 Running unit tests only..."
	pytest tests/unit/

test-integration:
	@echo "🧪 Running integration tests..."
	pytest tests/integration/

test-cov:
	@echo "🧪 Running tests with coverage report..."
	pytest --cov=src --cov-report=html --cov-report=term-missing
	@echo "📊 Coverage report generated in htmlcov/"

test-fast:
	@echo "🧪 Running fast tests (excluding slow markers)..."
	pytest -m "not slow"

# ============================================================================
# Code Quality Commands
# ============================================================================

lint:
	@echo "🔍 Running linting checks..."
	ruff check src tests
	mypy src

format:
	@echo "✨ Formatting code..."
	black src tests
	ruff check --fix src tests
	@echo "✅ Code formatted successfully!"

check: format lint test-unit
	@echo "✅ All quality checks passed!"

# ============================================================================
# Database Commands
# ============================================================================

db-init:
	@echo "🗄️  Initializing database..."
	python -c "from src.models.database import init_db; init_db()"
	@echo "✅ Database initialized!"

db-reset:
	@echo "🗄️  Resetting database..."
	rm -f data/database/*.db data/database/*.sqlite*
	$(MAKE) db-init
	@echo "✅ Database reset complete!"

db-migrate:
	@echo "🗄️  Running database migrations..."
	python scripts/migrate_db.py
	@echo "✅ Database migrated!"

# ============================================================================
# Docker Commands
# ============================================================================

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -f docker/Dockerfile -t qcm-generator-pro .
	@echo "✅ Docker image built successfully!"

docker-run:
	@echo "🐳 Running containerized application..."
	docker-compose -f docker/docker-compose.yml up --build

docker-stop:
	@echo "🐳 Stopping Docker containers..."
	docker-compose -f docker/docker-compose.yml down

# ============================================================================
# Data Management Commands
# ============================================================================

setup-dirs:
	@echo "📁 Creating project directories..."
	mkdir -p data/pdfs data/vectorstore data/database data/exports data/cache
	mkdir -p logs models
	touch data/pdfs/.gitkeep data/vectorstore/.gitkeep data/database/.gitkeep
	touch data/exports/.gitkeep data/cache/.gitkeep
	@echo "✅ Project directories created!"

clean-data:
	@echo "🧹 Cleaning data directories..."
	rm -rf data/pdfs/* data/vectorstore/* data/database/* data/exports/* data/cache/*
	@echo "✅ Data directories cleaned!"

# ============================================================================
# Development Utilities
# ============================================================================

clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov/ .mypy_cache .ruff_cache
	rm -rf build/ dist/ *.egg-info/
	rm -rf tmp/ temp/ logs/*.log
	@echo "✅ Cleanup complete!"

requirements:
	@echo "📦 Updating requirements files..."
	pip freeze > requirements-freeze.txt
	@echo "✅ Requirements updated!"

# ============================================================================
# Documentation Commands
# ============================================================================

docs-build:
	@echo "📖 Building documentation..."
	mkdocs build
	@echo "✅ Documentation built in site/"

docs-serve:
	@echo "📖 Serving documentation..."
	mkdocs serve

# ============================================================================
# Monitoring & Performance
# ============================================================================

profile:
	@echo "📊 Starting performance profiling..."
	python -m cProfile -o profile_results.prof -m src.api.main
	@echo "✅ Profiling complete! Results in profile_results.prof"

benchmark:
	@echo "⚡ Running benchmarks..."
	python scripts/benchmark.py
	@echo "✅ Benchmark complete!"

# ============================================================================
# CI/CD Helpers
# ============================================================================

ci-install:
	pip install -e ".[dev]"

ci-test:
	pytest --cov=src --cov-report=xml --junit-xml=test-results.xml

ci-lint:
	ruff check src tests --output-format=github
	mypy src --junit-xml=mypy-results.xml

# ============================================================================
# Development Workflow Shortcuts
# ============================================================================

dev-setup: install-dev setup-dirs db-init
	@echo "🎉 Development environment fully set up!"
	@echo "💡 Next steps:"
	@echo "   1. Copy .env.example to .env and configure"
	@echo "   2. Run 'make setup-models' to download LLM models"
	@echo "   3. Run 'make run' to start the development server"

quick-check: format lint test-fast
	@echo "⚡ Quick development check complete!"

full-check: format lint test
	@echo "✅ Full development check complete!"

# ============================================================================
# Debugging Commands
# ============================================================================

debug:
	@echo "🐛 Starting debug mode..."
	python -m debugpy --listen 5678 --wait-for-client -m src.api.main

debug-ui:
	@echo "🐛 Starting UI debug mode..."
	python -m debugpy --listen 5679 --wait-for-client -m src.ui.gradio_app

# ============================================================================
# Model Management
# ============================================================================

download-spacy:
	@echo "📥 Downloading spaCy models..."
	python -m spacy download fr_core_news_md
	python -m spacy download en_core_web_md
	@echo "✅ spaCy models downloaded!"

update-models:
	@echo "🔄 Updating all models..."
	$(MAKE) download-spacy
	$(MAKE) setup-models
	@echo "✅ All models updated!"

# ============================================================================
# Security Commands
# ============================================================================

security-check:
	@echo "🔒 Running security checks..."
	safety check
	bandit -r src/
	@echo "✅ Security check complete!"