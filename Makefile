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
	@echo "  run-ui          Start Streamlit interface (main entry point)"
	@echo "  run-ui-clean    Start Streamlit interface (clean mode, no page navigation)"
	@echo "  run-app         Start complete app (API + UI)"
	@echo "  run-app-debug   Start complete app in debug mode"
	@echo "  run-api-only    Start FastAPI backend only"
	@echo "  run-ui-only     Start Streamlit frontend only"
	@echo "  setup-models    Download local LLM models"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test            Run all tests (26 tests, all passing)"
	@echo "  test-basic      Run basic functionality tests only (6 tests)"
	@echo "  test-working    Run core tests (basic + models) (21 tests)"
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
	uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8001

run-ui:
	@echo "🚀 Starting Streamlit interface (main entry point)..."
	python -m streamlit run main_app.py

run-ui-clean:
	@echo "🚀 Starting Streamlit interface (clean mode)..."
	python -m streamlit run main_app.py --server.headless=true --browser.gatherUsageStats=false

run-app:
	@echo "🚀 Starting complete QCM Generator Pro (API + UI)..."
	python3 scripts/start_app.py

run-app-debug:
	@echo "🚀 Starting QCM Generator Pro in debug mode..."
	python3 scripts/start_app.py --debug

run-api-only:
	@echo "🚀 Starting FastAPI backend only..."
	python3 scripts/start_app.py --api-only

run-ui-only:
	@echo "🚀 Starting Streamlit frontend only..."
	python3 scripts/start_app.py --ui-only

# ============================================================================
# Testing Commands
# ============================================================================

test:
	@echo "🧪 Running all tests..."
	@if command -v pytest-cov > /dev/null 2>&1; then \
		pytest --cov=src --cov-report=term-missing; \
	else \
		pytest; \
	fi

test-simple:
	@echo "🧪 Running tests (no coverage)..."
	pytest

test-basic:
	@echo "🧪 Running basic functionality tests..."
	pytest tests/unit/test_basic.py -v

test-working:
	@echo "🧪 Running working tests only..."
	pytest tests/unit/test_basic.py tests/unit/test_models.py -v

test-unit:
	@echo "🧪 Running unit tests only..."
	@if command -v pytest-cov > /dev/null 2>&1; then \
		pytest tests/unit/ --cov=src --cov-report=term-missing; \
	else \
		pytest tests/unit/; \
	fi

test-integration:
	@echo "🧪 Running integration tests..."
	@if command -v pytest-cov > /dev/null 2>&1; then \
		pytest tests/integration/ --cov=src --cov-report=term-missing; \
	else \
		pytest tests/integration/; \
	fi

test-cov:
	@echo "🧪 Running tests with coverage report..."
	pytest --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml
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

check-setup:
	@echo "🔍 Checking development setup..."
	python3 scripts/check_setup.py

integration-test:
	@echo "🧪 Running integration test..."
	python3 scripts/integration_test.py

demo:
	@echo "🎯 Running system demo..."
	python3 scripts/demo.py

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
	docker build -t qcm-generator-pro .
	@echo "✅ Docker image built successfully!"

docker-build-no-cache:
	@echo "🐳 Building Docker image (no cache)..."
	docker build --no-cache -t qcm-generator-pro .
	@echo "✅ Docker image built successfully!"

docker-run:
	@echo "🐳 Running containerized application (GPU)..."
	docker compose up --build -d
	@echo "🚀 Application starting... Check logs with 'make docker-logs'"

docker-run-cpu:
	@echo "🐳 Running containerized application (CPU only)..."
	docker compose -f docker-compose.cpu.yml up --build -d
	@echo "🚀 Application starting... Check logs with 'make docker-logs'"

docker-run-detached:
	@echo "🐳 Running containerized application in background..."
	docker compose up --build -d

docker-stop:
	@echo "🐳 Stopping Docker containers..."
	docker compose down

docker-restart:
	@echo "🐳 Restarting Docker containers..."
	docker compose restart

docker-logs:
	@echo "📋 Showing container logs..."
	docker compose logs -f

docker-logs-api:
	@echo "📋 Showing API container logs..."
	docker compose logs -f qcm_app

docker-logs-ollama:
	@echo "📋 Showing Ollama container logs..."
	docker compose logs -f ollama

docker-shell:
	@echo "🐚 Opening shell in main container..."
	docker compose exec qcm_app /bin/bash

docker-shell-ollama:
	@echo "🐚 Opening shell in Ollama container..."
	docker compose exec ollama /bin/bash

docker-setup:
	@echo "⚙️ Running Docker setup tasks..."
	docker compose exec qcm_app python3 scripts/docker_setup.py

docker-health:
	@echo "🏥 Checking container health..."
	docker compose exec qcm_app python3 scripts/docker_setup.py --health-check

docker-clean:
	@echo "🧹 Cleaning Docker resources..."
	docker compose down -v
	docker system prune -f
	@echo "✅ Docker cleanup complete!"

docker-reset: docker-clean docker-build docker-run
	@echo "🔄 Docker environment reset complete!"

docker-dev:
	@echo "🐳 Starting Docker development environment..."
	cp .env .env
	docker compose up --build

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
	ruff check src tests

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
	python -m debugpy --listen 5679 --wait-for-client -m streamlit run main_app.py

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

# ============================================================================
# Playwright Testing Commands
# ============================================================================

test-playwright-install:	## Install Playwright dependencies
	@echo "🎭 Installing Playwright dependencies..."
	uv pip install -r tests/playwright/requirements.txt
	python -m playwright install chromium
	@echo "✅ Playwright installed successfully"

test-playwright-baseline:	## Run Streamlit baseline tests
	@echo "🎭 Running Streamlit baseline tests..."
	python tests/playwright/run_tests.py baseline --headed
	@echo "✅ Baseline tests completed"

test-playwright-comparison:	## Run interface comparison tests
	@echo "🎭 Running interface comparison tests..."
	python tests/playwright/run_tests.py comparison --headed
	@echo "✅ Comparison tests completed"

test-playwright-all:	## Run all Playwright tests
	@echo "🎭 Running all Playwright tests..."
	python tests/playwright/run_tests.py all
	@echo "✅ All Playwright tests completed"

test-playwright-report:	## Generate comparison report
	@echo "🎭 Generating comparison report..."
	python tests/playwright/run_tests.py report
	@echo "📊 Comparison report generated"

test-playwright-clean:	## Clean Playwright artifacts
	@echo "🎭 Cleaning Playwright artifacts..."
	python tests/playwright/run_tests.py clean
	@echo "🧹 Playwright artifacts cleaned"

test-playwright-services:	## Check if services are running
	@echo "🎭 Checking services status..."
	python tests/playwright/run_tests.py check-services