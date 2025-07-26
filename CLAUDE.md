# QCM Generator Pro - Local Edition
## Project Context for Claude Code

This document provides comprehensive context for developing the QCM Generator Pro application - a local multilingual QCM (Multiple Choice Questions) generation system from PDF documents.

---

To Do Next

- Améliorer RAG avec screening de la hiérarchie des fichiers uploadés : détection des titres (vérifier que chaque chunk se voit associé des titres logiques et exhaustifs)
=> si titre H1, chaque chunk se voit affecter ce titre H1 jusqu'au prochain titre H1, etc. 
- Ajouter des exemples de questions attendues
- Faire fonctionner avec Docker
- Faire fonctionner en local sur RTX 4090 avec Ollama
- Améliorer tests
- Refactor avec Clean Archi + SOLID
- Modifier le front avec du React ald Streamlit

---

## 🎯 Project Overview

### Description
Application locale de génération automatique de QCM multilingues à partir de documents PDF, avec support de modèles LLM locaux (RTX 4090) et APIs cloud.

### Key Features
- ✅ Multilingual QCM generation (FR/EN + extensible)
- ✅ Progressive validation workflow (1 → 5 → all questions)
- ✅ LLM-based automatic theme extraction from PDFs
- ✅ Local LLM support (RTX 4090 optimized) + Cloud APIs
- ✅ Direct CSV export for Udemy + JSON format
- ✅ Complete web UI with Streamlit interface
- ✅ Docker deployment (GPU/CPU) with Ollama integration
- ✅ RAG-based intelligent question generation with ChromaDB
- ✅ Multi-provider LLM support (OpenAI, Anthropic, Ollama)
- ✅ Real-time system monitoring and health checks

### Technology Stack
```yaml
Backend:
  - Python 3.11+
  - FastAPI + Pydantic v2
  - Langchain 0.1.0+
  - ChromaDB (vectorstore local)
  - SQLite + SQLAlchemy (metadata)
  - Ollama (local LLM serving)
  
Frontend:
  - Streamlit 4.0+ (complete web interface)
  
LLM Integration:
  - OpenAI API (GPT-3.5/4)
  - Anthropic API (Claude)
  - Ollama (Mistral, Llama3, Phi-3)
  
Deployment:
  - Docker + Docker Compose
  - Multi-service orchestration
  - GPU/CPU deployment options
  
Testing:
  - pytest + pytest-asyncio
  - coverage.py
  - pytest-mock
  
CI/CD:
  - GitHub Actions
  - pre-commit hooks
  - Black + Ruff (linting)
```

---

## 🏗️ Project Architecture

### Complete File Structure
```
qcm-generator/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Tests et linting
│       └── release.yml         # Build et release
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI app
│   │   ├── routes/
│   │   │   ├── documents.py   # Endpoints PDF
│   │   │   ├── generation.py  # Endpoints génération
│   │   │   └── export.py      # Endpoints export
│   │   └── dependencies.py    # Dépendances FastAPI
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration Pydantic
│   │   ├── constants.py       # Constantes
│   │   └── exceptions.py      # Exceptions custom
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py        # SQLAlchemy models
│   │   ├── schemas.py         # Pydantic schemas
│   │   └── enums.py           # Enums (types questions, etc.)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pdf_processor.py   # Traitement PDF + thèmes
│   │   ├── theme_extractor.py # Extraction thématique
│   │   ├── rag_engine.py      # ChromaDB + retrieval
│   │   ├── llm_manager.py     # Gestion multi-LLM
│   │   ├── qcm_generator.py   # Génération questions
│   │   ├── validator.py       # Validation questions
│   │   └── exporter.py        # Export CSV Udemy
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── templates.py       # Templates multilingues
│   │   └── languages/
│   │       ├── fr.py          # Prompts français
│   │       ├── en.py          # Prompts anglais
│   │       └── base.py        # Template de base
│   └── ui/
│       ├── __init__.py
│       └── streamlit_app.py      # Interface Streamlit
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Fixtures pytest
│   ├── unit/
│   │   ├── test_pdf_processor.py
│   │   ├── test_theme_extractor.py
│   │   ├── test_rag_engine.py
│   │   ├── test_qcm_generator.py
│   │   └── test_validator.py
│   ├── integration/
│   │   ├── test_api_endpoints.py
│   │   ├── test_generation_flow.py
│   │   └── test_export_flow.py
│   └── fixtures/
│       ├── sample.pdf
│       └── test_data.json
├── data/
│   ├── pdfs/                  # PDFs uploadés
│   ├── vectorstore/           # ChromaDB
│   ├── database/              # SQLite
│   └── exports/               # CSV générés
├── models/                    # Modèles LLM locaux
├── scripts/
│   ├── start_app.py            # Multi-process startup
│   ├── docker_setup.py         # Docker initialization
│   ├── docker_start.py         # Container startup
│   ├── setup_local_models.py   # Model downloads
│   └── migrate_db.py          # Migrations DB
├── Dockerfile                  # Container image
├── docker-compose.yml          # GPU deployment
├── docker-compose.cpu.yml      # CPU deployment
├── .dockerignore              # Docker build context
├── .env.docker                # Docker environment
├── DOCKER.md                  # Deployment guide
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── CLAUDE.md                  # This file
└── Makefile
```

## 📋 Development Best Practices

### 1. Use Plan Mode (Shift + Tab)
Always start complex tasks in plan mode to outline approach before implementation.

### 2. CLAUDE.md Context
This file provides comprehensive project context - reference it for architecture decisions.

### 7. Use Subagents
For massive tasks, leverage subagents to handle specific components.

### 8. Edge Case Analysis
Always ask Claude to identify and handle edge cases in implementation.

### 10. MCP Context7
Use MCP Context7 for updated documentation and best practices.

### 3. Always clean up the mess
Inspect directory and clean what is not necessary but ask user always

### 4. Cut project into different phases
Each phase must enable the user to test properly what has been done through terminal or UI

### 5. Follow best practices rules
SRP / SOLID
KISS / YAGNI
Dependency Injection
Clean Architecture

### 6. Code must be done in English (docstrings, comments)



---

## 📊 Data Models

### SQLAlchemy Models (database.py)
```python
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    upload_date = Column(DateTime, nullable=False)
    total_pages = Column(Integer)
    language = Column(String(10), default="fr")
    processing_status = Column(String(50))
    metadata = Column(JSON)  # {title, author, creation_date, etc.}
    themes = relationship("DocumentTheme", back_populates="document")
    chunks = relationship("DocumentChunk", back_populates="document")

class DocumentTheme(Base):
    __tablename__ = "document_themes"
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    theme_name = Column(String(100), nullable=False)
    start_page = Column(Integer)
    end_page = Column(Integer)
    confidence_score = Column(Float)
    keywords = Column(JSON)  # Liste de mots-clés

class Question(Base):
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(50))
    question_text = Column(Text, nullable=False)
    question_type = Column(String(20))  # multiple-choice, multiple-selection
    language = Column(String(10))
    difficulty = Column(String(20))
    theme = Column(String(100))
    options = Column(JSON)  # Liste de 3-6 options
    correct_answers = Column(JSON)  # Liste d'indices (base 0)
    explanation = Column(Text)
    metadata = Column(JSON)
    validation_status = Column(String(20))  # pending, validated, rejected
    created_at = Column(DateTime)
```

### Pydantic Schemas (schemas.py)
```python
class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "multiple-choice"
    MULTIPLE_SELECTION = "multiple-selection"

class Language(str, Enum):
    FR = "fr"
    EN = "en"
    ES = "es"
    DE = "de"

class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    MIXED = "mixed"

class GenerationConfig(BaseModel):
    num_questions: int = Field(ge=1, le=250)
    question_types: Dict[QuestionType, float] = {
        QuestionType.MULTIPLE_CHOICE: 0.7,
        QuestionType.MULTIPLE_SELECTION: 0.3
    }
    difficulty_distribution: Dict[Difficulty, float] = {
        Difficulty.EASY: 0.3,
        Difficulty.MEDIUM: 0.5,
        Difficulty.HARD: 0.2
    }
    language: Language = Language.FR
    themes_filter: Optional[List[str]] = None
    model: str = "mistral-local"
    validation_mode: str = "progressive"  # progressive, direct
    batch_sizes: List[int] = [1, 5, -1]  # -1 pour "toutes"
```

---

## 🔧 Core Services

### Theme Extractor Service
- **Purpose**: Automatic theme extraction from PDF documents
- **Features**: Structural analysis + content clustering
- **Technologies**: spaCy, scikit-learn TF-IDF, KMeans clustering
- **Output**: ThemeDetection objects with confidence scores

### RAG Engine Service
- **Purpose**: Retrieval-Augmented Generation for question context
- **Vector Store**: ChromaDB for local embeddings
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Features**: Theme-filtered similarity search, chunk retrieval

### Progressive QCM Generator
- **Workflow**: 1 question → 5 questions → all remaining
- **Validation**: Automatic + manual approval points
- **Features**: Theme distribution, difficulty balancing, duplicate prevention

### Question Validator
- **Structure Validation**: Options count (3-6), correct answers format
- **Content Validation**: Question clarity, distractor quality
- **Linguistic Validation**: Language consistency, grammar check

### LLM Manager
- **Local Models**: Mistral-7B, Llama3-8B, Phi-3 (RTX 4090 optimized)
- **Cloud APIs**: OpenAI, Anthropic (optional)
- **Ollama Integration**: Local model serving
- **Features**: Model switching, performance monitoring

---

## 🌍 Multilingual Support

### Language Templates Structure
```python
# Base template for all languages
class LanguageTemplate(ABC):
    @abstractmethod
    def get_question_generation_prompt(self) -> str: pass
    
    @abstractmethod
    def get_distractor_generation_prompt(self) -> str: pass
    
    @abstractmethod
    def get_validation_prompt(self) -> str: pass
    
    @abstractmethod
    def get_theme_extraction_prompt(self) -> str: pass

# Language-specific implementations
class FrenchTemplate(LanguageTemplate): ...
class EnglishTemplate(LanguageTemplate): ...
```

### Supported Languages
- **Primary**: French (fr), English (en)
- **Extensible**: Spanish (es), German (de)
- **Configuration**: Language detection, prompt templates, validation rules

---

## 🧪 Testing Strategy

### Unit Tests Structure
```
tests/unit/
├── test_pdf_processor.py      # PDF parsing, metadata extraction
├── test_theme_extractor.py    # Theme detection algorithms
├── test_rag_engine.py         # Vector search, chunk retrieval
├── test_qcm_generator.py      # Question generation logic
├── test_validator.py          # Question validation rules
└── test_llm_manager.py        # Model management
```

### Integration Tests
```
tests/integration/
├── test_api_endpoints.py      # FastAPI routes testing
├── test_generation_flow.py    # End-to-end generation workflow
└── test_export_flow.py        # CSV export functionality
```

### Testing Requirements
- **Coverage Target**: >90% for core services
- **Fixtures**: Sample PDFs, mock LLM responses
- **Async Testing**: pytest-asyncio for async operations
- **Database**: In-memory SQLite for tests

---

## 🚀 Development Commands

### Makefile Commands
```bash
# Development
make install-dev      # Install dependencies + pre-commit
make run              # Start FastAPI server
make run-ui           # Start Streamlit interface
make setup-models     # Download local LLM models

# Testing
make test             # Run all tests
make test-unit        # Unit tests only
make test-cov         # Tests with coverage report

# Code Quality
make lint             # Run linting (ruff + mypy)
make format           # Format code (black + isort)

# Database
make db-init          # Initialize database
make db-reset         # Reset database

# Docker
make docker-build     # Build container
make docker-run       # Run containerized app (GPU)
make docker-run-cpu   # Run containerized app (CPU)
make docker-logs      # View container logs
make docker-shell     # Shell access to container
```

---

## ⚙️ Configuration

### Environment Variables (.env)
```bash
# Application
APP_NAME="QCM Generator"
DEBUG=false

# Database
DATABASE_URL="sqlite:///data/qcm_generator.db"

# LLM Configuration
DEFAULT_LLM="mistral-local"
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""

# Local Models
LOCAL_MODELS_DIR="./models"
OLLAMA_BASE_URL="http://localhost:11434"

# Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_PDF_SIZE_MB=50

# UI
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SHARE=false
```

### Local Model Configuration
```python
local_models_config = {
    "mistral-7b": {
        "path": "models/mistral-7b-instruct-v0.2.Q5_K_M.gguf",
        "context_length": 4096,
        "gpu_layers": -1  # Full GPU acceleration
    },
    "llama3-8b": {
        "path": "models/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf",
        "context_length": 8192,
        "gpu_layers": -1
    }
}
```

---


## 🔐 Security & Quality

### Pre-commit Hooks
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/psf/black
    hooks:
      - id: black
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-merge-conflict
```

### CI/CD Pipeline (GitHub Actions)
```yaml
# .github/workflows/ci.yml
- Python 3.10, 3.11, 3.12 matrix testing
- Linting: ruff check, black --check
- Type checking: mypy
- Testing: pytest with coverage
- Security: Trivy container scanning
- Docker: Multi-stage build optimization
```

---

## 🎯 Key Implementation Notes

### Progressive Generation Workflow
1. **Initial Test**: Generate 1 question for validation
2. **Small Batch**: Generate 5 questions, manual review
3. **Full Generation**: Generate remaining questions automatically
4. **Quality Control**: Continuous validation at each step

### Theme-Based Question Distribution
- Extract themes automatically from PDF structure
- Balance questions across identified themes
- Ensure comprehensive content coverage
- Support manual theme filtering

### Export Format (Udemy CSV)
```csv
question,answer_1,answer_2,answer_3,answer_4,correct_answer,explanation
"Qu'est-ce que Python?","Un serpent","Un langage","Un fruit","Un outil","2","Python est un langage de programmation..."
```

### Local LLM Optimization (RTX 4090)
- **Recommended**: Mistral-7B for speed/quality balance
- **High Quality**: Llama3-8B for complex questions
- **Ultra Fast**: Phi-3 for rapid iteration
- **GPU Memory**: Full 24GB VRAM utilization

---

## 🐛 Troubleshooting

### Common Issues
1. **Model Loading**: Ensure sufficient VRAM (24GB RTX 4090)
2. **PDF Processing**: Check file size limits and encoding
3. **Database Locks**: Use connection pooling for concurrent access
4. **Theme Extraction**: Verify spaCy model installation

### Debug Commands
```bash
# Verbose logging
export LOG_LEVEL=DEBUG

# Reset everything
make clean && make db-reset && make install-dev

# Test specific component
pytest tests/unit/test_theme_extractor.py -v -s
```

### Performance Monitoring
- Track generation time per question
- Monitor GPU memory usage
- Log validation failure rates
- Measure theme extraction accuracy

---

## 🚧 Implementation Status

### ✅ Phase 1: Project Foundation (COMPLETED)
- **Project Structure**: Complete directory tree with all required folders
- **Configuration**: pyproject.toml, .env.example, .gitignore, Makefile
- **Quality Tools**: Pre-commit hooks, CI/CD pipeline, linting/formatting
- **Database Models**: SQLAlchemy models (Document, Question, DocumentTheme, etc.)
- **Development Environment**: Ready for local development with make commands

### ✅ Phase 2: Core Models & Services (COMPLETED)
- **Data Models**: Complete SQLAlchemy models with relationships and constraints
- **Pydantic Schemas**: Comprehensive request/response schemas with validation (Pydantic v2 compatible)
- **Enums & Constants**: Type-safe enums and application constants
- **Configuration**: Pydantic Settings with environment variable support (using pydantic-settings)
- **Exception Handling**: Custom exception hierarchy with detailed error context
- **Database Management**: Connection pooling, session management, async support
- **Testing Infrastructure**: Comprehensive test suite with fixtures and utilities

### ✅ Phase 3: Core Services Implementation (COMPLETED)
**Implemented:**
- **PDF Processor**: Text extraction, metadata parsing, document chunking
- **Theme Extractor**: LLM-based intelligent theme detection with fallback
- **RAG Engine**: ChromaDB integration with semantic search and context retrieval
- **LLM Manager**: Multi-provider support (OpenAI, Anthropic, Ollama) with fallback mechanisms
- **QCM Generator**: Progressive workflow (1→5→all) with RAG context and validation
- **Question Validator**: Comprehensive quality validation system
- **Export Service**: Multi-format export (CSV, JSON) with Udemy-compatible formatting

### ✅ Phase 4: API & Export Layer (COMPLETED)
**Implemented:**
- **FastAPI Routes**: Complete REST API with document upload, generation, and export endpoints
- **API Dependencies**: Authentication, validation, and dependency injection
- **Progressive QCM Workflow**: API endpoints for 1→5→all generation with validation checkpoints
- **Export System**: CSV for Udemy, JSON export with metadata and download functionality
- **API Documentation**: Health checks, metrics, auto-generated OpenAPI docs with examples
- **Middleware & Security**: CORS, logging, error handling, and security headers

### ✅ Phase 5: UI & Advanced Features (COMPLETED)
**Implemented:**
- **Streamlit Interface**: Complete web UI with document upload, generation, and export functionality
- **System Monitoring**: Real-time metrics, health checks, and configuration interface
- **Multi-language Support**: French/English question generation with extensible templates
- **Question Editing**: Interactive validation and editing interface for generated questions
- **Export Interface**: Format selection (CSV/JSON) and download functionality
- **Responsive Design**: Mobile-friendly interface with progress indicators and status updates

### ✅ Phase 6: Docker Deployment (COMPLETED)
**Implemented:**
- **Multi-service Containerization**: Docker setup with Ollama LLM server, FastAPI backend, Streamlit UI, and Redis cache
- **GPU/CPU Flexibility**: Support for both GPU-accelerated (RTX 4090) and CPU-only deployments
- **Automated Setup**: Docker scripts for model downloads, database initialization, and health checks
- **Production Configuration**: Security best practices, volume management, environment configuration
- **Management Tools**: Comprehensive Docker commands via Makefile for build, run, monitor, and debug operations

**Current Status**: 
- ✅ All core phases completed (1-6)
- ✅ Full application stack operational and production-ready
- ✅ Docker deployment with GPU/CPU support
- ✅ Complete UI with all planned features implemented
- ✅ PDF processing with LLM-based theme extraction
- ✅ Multi-provider LLM integration with fallback mechanisms
- ✅ Progressive QCM generation workflow (1→5→all) fully functional
- ✅ Export system supporting multiple formats (CSV, JSON)

**All Services Completed:**
- `pdf_processor.py`: PDF text extraction, metadata parsing, and document chunking
- `theme_extractor.py`: LLM-based theme detection with intelligent fallback
- `rag_engine.py`: ChromaDB vector store with semantic similarity search
- `llm_manager.py`: Multi-provider LLM integration (OpenAI, Anthropic, Ollama)
- `qcm_generator.py`: Progressive question generation with RAG context
- `validator.py`: Comprehensive question quality and structure validation
- `exporter.py`: Multi-format export with Udemy CSV compatibility

**API Layer Completed:**
- `dependencies.py`: FastAPI dependency injection and authentication
- `routes/documents.py`: Document upload, processing, and management endpoints
- `routes/generation.py`: QCM generation with progressive workflow endpoints
- `routes/export.py`: Export functionality with format selection and download
- `routes/health.py`: Health checks, system metrics, and monitoring endpoints
- `main.py`: Complete FastAPI application with middleware and security

**UI & Deployment Completed:**
- `streamlit_app.py`: Complete Streamlit interface with all features
- `start_app.py`: Multi-process startup script for API and UI
- `docker_setup.py`: Automated Docker deployment setup and configuration
- `docker_start.py`: Container startup orchestration with health monitoring
- Docker compose files for GPU and CPU deployment scenarios

**Remaining Optional Tasks**: 
- ⏳ Enhanced multilingual prompt templates (currently basic FR/EN support)
- ⏳ Advanced performance optimization for large document processing
- ⏳ Extended testing coverage for Docker deployment scenarios

---

## 🔄 Next Steps - UI Migration

### Phase 7: Streamlit to Streamlit Migration (PRIORITY)

**Issue**: Streamlit has persistent compatibility issues in Docker environments due to JSON schema bugs that affect complex interfaces.

**Solution**: Complete migration from Streamlit to Streamlit for a more stable and Docker-compatible UI framework.

#### Migration Tasks:

1. **Remove Streamlit Dependencies**
   - Remove all Streamlit imports and references from codebase
   - Update requirements.txt to remove Streamlit and add Streamlit
   - Clean up Streamlit-specific configurations

2. **Create Streamlit Interface**
   - Design new Streamlit UI structure matching current functionality
   - Implement document upload interface
   - Create QCM generation workflow (1→5→all progression)
   - Build export functionality interface
   - Add system monitoring dashboard

3. **Update Docker Configuration**
   - Modify docker_start.py to launch Streamlit instead of Streamlit
   - Update port configurations (Streamlit default: 8501)
   - Test Docker compatibility

4. **Code Migration Strategy**
   - Replace `src/ui/streamlit_app.py` with `src/ui/streamlit_app.py`
   - Convert Streamlit components to Streamlit equivalents:
     - `gr.File` → `st.file_uploader`
     - `gr.Button` → `st.button`
     - `gr.Textbox` → `st.text_input`/`st.text_area`
     - `gr.Slider` → `st.slider`
     - `gr.Dropdown` → `st.selectbox`
     - `gr.CheckboxGroup` → `st.multiselect`
     - `gr.Progress` → `st.progress`
     - `gr.Tabs` → `st.tabs`

5. **Benefits of Streamlit Migration**
   - Better Docker compatibility
   - More intuitive state management
   - Easier component handling
   - Better error handling
   - Cleaner code structure
   - More robust in production environments

#### Implementation Priority:
- **High Priority**: This migration resolves the Docker UI accessibility issue
- **Timeline**: Should be completed before any major deployments
- **Impact**: Improves user experience and deployment reliability

---

## 📚 Additional Resources

### Documentation
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Langchain Docs](https://docs.langchain.com/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Streamlit Docs](https://docs.streamlit.io/)

### Model Resources
- [Hugging Face Models](https://huggingface.co/models)
- [Ollama Models](https://ollama.ai/library)
- [GGUF Quantized Models](https://huggingface.co/models?library=gguf)

This document serves as the comprehensive context for developing the QCM Generator Pro application. Reference it throughout development to maintain consistency with the project architecture and requirements.