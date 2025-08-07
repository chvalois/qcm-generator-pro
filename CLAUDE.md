# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# QCM Generator Pro - Local Edition
## Project Context for Claude Code

This document provides comprehensive context for developing the QCM Generator Pro application - a local multilingual QCM (Multiple Choice Questions) generation system from PDF documents.

---

To Do Next

✅ **COMPLETED** - Services Architecture Reorganization (January 2025)
- Réorganisé 21 services en 5 domaines métier
- Mis à jour 50+ imports dans toute la codebase
- Structure claire : document/, generation/, quality/, llm/, infrastructure/

✅ **COMPLETED** - Component-Based UI Architecture (January 2025)
- Interface Streamlit réorganisée en composants réutilisables (95% réduction de taille : 2992 → 146 lignes)
- Architecture prête pour migration React avec séparation claire des responsabilités
- Composants modulaires : pages, common, core avec InterfaceManager central
- Gestion manuelle des modèles Ollama (désactivation téléchargement automatique)

**CURRENT FOCUS: React Migration (January 2025)**
- 🚀 **React Migration en cours** : Transition de Streamlit vers React/TypeScript avec Shadcn/ui
- 🎭 **Tests Playwright configurés** : Comparaison automatisée des interfaces Streamlit vs React
- 🔄 **Migration progressive** : Maintien des deux interfaces en parallèle pendant la transition
- 📱 **UX moderne** : Interface responsive avec composants Shadcn/ui et TanStack Query

**NEXT PRIORITIES:**
- Finaliser la migration React complète (12 semaines planifiées)
- Améliorer tests avec la nouvelle architecture de composants
- Améliorer la détection automatique de titres et le découpage en chunks intelligents (par ex, dans le cas de slides, le titre est en haut, et le chunk contient l'ensemble de la slide)
- Implémenter les fonctionnalités réelles de téléchargement des modèles Ollama via l'interface

---

## 🦙 Gestion des Modèles Ollama

### Téléchargement Manuel (Nouvelle Fonctionnalité)

**Changement Important :** Le téléchargement automatique des modèles Ollama au démarrage a été **désactivé par défaut** pour éviter les téléchargements non souhaités.

#### **Configuration :**
```bash
# Variable d'environnement pour contrôler le téléchargement automatique
OLLAMA_AUTO_DOWNLOAD_MODELS=false  # Désactivé par défaut
```

#### **Interface de Téléchargement Manuel :**
Accessible via **Système → Gestion des modèles Ollama** dans l'interface Streamlit :

- ✅ **Modèles Recommandés** : Boutons de téléchargement pour `mistral:7b-instruct`, `llama3:8b-instruct`, `phi3:mini`
- ✅ **Téléchargement Personnalisé** : Champ de saisie pour télécharger n'importe quel modèle Ollama
- ✅ **Statut des Modèles** : Indication visuelle des modèles installés/non installés
- ✅ **Gestion d'Erreurs** : Messages d'erreur clairs en cas d'échec

#### **Réactivation du Téléchargement Automatique :**
Pour réactiver le téléchargement automatique, modifiez :
```bash
# Dans docker-compose.yml ou .env.docker
OLLAMA_AUTO_DOWNLOAD_MODELS=true
```

#### **Avantages :**
- 🚀 **Démarrage Plus Rapide** : L'app démarre immédiatement sans attendre les téléchargements
- 💾 **Contrôle de l'Espace Disque** : Téléchargez uniquement les modèles nécessaires
- 🎯 **Expérience Utilisateur** : Choisissez quels modèles installer selon vos besoins

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
│   ├── services/              # 🎯 NOUVELLE ARCHITECTURE (Jan 2025)
│   │   ├── __init__.py
│   │   ├── document/          # Documents & PDF processing
│   │   │   ├── __init__.py
│   │   │   ├── pdf_processor.py     # Extraction PDF + métadonnées
│   │   │   ├── theme_extractor.py   # Extraction thématique LLM
│   │   │   ├── title_detector.py    # Détection titres documents
│   │   │   └── document_manager.py  # Gestion cycle de vie docs
│   │   ├── generation/        # Génération QCM & workflows
│   │   │   ├── __init__.py
│   │   │   ├── qcm_generator.py           # Générateur principal
│   │   │   ├── chunk_based_generator.py   # Génération par chunks
│   │   │   ├── title_based_generator.py   # Génération par titres
│   │   │   ├── enhanced_qcm_generator.py  # Génération avancée
│   │   │   ├── progressive_workflow.py    # Workflow 1→5→all
│   │   │   ├── question_prompt_builder.py # Construction prompts
│   │   │   ├── question_parser.py         # Parse JSON questions
│   │   │   └── question_selection.py      # Sélection & filtrage
│   │   ├── quality/           # Assurance qualité
│   │   │   ├── __init__.py
│   │   │   ├── validator.py                    # Validation questions
│   │   │   ├── question_deduplicator.py        # Déduplication
│   │   │   ├── question_diversity_enhancer.py  # Amélioration diversité
│   │   │   └── chunk_variety_validator.py      # Validation variété chunks
│   │   ├── llm/              # Intégration LLM
│   │   │   ├── __init__.py
│   │   │   ├── llm_manager.py           # Gestion multi-providers
│   │   │   ├── langsmith_tracker.py     # Tracking LangSmith
│   │   │   └── simple_examples_loader.py # Gestion exemples few-shot
│   │   └── infrastructure/    # Services infrastructure
│   │       ├── __init__.py
│   │       ├── rag_engine.py      # ChromaDB + recherche vectorielle
│   │       └── progress_tracker.py # Suivi progrès temps réel
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

## 🎯 **NOUVELLE ARCHITECTURE SERVICES** *(Janvier 2025)*

### Architecture Réorganisée par Domaines Métier

L'architecture des services a été **complètement réorganisée** pour une meilleure maintenabilité et séparation des responsabilités :

#### **🏗️ Structure Actuelle (21 services → 5 domaines)**
```
src/services/
├── document/          # 📄 Gestion documents (4 services)
├── generation/        # ⚡ Génération QCM (8 services)  
├── quality/          # ✅ Assurance qualité (4 services)
├── llm/             # 🤖 Intégration LLM (3 services)
└── infrastructure/   # 🔧 Services infrastructure (2 services)
```

#### **🔄 Imports Mis à Jour**
- **Avant** : `from src.services.llm_manager import ...`
- **Après** : `from src.services.llm.llm_manager import ...`

#### **📦 Exports Publics**
Chaque domaine expose une API publique via `__init__.py` :
```python
# Exemple : src.services.generation
from src.services.generation import get_qcm_generator, generate_progressive_qcm
```

#### **✅ Validation**
- **21/21 tests** passent toujours ✅
- **50+ imports** mis à jour dans toute la codebase
- **Scripts** `/scripts/` corrigés pour les nouveaux chemins
- **Interface Streamlit** fonctionne avec la nouvelle architecture

---

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

## 🔧 Core Services Architecture

### Service Layer Overview
The application follows a clean service-oriented architecture with clear separation of concerns:

**Core Services (`src/services/`):**
- `llm_manager.py` - Multi-provider LLM integration (OpenAI, Anthropic, Ollama) with LangSmith tracking
- `rag_engine.py` - ChromaDB vector store with semantic search
- `qcm_generator.py` - Main question generation logic with progressive workflow
- `pdf_processor.py` - PDF text extraction and document chunking
- `theme_extractor.py` - LLM-based theme detection with fallback mechanisms
- `validator.py` - Question structure and content validation
- `progressive_workflow.py` - 1→5→all generation workflow management

**Specialized Generators:**
- `title_based_generator.py` - Title-specific question generation
- `chunk_based_generator.py` - Chunk-based question generation
- `enhanced_qcm_generator.py` - Enhanced generation with diversity controls

**Supporting Services:**
- `question_prompt_builder.py` - Dynamic prompt construction with few-shot examples
- `simple_examples_loader.py` - Few-shot example management
- `langsmith_tracker.py` - LangSmith integration for LLM call tracking
- `document_manager.py` - Document lifecycle management
- `progress_tracker.py` - Real-time progress tracking

### Key Architectural Patterns

**Progressive Generation Workflow:**
1. **Initial Test** (1 question) → user validation
2. **Small Batch** (5 questions) → manual review
3. **Full Generation** → automated completion
4. **Continuous Validation** at each step

**Multi-Provider LLM Support:**
- **Default**: `gpt-4o-mini` (OpenAI) for cost-effectiveness
- **High Quality**: `gpt-4o` for complex questions
- **Local**: Ollama integration for on-premise deployment
- **Fallback Chain**: Automatic provider switching on failures

**RAG-Enhanced Generation:**
- ChromaDB vector store with semantic similarity search
- Context-aware question generation using document chunks
- Theme-filtered retrieval for targeted question generation

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

### Essential Development Commands

**Setup & Environment:**
```bash
make dev-setup        # Full development environment setup (install + dirs + db)
make install-dev      # Install dependencies + pre-commit hooks
```

**Running the Application:**
```bash
make run-app          # Start complete app (API + UI) - RECOMMENDED
make run-app-debug    # Start complete app in debug mode
make run              # Start FastAPI server only
make run-ui           # Start Streamlit interface only
make check-setup      # Verify development setup
```

**Testing (26 tests, all passing):**
```bash
make test-working     # Run core working tests (21 tests) - RECOMMENDED
make test-basic       # Run basic functionality tests (6 tests)
make test             # Run all tests with coverage
make test-simple      # Run all tests without coverage
```

**Code Quality:**
```bash
make format           # Format code (black + ruff)
make lint             # Run linting (ruff + mypy)
make quick-check      # Fast development check (format + lint + fast tests)
make full-check       # Complete development check (format + lint + all tests)
```

**Database Management:**
```bash
make db-init          # Initialize database
make db-reset         # Reset database completely
make db-migrate       # Run database migrations
```

**Docker Deployment:**
```bash
make docker-run       # Run containerized app (GPU-enabled)
make docker-run-cpu   # Run containerized app (CPU-only)
make docker-logs      # View all container logs
make docker-shell     # Shell access to main container
make docker-health    # Check container health status
make docker-clean     # Clean Docker resources
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

### Current Branch: `clean_architecture`
Cette branche contient la **réorganisation complète des services** en domaines métier (Janvier 2025).

**Changements Majeurs :**
- ✅ **Réorganisation Services** : 21 services → 5 domaines métier
- ✅ **Architecture Propre** : Séparation claire des responsabilités
- ✅ **Imports Mis à Jour** : 50+ imports corrigés dans toute la codebase
- ✅ **Tests Validés** : 21/21 tests passent avec la nouvelle architecture
- ✅ **Scripts Corrigés** : Tous les scripts dans `/scripts/` utilisent les nouveaux chemins

**Branches Précédentes :**
- `questions_fewshots` : Few-shot learning + LangSmith tracking ✅
- `fix_ollama` : Correction intégration Ollama + OpenAI ✅

**Services Réorganisés par Domaine :**
- 📄 **document/** : PDF processing, theme extraction (4 services)
- ⚡ **generation/** : QCM generation, workflows, prompts (8 services)
- ✅ **quality/** : Validation, deduplication, diversity (4 services)
- 🤖 **llm/** : LLM providers, tracking, examples (3 services)
- 🔧 **infrastructure/** : RAG engine, progress tracking (2 services)

### Few-Shot Learning Integration
- **Examples Storage**: `data/few_shot_examples/` contains JSON files with domain-specific examples
- **Dynamic Loading**: `simple_examples_loader.py` loads examples based on document content
- **Prompt Enhancement**: Examples are integrated into prompts for better question quality
- **Tracking**: LangSmith tracks example usage and effectiveness

### Quality Validation Pipeline
1. **Structure Validation**: JSON parsing, option count (3-6), correct answer format
2. **Content Validation**: Question clarity, option diversity, explanation quality
3. **Deduplication**: Prevent similar questions using semantic similarity
4. **Language Consistency**: Maintain consistent language throughout questions

### Export Formats
**Udemy CSV Format:**
```csv
question,answer_1,answer_2,answer_3,answer_4,correct_answer,explanation
"What is Python?","A snake","A programming language","A fruit","A tool","2","Python is a programming language..."
```

**JSON Format (with metadata):**
```json
{
  "question": "What is Python?",
  "options": ["A snake", "A programming language", "A fruit", "A tool"],
  "correct_answers": [1],
  "explanation": "Python is a programming language...",
  "metadata": {"theme": "Programming", "difficulty": "easy"}
}
```

### Environment Configuration
- **Primary LLM**: OpenAI GPT-4o-mini (cost-effective default)
- **Database**: SQLite with ChromaDB vector store
- **Few-Shot Examples**: JSON files in `data/few_shot_examples/`
- **LangSmith**: Optional tracking for production monitoring

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

---

## 🚀 **REACT MIGRATION PLAN** *(Janvier 2025)*

### Phase 1: API Layer Enhancement (Semaines 1-2)
**Objectifs :**
- Améliorer les endpoints FastAPI pour React
- Ajouter WebSocket pour suivi temps réel
- Configuration CORS pour frontend React

**Nouvelles API nécessaires :**
```python
# Real-time progress WebSocket
@router.websocket("/ws/progress/{session_id}")
async def progress_websocket(websocket: WebSocket, session_id: str)

# Document chunk preview pour React
@router.get("/documents/{doc_id}/chunks") 
async def get_document_chunks(doc_id: int)

# Theme-based generation
@router.post("/generation/by-theme")
async def generate_by_theme(theme_config: ThemeGenerationConfig)
```

### Phase 2: React Frontend Development (Semaines 3-6)
**Structure Frontend :**
```
frontend/
├── src/
│   ├── components/ui/         # Shadcn/ui components
│   ├── components/documents/  # Document management
│   ├── components/generation/ # QCM generation
│   ├── pages/                 # Pages React Router
│   ├── hooks/                 # Custom React hooks
│   ├── lib/                   # API client (TanStack Query)
│   └── types/                 # TypeScript definitions
```

**Technologies :**
- React 18 + TypeScript
- Shadcn/ui components (design system moderne)
- TanStack Query (state management API)
- React Router (navigation)
- WebSocket (temps réel)

### Phase 3: Component Migration (Semaines 7-10)
**Composants prioritaires :**
1. **DocumentUpload** : Upload + configuration chunking
2. **ProgressiveGenerator** : Workflow 1→5→all questions
3. **DocumentDisplay** : Visualisation chunks et thèmes
4. **ExportInterface** : Export CSV/JSON avec téléchargement

**Avantages Architecture Clean :**
- Services `src/services/` → API business logic
- Logique métier déjà séparée de l'UI
- Réutilisation des services existants via API

### Phase 4: Integration & Deployment (Semaines 11-12)
**Docker Configuration :**
```yaml
services:
  backend:    # FastAPI existant (inchangé)
  frontend:   # Nouveau service React
    ports: ["3000:3000"]
    environment:
      - REACT_APP_API_URL=http://localhost:8000
  ollama:     # Services existants (inchangés)
  redis:      # Services existants (inchangés)
```

**Migration Progressive :**
- Streamlit (:8501) et React (:3000) en parallèle
- Tests Playwright pour validation comparative
- Transition graduelle sans interruption de service

### Avantages de la Migration
✅ **UX Moderne** : Interface responsive, animations fluides
✅ **Performance** : Rendu côté client, lazy loading, optimisations
✅ **Maintenance** : TypeScript, composants modulaires, tests automatisés
✅ **Évolutivité** : Ecosystem React mature, extensions faciles
✅ **Temps Réel** : WebSocket natif vs polling Streamlit

### Tests Playwright
- **Comparaison Visuelle** : Screenshots automatiques Streamlit vs React
- **Tests Fonctionnels** : Validation parité features entre interfaces
- **Tests Performance** : Mesure temps de chargement et réactivité
- **Tests d'Intégration** : Workflows complets upload→génération→export

**Remaining Optional Tasks**: 
- ⏳ Enhanced multilingual prompt templates (currently basic FR/EN support)
- ⏳ Advanced performance optimization for large document processing
- ⏳ Extended testing coverage for Docker deployment scenarios

---

## 🔄 Development Workflow

### Quick Start for New Features
1. **Create todo list** using TodoWrite tool for complex tasks
2. **Branch from current**: Base work on `questions_fewshots` branch
3. **Run setup check**: `make check-setup` to verify environment
4. **Test existing functionality**: `make test-working` (21 tests should pass)
5. **Implement changes** following service-oriented architecture
6. **Validate changes**: Run `make quick-check` for fast validation
7. **Full validation**: Run `make full-check` before commits

### Code Modification Guidelines
- **Service Layer**: Add new functionality to appropriate service in `src/services/`
- **API Endpoints**: Add routes to appropriate module in `src/api/routes/`
- **Database Changes**: Update models in `src/models/database.py` and schemas in `src/models/schemas.py`
- **Configuration**: Add settings to `src/core/config.py` with environment variable support
- **UI Changes**: Modify `src/ui/streamlit_app.py` (currently 1200+ lines, needs refactoring)

### Testing Strategy
- **Unit Tests**: Focus on individual service functionality
- **Integration Tests**: Test complete workflows (PDF → QCM generation → Export)
- **Test Data**: Use fixtures in `tests/fixtures/` and `data/few_shot_examples/`
- **Coverage**: Maintain >90% coverage for core services

### Current Priorities (from To Do List)
1. **Question Quality**: Integrate more sophisticated few-shot examples and improve "hands-on" question generation
2. **Code Refactoring**: Split large Streamlit app into smaller components
3. **Performance**: Optimize PDF processing for documents with wide fonts/spacing issues
4. **Architecture**: Implement Clean Architecture with SOLID principles
5. **Testing**: Expand test coverage and improve integration tests

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