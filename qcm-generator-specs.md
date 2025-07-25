# Cahier des Charges - QCM Generator Pro (Local Edition)
## Spécifications Techniques pour Claude Code

---

## 1. Vue d'ensemble

### Description
Application locale de génération automatique de QCM multilingues à partir de documents PDF, avec support de modèles LLM locaux (RTX 4090) et APIs cloud.

### Objectifs Techniques
- Architecture modulaire et testable
- Support multilingue natif (FR/EN + extensible)
- Validation progressive des questions
- Découpage thématique intelligent
- CI/CD automatisé avec GitHub Actions

### Stack Technique
```yaml
Backend:
  - Python 3.11+
  - FastAPI + Pydantic v2
  - Langchain 0.1.0+
  - ChromaDB (vectorstore local)
  - SQLite + SQLAlchemy (métadonnées)
  
Frontend:
  - Streamlit 4.0+ (interface simple)
  
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

## 2. Architecture du Projet

### Structure des Fichiers
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
│   ├── setup_local_models.py  # Installation modèles
│   └── migrate_db.py          # Migrations DB
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── Makefile
```

---

## 3. Modèles de Données

### Base de Données (SQLite avec support JSON)

```python
# models/database.py
from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

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
    document = relationship("Document", back_populates="themes")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    chunk_index = Column(Integer)
    content = Column(Text, nullable=False)
    theme_id = Column(Integer, ForeignKey("document_themes.id"))
    page_number = Column(Integer)
    metadata = Column(JSON)  # {section_title, is_title, formatting, etc.}
    document = relationship("Document", back_populates="chunks")
    theme = relationship("DocumentTheme")

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
    metadata = Column(JSON)  # {source_chunks, generation_model, score, etc.}
    validation_status = Column(String(20))  # pending, validated, rejected
    created_at = Column(DateTime)

class FewShotExample(Base):
    __tablename__ = "few_shot_examples"
    
    id = Column(Integer, primary_key=True)
    domain = Column(String(100))
    language = Column(String(10))
    context = Column(Text)
    question_data = Column(JSON)  # Structure complète de la question
    quality_score = Column(Float)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime)

class GenerationSession(Base):
    __tablename__ = "generation_sessions"
    
    id = Column(String(50), primary_key=True)  # UUID
    document_ids = Column(JSON)  # Liste des IDs de documents
    config = Column(JSON)  # Configuration complète
    status = Column(String(50))  # pending, processing, completed, failed
    progress = Column(JSON)  # {current: 5, total: 20, phase: "validation"}
    created_at = Column(DateTime)
    completed_at = Column(DateTime, nullable=True)
```

### Schemas Pydantic

```python
# models/schemas.py
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

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

class ThemeDetection(BaseModel):
    theme_name: str
    confidence: float = Field(ge=0, le=1)
    keywords: List[str]
    page_range: tuple[int, int]

class ChunkMetadata(BaseModel):
    page_number: int
    theme: Optional[str] = None
    section_title: Optional[str] = None
    is_title: bool = False
    formatting: Dict[str, Any] = {}

class QuestionGeneration(BaseModel):
    question_text: str
    question_type: QuestionType
    options: List[str] = Field(min_length=3, max_length=6)
    correct_answers: List[int]
    explanation: Optional[str] = None
    difficulty: Difficulty
    theme: str
    language: Language
    metadata: Dict[str, Any] = {}

class GenerationConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
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

## 4. Services Principaux

### 4.1 Extracteur de Thèmes

```python
# services/theme_extractor.py
from typing import List, Dict, Any
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

class ThemeExtractor:
    """
    Extrait automatiquement les thèmes d'un document PDF
    en analysant la structure et le contenu
    """
    
    def __init__(self, language: str = "fr"):
        self.language = language
        self.nlp = self._load_spacy_model(language)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 3),
            stop_words=self._get_stopwords(language)
        )
    
    def extract_themes(self, chunks: List[Dict[str, Any]]) -> List[ThemeDetection]:
        """
        Identifie les thèmes principaux dans les chunks
        """
        # 1. Détection basée sur la structure (titres, sections)
        structural_themes = self._extract_structural_themes(chunks)
        
        # 2. Clustering basé sur le contenu
        content_themes = self._extract_content_themes(chunks)
        
        # 3. Fusion et scoring
        merged_themes = self._merge_themes(structural_themes, content_themes)
        
        return merged_themes
    
    def _extract_structural_themes(self, chunks):
        """Analyse les titres et la structure du document"""
        themes = []
        current_theme = None
        
        for i, chunk in enumerate(chunks):
            # Détection de titre (police plus grande, etc.)
            if chunk['metadata'].get('is_title'):
                if current_theme:
                    themes.append(current_theme)
                
                current_theme = {
                    'name': chunk['content'].strip(),
                    'start_page': chunk['metadata']['page_number'],
                    'chunks': [i]
                }
            elif current_theme:
                current_theme['chunks'].append(i)
        
        return themes
    
    def _extract_content_themes(self, chunks):
        """Clustering basé sur la similarité du contenu"""
        # Vectorisation TF-IDF
        texts = [chunk['content'] for chunk in chunks]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Clustering
        num_clusters = min(10, len(chunks) // 5)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Extraction des thèmes par cluster
        themes = []
        for cluster_id in range(num_clusters):
            cluster_chunks = [i for i, c in enumerate(clusters) if c == cluster_id]
            
            # Mots-clés du cluster
            cluster_texts = [texts[i] for i in cluster_chunks]
            keywords = self._extract_keywords(cluster_texts)
            
            themes.append({
                'name': self._generate_theme_name(keywords),
                'chunks': cluster_chunks,
                'keywords': keywords
            })
        
        return themes
```

### 4.2 Générateur avec Validation Progressive

```python
# services/qcm_generator.py
from typing import List, Optional, Dict, Any
import asyncio
from langchain.callbacks import StreamingStdOutCallbackHandler

class ProgressiveQCMGenerator:
    """
    Génère des questions avec validation progressive
    (1 → 5 → toutes)
    """
    
    def __init__(
        self,
        llm_manager: LLMManager,
        rag_engine: RAGEngine,
        validator: QuestionValidator
    ):
        self.llm = llm_manager
        self.rag = rag_engine
        self.validator = validator
        self.batch_sizes = [1, 5, -1]  # -1 = remaining
    
    async def generate_progressive(
        self,
        config: GenerationConfig,
        session_id: str,
        callback=None
    ) -> List[QuestionGeneration]:
        """
        Génération progressive avec validation à chaque étape
        """
        all_questions = []
        remaining = config.num_questions
        
        for batch_idx, batch_size in enumerate(self.batch_sizes):
            if remaining <= 0:
                break
            
            # Déterminer la taille du batch
            current_batch_size = (
                remaining if batch_size == -1 
                else min(batch_size, remaining)
            )
            
            # Mise à jour du statut
            if callback:
                await callback({
                    'phase': f'batch_{batch_idx + 1}',
                    'batch_size': current_batch_size,
                    'total_generated': len(all_questions)
                })
            
            # Génération du batch
            batch_questions = await self._generate_batch(
                config, 
                current_batch_size,
                exclude_questions=all_questions
            )
            
            # Validation automatique
            validated_questions = []
            for question in batch_questions:
                is_valid, issues = self.validator.validate_question(question)
                if is_valid:
                    validated_questions.append(question)
                else:
                    # Log les problèmes
                    print(f"Question rejetée: {issues}")
            
            # Pause pour validation manuelle (sauf dernier batch)
            if batch_idx < len(self.batch_sizes) - 1 and callback:
                approval = await callback({
                    'phase': 'validation_pause',
                    'questions': validated_questions,
                    'awaiting_approval': True
                })
                
                if not approval.get('continue', True):
                    break
            
            all_questions.extend(validated_questions)
            remaining -= current_batch_size
        
        return all_questions
    
    async def _generate_batch(
        self,
        config: GenerationConfig,
        batch_size: int,
        exclude_questions: List[QuestionGeneration]
    ) -> List[QuestionGeneration]:
        """Génère un batch de questions"""
        questions = []
        
        # Récupération des thèmes à couvrir
        themes = await self._get_themes_distribution(config, batch_size)
        
        for theme, count in themes.items():
            # Contexte spécifique au thème
            theme_chunks = self.rag.similarity_search(
                query=theme,
                k=5,
                filter={'theme': theme}
            )
            
            # Génération pour ce thème
            theme_questions = await self._generate_for_theme(
                theme=theme,
                context=theme_chunks,
                count=count,
                config=config,
                exclude_questions=exclude_questions
            )
            
            questions.extend(theme_questions)
        
        return questions
```

### 4.3 Support Multilingue

```python
# prompts/languages/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class LanguageTemplate(ABC):
    """Template de base pour les prompts multilingues"""
    
    @abstractmethod
    def get_question_generation_prompt(self) -> str:
        pass
    
    @abstractmethod
    def get_distractor_generation_prompt(self) -> str:
        pass
    
    @abstractmethod
    def get_validation_prompt(self) -> str:
        pass
    
    @abstractmethod
    def get_theme_extraction_prompt(self) -> str:
        pass

# prompts/languages/fr.py
class FrenchTemplate(LanguageTemplate):
    def get_question_generation_prompt(self) -> str:
        return """
        Contexte du cours :
        {context}
        
        Thème : {theme}
        
        Génère une question de type {question_type} en français.
        Niveau de difficulté : {difficulty}
        
        La question doit :
        1. Tester la compréhension, pas seulement la mémorisation
        2. Être claire et sans ambiguïté
        3. Avoir entre {min_options} et {max_options} options
        4. Inclure des distracteurs plausibles basés sur des erreurs courantes
        
        Format JSON attendu :
        {{
            "question": "texte de la question",
            "options": ["option1", "option2", ...],
            "correct_answers": [indices des bonnes réponses],
            "explanation": "explication de la réponse",
            "difficulty": "niveau",
            "theme": "thème"
        }}
        """

# prompts/languages/en.py
class EnglishTemplate(LanguageTemplate):
    def get_question_generation_prompt(self) -> str:
        return """
        Course context:
        {context}
        
        Theme: {theme}
        
        Generate a {question_type} question in English.
        Difficulty level: {difficulty}
        
        The question must:
        1. Test understanding, not just memorization
        2. Be clear and unambiguous
        3. Have between {min_options} and {max_options} options
        4. Include plausible distractors based on common mistakes
        
        Expected JSON format:
        {{
            "question": "question text",
            "options": ["option1", "option2", ...],
            "correct_answers": [indices of correct answers],
            "explanation": "answer explanation",
            "difficulty": "level",
            "theme": "theme"
        }}
        """
```

---

## 5. Tests Unitaires

### Structure des Tests

```python
# tests/conftest.py
import pytest
from pathlib import Path
import tempfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def test_db():
    """Base de données en mémoire pour les tests"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()

@pytest.fixture
def sample_pdf():
    """PDF de test"""
    return Path(__file__).parent / "fixtures" / "sample.pdf"

@pytest.fixture
def mock_llm():
    """Mock du LLM pour les tests"""
    class MockLLM:
        async def ainvoke(self, prompt):
            return {
                "question": "Question test",
                "options": ["A", "B", "C", "D"],
                "correct_answers": [0],
                "explanation": "Test explanation"
            }
    return MockLLM()

# tests/unit/test_theme_extractor.py
import pytest
from src.services.theme_extractor import ThemeExtractor

class TestThemeExtractor:
    def test_extract_themes_from_chunks(self):
        """Test l'extraction de thèmes depuis des chunks"""
        extractor = ThemeExtractor(language="fr")
        
        chunks = [
            {
                'content': 'Introduction à Python',
                'metadata': {'is_title': True, 'page_number': 1}
            },
            {
                'content': 'Python est un langage de programmation...',
                'metadata': {'is_title': False, 'page_number': 1}
            }
        ]
        
        themes = extractor.extract_themes(chunks)
        
        assert len(themes) > 0
        assert themes[0].theme_name == "Introduction à Python"
        assert themes[0].confidence > 0.5
    
    def test_multilingual_support(self):
        """Test le support multilingue"""
        for lang in ["fr", "en"]:
            extractor = ThemeExtractor(language=lang)
            assert extractor.language == lang

# tests/unit/test_validator.py
class TestQuestionValidator:
    def test_validate_question_structure(self):
        """Test la validation de la structure d'une question"""
        validator = QuestionValidator()
        
        valid_question = QuestionGeneration(
            question_text="Qu'est-ce que Python?",
            question_type=QuestionType.MULTIPLE_CHOICE,
            options=["Un serpent", "Un langage", "Un fruit", "Un outil"],
            correct_answers=[1],
            difficulty=Difficulty.EASY,
            theme="Introduction",
            language=Language.FR
        )
        
        is_valid, issues = validator.validate_question(valid_question)
        assert is_valid
        assert len(issues) == 0
    
    def test_reject_invalid_options_count(self):
        """Test le rejet si nombre d'options incorrect"""
        validator = QuestionValidator()
        
        invalid_question = QuestionGeneration(
            question_text="Question?",
            question_type=QuestionType.MULTIPLE_CHOICE,
            options=["A", "B"],  # Trop peu
            correct_answers=[0],
            difficulty=Difficulty.EASY,
            theme="Test",
            language=Language.FR
        )
        
        is_valid, issues = validator.validate_question(invalid_question)
        assert not is_valid
        assert "options" in str(issues)
```

### Tests d'Intégration

```python
# tests/integration/test_generation_flow.py
import pytest
from fastapi.testclient import TestClient

class TestGenerationFlow:
    @pytest.mark.asyncio
    async def test_progressive_generation(self, client: TestClient, sample_pdf):
        """Test le flow complet de génération progressive"""
        # 1. Upload PDF
        with open(sample_pdf, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": f}
            )
        assert response.status_code == 200
        doc_id = response.json()["id"]
        
        # 2. Attendre le processing
        response = client.get(f"/api/documents/{doc_id}/status")
        assert response.json()["status"] == "completed"
        
        # 3. Lancer génération progressive
        config = {
            "num_questions": 10,
            "language": "fr",
            "validation_mode": "progressive",
            "model": "mistral-local"
        }
        
        response = client.post(
            "/api/generation/start",
            json={"document_ids": [doc_id], "config": config}
        )
        assert response.status_code == 200
        session_id = response.json()["session_id"]
        
        # 4. Vérifier les batches
        response = client.get(f"/api/generation/{session_id}/progress")
        progress = response.json()
        
        assert progress["phase"] == "batch_1"
        assert progress["current"] == 1
```

---

## 6. CI/CD avec GitHub Actions

### Workflow Principal

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run linting
      run: |
        ruff check src tests
        black --check src tests
    
    - name: Run type checking
      run: |
        mypy src
    
    - name: Run tests with coverage
      run: |
        pytest tests/unit -v --cov=src --cov-report=xml
        pytest tests/integration -v
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: |
        docker build -t qcm-generator:latest .
    
    - name: Run security scan
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: qcm-generator:latest
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

### Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]
      
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.11
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-json
      - id: check-merge-conflict
  
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]
```

---

## 7. Configuration et Variables d'Environnement

### Configuration Pydantic

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Dict, List, Optional
from pathlib import Path

class Settings(BaseSettings):
    # Application
    app_name: str = "QCM Generator"
    version: str = "1.0.0"
    debug: bool = False
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = base_dir / "data"
    models_dir: Path = base_dir / "models"
    
    # Database
    database_url: str = f"sqlite:///{data_dir}/qcm_generator.db"
    
    # ChromaDB
    chroma_persist_dir: str = str(data_dir / "vectorstore")
    chroma_collection_name: str = "qcm_documents"
    
    # LLM Configuration
    default_llm: str = "mistral-local"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Local Models
    local_models_config: Dict[str, Dict] = {
        "mistral-7b": {
            "path": "models/mistral-7b-instruct-v0.2.Q5_K_M.gguf",
            "context_length": 4096,
            "gpu_layers": -1
        },
        "llama3-8b": {
            "path": "models/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf",
            "context_length": 8192,
            "gpu_layers": -1
        }
    }
    
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_models: List[str] = ["mistral", "llama3", "phi3"]
    
    # Embeddings
    use_local_embeddings: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cuda"
    
    # Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_pdf_size_mb: int = 50
    
    # Generation
    default_num_questions: int = 20
    max_questions_per_session: int = 250
    batch_sizes: List[int] = [1, 5, -1]
    
    # Export
    export_dir: Path = data_dir / "exports"
    
    # UI
    streamlit_server_port: int = 8501
    streamlit_share: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

---

## 8. Makefile pour Développement

```makefile
# Makefile
.PHONY: help install install-dev test lint format run clean docker-build docker-run

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
APP_NAME := qcm-generator

help:
	@echo "Available commands:"
	@echo "  install       Install dependencies"
	@echo "  install-dev   Install dev dependencies"
	@echo "  test          Run all tests"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-cov      Run tests with coverage"
	@echo "  lint          Run linting"
	@echo "  format        Format code"
	@echo "  run           Run the application"
	@echo "  clean         Clean cache files"
	@echo "  setup-models  Download local models"

install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt
	pre-commit install

test:
	$(PYTEST) tests/ -v

test-unit:
	$(PYTEST) tests/unit/ -v

test-cov:
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint:
	ruff check src tests
	mypy src

format:
	black src tests
	isort src tests

run:
	$(PYTHON) -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	$(PYTHON) src/ui/streamlit_app.py

setup-models:
	$(PYTHON) scripts/setup_local_models.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov .mypy_cache .pytest_cache .ruff_cache

docker-build:
	docker build -t $(APP_NAME):latest .

docker-run:
	docker run -p 8000:8000 -p 8501:8501 -v $(PWD)/data:/app/data $(APP_NAME):latest

# Base de données
db-init:
	$(PYTHON) -c "from src.models.database import init_db; init_db()"

db-upgrade:
	alembic upgrade head

db-reset: clean
	rm -f data/qcm_generator.db
	$(MAKE) db-init
```

---

## 9. README pour Claude Code

```markdown
# QCM Generator Pro - Local Edition

## 🚀 Quick Start

```bash
# 1. Clone et installation
git clone https://github.com/[user]/qcm-generator.git
cd qcm-generator
make install-dev

# 2. Configuration
cp .env.example .env
# Éditer .env avec vos clés API (optionnel)

# 3. Setup modèles locaux (optionnel)
make setup-models

# 4. Lancer l'application
make run        # API backend
make run-ui     # Interface Streamlit

# 5. Tests
make test-cov   # Tests avec couverture
```

## 📋 Features

- ✅ Génération multilingue (FR/EN + extensible)
- ✅ Validation progressive (1 → 5 → toutes)
- ✅ Extraction automatique des thèmes
- ✅ Support LLM locaux (RTX 4090 optimisé)
- ✅ Export CSV Udemy direct
- ✅ Tests unitaires complets
- ✅ CI/CD GitHub Actions

## 🏗️ Architecture

```
src/
├── api/        # FastAPI endpoints
├── core/       # Configuration
├── models/     # DB models + schemas
├── services/   # Business logic
├── prompts/    # Templates multilingues
└── ui/         # Interface Streamlit
```

## 🧪 Tests

```bash
# Unit tests
make test-unit

# Integration tests
make test

# Coverage report
make test-cov
# Ouvrir htmlcov/index.html
```

## 🔧 Configuration

Voir `.env.example` pour toutes les options.

### Modèles recommandés (RTX 4090)
- **Mistral 7B** : Meilleur rapport qualité/vitesse
- **Llama 3 8B** : Excellente qualité
- **Phi-3 Medium** : Ultra rapide

## 📝 Utilisation

1. **Upload PDF** : Glisser dans l'interface
2. **Configuration** : Choisir langue, nombre, modèle
3. **Validation progressive** : 
   - 1 question test
   - 5 questions validation
   - Génération complète
4. **Export** : CSV prêt pour Udemy

## 🐛 Debug

```bash
# Logs détaillés
export LOG_LEVEL=DEBUG
make run

# Reset base de données
make db-reset
```
```

---

Ce cahier des charges est maintenant optimisé pour Claude Code avec :
- Architecture modulaire et testable
- Tests unitaires dès la conception
- CI/CD GitHub Actions complet
- Support multilingue natif
- Validation progressive des questions
- Extraction thématique automatique
- Base SQLite avec JSON pour flexibilité des QCM (3-6 réponses)

Le projet est prêt à être développé avec une structure professionnelle et maintenable !