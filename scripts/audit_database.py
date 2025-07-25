#!/usr/bin/env python3
"""
Script d'audit de la base de données SQLite pour comprendre l'état actuel
"""

import sys
from pathlib import Path
import sqlite3
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def audit_database():
    """Audit the current SQLite database."""
    print("🔍 Audit de la Base de Données SQLite")
    print("=" * 60)
    
    db_path = project_root / "data" / "database" / "qcm_generator.db"
    
    if not db_path.exists():
        print("❌ Base de données non trouvée")
        return
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"📚 Tables trouvées: {len(tables)}")
        
        for table_name in tables:
            table = table_name[0]
            print(f"\n📋 Table: {table}")
            print("-" * 40)
            
            # Get table info
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            
            print("   Colonnes:")
            for col in columns:
                print(f"      - {col[1]} ({col[2]})")
            
            # Count records
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            count = cursor.fetchone()[0]
            print(f"   📊 Nombre d'enregistrements: {count}")
            
            # Show sample data if exists
            if count > 0:
                cursor.execute(f"SELECT * FROM {table} LIMIT 3;")
                samples = cursor.fetchall()
                
                print("   📄 Échantillons:")
                for i, sample in enumerate(samples, 1):
                    print(f"      Sample {i}: {sample}")
        
        # Specific analysis for documents
        print(f"\n{'='*60}")
        print("📄 ANALYSE DES DOCUMENTS")
        print("=" * 60)
        
        try:
            cursor.execute("""
                SELECT id, filename, processing_status, total_pages, language, upload_date 
                FROM documents 
                ORDER BY upload_date DESC;
            """)
            documents = cursor.fetchall()
            
            if documents:
                print(f"✅ {len(documents)} document(s) trouvé(s):")
                for doc in documents:
                    doc_id, filename, status, pages, lang, upload_date = doc
                    print(f"   📄 [{doc_id}] {filename}")
                    print(f"      Status: {status}, Pages: {pages}, Lang: {lang}")
                    print(f"      Upload: {upload_date}")
                    print()
            else:
                print("❌ Aucun document trouvé en base")
        except Exception as e:
            print(f"⚠️  Table documents non accessible: {e}")
        
        # Specific analysis for themes
        print("🎯 ANALYSE DES THÈMES")
        print("=" * 40)
        
        try:
            cursor.execute("""
                SELECT dt.id, dt.document_id, dt.theme_name, dt.confidence_score, d.filename
                FROM document_themes dt
                JOIN documents d ON dt.document_id = d.id
                ORDER BY dt.confidence_score DESC;
            """)
            themes = cursor.fetchall()
            
            if themes:
                print(f"✅ {len(themes)} thème(s) trouvé(s):")
                for theme in themes:
                    theme_id, doc_id, name, confidence, filename = theme
                    print(f"   🎯 {name} (confidence: {confidence:.2f})")
                    print(f"      Document: {filename} [ID: {doc_id}]")
                    print()
            else:
                print("❌ Aucun thème trouvé en base")
        except Exception as e:
            print(f"⚠️  Table themes non accessible: {e}")
        
        # Specific analysis for chunks
        print("📝 ANALYSE DES CHUNKS")
        print("=" * 40)
        
        try:
            cursor.execute("""
                SELECT dc.id, dc.document_id, dc.chunk_index, dc.content_preview, d.filename
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                ORDER BY dc.document_id, dc.chunk_index
                LIMIT 10;
            """)
            chunks = cursor.fetchall()
            
            if chunks:
                print(f"✅ Chunks trouvés (10 premiers):")
                for chunk in chunks:
                    chunk_id, doc_id, index, preview, filename = chunk
                    preview_text = preview[:100] + "..." if preview and len(preview) > 100 else preview
                    print(f"   📝 Chunk {index} from {filename}")
                    print(f"      Preview: {preview_text}")
                    print()
            else:
                print("❌ Aucun chunk trouvé en base")
        except Exception as e:
            print(f"⚠️  Table chunks non accessible: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Erreur lors de l'audit: {e}")


def check_current_workflow():
    """Check how the current application uses the database."""
    print(f"\n{'='*60}")
    print("🔧 ANALYSE DU WORKFLOW ACTUEL")
    print("=" * 60)
    
    try:
        # Check if services use the database
        from src.services.pdf_processor import process_pdf
        from src.services.rag_engine import get_rag_engine
        
        print("✅ Services importés avec succès")
        
        # Check RAG engine type
        rag_engine = get_rag_engine()
        rag_type = type(rag_engine).__name__
        print(f"📊 Type de RAG Engine: {rag_type}")
        
        if hasattr(rag_engine, 'document_chunks'):
            chunks_count = len(rag_engine.document_chunks)
            print(f"📝 Chunks en mémoire: {chunks_count}")
        
        # Check if database connection works
        try:
            from src.core.config import settings
            print(f"🔧 DB URL configurée: {settings.database.url}")
        except Exception as e:
            print(f"⚠️  Configuration DB: {e}")
            
    except Exception as e:
        print(f"❌ Erreur analyse workflow: {e}")


def generate_migration_plan():
    """Generate a migration plan based on current state."""
    print(f"\n{'='*60}")
    print("📋 PLAN DE MIGRATION RECOMMANDÉ")
    print("=" * 60)
    
    print("""
🎯 OBJECTIFS:
   1. Rendre les documents et thèmes persistants et réutilisables
   2. Intégrer ChromaDB pour la persistance des chunks RAG
   3. Créer une interface de gestion des documents existants

📋 PHASES RECOMMANDÉES:

Phase 1: PERSISTANCE RAG (PRIORITÉ HAUTE)
   ├── Migrer SimpleRAGEngine vers ChromaDBRAGEngine
   ├── Sauvegarder les chunks dans ChromaDB lors du traitement
   ├── Charger les chunks depuis ChromaDB au démarrage
   └── Tester la cohérence données SQLite ↔ ChromaDB

Phase 2: INTERFACE DOCUMENTS EXISTANTS (PRIORITÉ MOYENNE)
   ├── Ajouter onglet "Documents Existants" dans l'UI
   ├── Lister les documents traités avec leurs thèmes
   ├── Permettre sélection/désélection pour génération
   └── Bouton de suppression des documents

Phase 3: RÉUTILISATION THÈMES (PRIORITÉ MOYENNE)
   ├── Interface de sélection des thèmes existants
   ├── Combinaison thèmes de plusieurs documents
   ├── Sauvegarde des préférences de thèmes
   └── Export/Import des configurations de thèmes

Phase 4: OPTIMISATIONS (PRIORITÉ BASSE)
   ├── Cache intelligent des embeddings
   ├── Déduplication des documents identiques
   ├── Compression/archivage des anciens documents
   └── Statistiques d'utilisation

💡 POINTS D'ATTENTION:
   - Backup de la DB avant migration
   - Tests avec documents existants
   - Interface progressive (ne pas tout casser)
   - Performance avec beaucoup de documents
""")


if __name__ == "__main__":
    audit_database()
    check_current_workflow()
    generate_migration_plan()