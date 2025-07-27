#!/usr/bin/env python3
"""
Debug chunk to title mapping issue.
"""

import sys
import os
from pathlib import Path

# Add src to Python path and set up module structure
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

from src.models.schemas import TitleStructureConfig
from src.services.title_detector import TitleDetector, build_chunk_title_hierarchies


def debug_chunk_title_mapping():
    """Debug why chunks are getting wrong title assignments."""
    print("🔧 Debug Chunk to Title Mapping")
    print("=" * 50)
    
    # Simulate DP-600 document structure
    test_text = """
--- Page 1 ---
Introduction to DP-600 Course
This course will cover Microsoft Fabric.

Parcours 1: Bien démarrer avec Microsoft Fabric
Welcome to the first learning path.

Module 1: Comprendre Microsoft Fabric
Understanding the basics of Microsoft Fabric.

Unité 1: Explorer les équipes de données
Learn about data teams and collaboration.

--- Page 2 ---
Type: Contenu
Explorer les équipes de données et Microsoft Fabric
La plateforme d'analytique des données unifiée de Microsoft Fabric facilite la collaboration des professionnels des données sur des projets.

Unité 2: Comprendre l'architecture
Understanding the platform architecture.

--- Page 3 ---
More content for Unité 2.
Technical details and examples.

--- Page 149 ---
Parcours 4: Implémenter un lakehouse avec Microsoft
This is much later in the document.

Module 15: Advanced Lakehouse Concepts
Advanced topics for experienced users.
"""
    
    # Create pages data that reflects actual page structure
    pages_data = [
        {
            'page_number': 1,
            'text': """Introduction to DP-600 Course
This course will cover Microsoft Fabric.

Parcours 1: Bien démarrer avec Microsoft Fabric
Welcome to the first learning path.

Module 1: Comprendre Microsoft Fabric
Understanding the basics of Microsoft Fabric.

Unité 1: Explorer les équipes de données
Learn about data teams and collaboration.""",
            'char_count': 300,
            'word_count': 50
        },
        {
            'page_number': 2,
            'text': """Type: Contenu
Explorer les équipes de données et Microsoft Fabric
La plateforme d'analytique des données unifiée de Microsoft Fabric facilite la collaboration des professionnels des données sur des projets.

Unité 2: Comprendre l'architecture
Understanding the platform architecture.""",
            'char_count': 250,
            'word_count': 40
        },
        {
            'page_number': 3,
            'text': """More content for Unité 2.
Technical details and examples.""",
            'char_count': 60,
            'word_count': 8
        },
        {
            'page_number': 149,
            'text': """Parcours 4: Implémenter un lakehouse avec Microsoft
This is much later in the document.

Module 15: Advanced Lakehouse Concepts
Advanced topics for experienced users.""",
            'char_count': 150,
            'word_count': 25
        }
    ]
    
    # Create chunks (simulating how PDF processor would chunk this)
    chunks = [
        "Introduction to DP-600 Course\nThis course will cover Microsoft Fabric.\n\nParcours 1: Bien démarrer avec Microsoft Fabric\nWelcome to the first learning path.",
        "Explorer les équipes de données et Microsoft Fabric\nLa plateforme d'analytique des données unifiée de Microsoft Fabric facilite la collaboration des professionnels des données sur des projets.",
        "More content for Unité 2.\nTechnical details and examples.",
        "Parcours 4: Implémenter un lakehouse avec Microsoft\nThis is much later in the document."
    ]
    
    print(f"📄 Test document created:")
    print(f"  - {len(pages_data)} pages")
    print(f"  - {len(chunks)} chunks")
    
    # Test with intelligent configuration
    config = TitleStructureConfig(
        h1_patterns=["Parcours 1"],
        h2_patterns=["Module 1"],
        h3_patterns=["Unité 1"],
        h4_patterns=[],
        use_auto_detection=False
    )
    
    print(f"\n🧪 Testing Current Implementation:")
    
    # Get title hierarchies using current implementation
    hierarchies = build_chunk_title_hierarchies(chunks, test_text, pages_data, config)
    
    print(f"\nChunk-to-Title Mapping (CURRENT):")
    for i, (chunk, hierarchy) in enumerate(zip(chunks, hierarchies)):
        chunk_preview = (chunk[:50] + "...") if len(chunk) > 50 else chunk
        chunk_preview = chunk_preview.replace('\n', ' ')
        
        print(f"\nChunk {i+1}: {chunk_preview}")
        print(f"  H1: {hierarchy.h1_title or 'None'}")
        print(f"  H2: {hierarchy.h2_title or 'None'}")
        print(f"  H3: {hierarchy.h3_title or 'None'}")
        print(f"  Full Path: {hierarchy.get_full_path() or 'None'}")
    
    # Expected mappings
    print(f"\n✅ Expected Mappings:")
    print("Chunk 1 (page 1): Parcours 1 > Module 1 > Unité 1")
    print("Chunk 2 (page 2): Parcours 1 > Module 1 > Unité 2") 
    print("Chunk 3 (page 3): Parcours 1 > Module 1 > Unité 2")
    print("Chunk 4 (page 149): Parcours 4 > Module 15")
    
    # Check if chunk 2 is incorrectly assigned to Parcours 4
    chunk2_hierarchy = hierarchies[1] if len(hierarchies) > 1 else None
    if chunk2_hierarchy and chunk2_hierarchy.h1_title and "Parcours 4" in chunk2_hierarchy.h1_title:
        print(f"\n❌ PROBLEM CONFIRMED!")
        print("Chunk 2 (page 2) is incorrectly assigned to Parcours 4 (page 149)")
        print("This is the exact issue reported by the user.")
        return False
    else:
        print(f"\n✅ Mapping appears correct")
        return True


if __name__ == "__main__":
    success = debug_chunk_title_mapping()
    if not success:
        print(f"\n🔧 Need to fix the position mapping algorithm!")
    else:
        print(f"\n🎉 Position mapping working correctly!")