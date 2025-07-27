#!/usr/bin/env python3
"""
Test custom title configuration system.
"""

import sys
import os
from pathlib import Path

# Add src to Python path and set up module structure
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

from src.models.schemas import TitleStructureConfig
from src.services.title_detector import TitleDetector, build_chunk_title_hierarchies


def test_custom_title_config():
    """Test custom title configuration similar to DP-600 document."""
    print("🔧 Testing Custom Title Configuration")
    print("=" * 60)
    
    # Create test document similar to DP-600 format
    test_text = """
Introduction to Microsoft DP-600
This course covers the complete DP-600 curriculum.

I. Data Engineering Fundamentals
Data engineering is the foundation of modern analytics.

1. Data Storage Concepts
Understanding different storage paradigms.

i. Relational Databases
Traditional RDBMS systems and their uses.

ii. NoSQL Databases  
Modern non-relational database systems.

2. Data Processing Patterns
Different approaches to processing data.

i. Batch Processing
Processing data in large batches.

II. Microsoft Fabric Overview
Microsoft Fabric is a unified analytics platform.

1. Fabric Components
Understanding the key components of Fabric.

i. Data Factory
For data integration and orchestration.

ii. Synapse Data Engineering
For big data processing capabilities.

2. Fabric Administration
Managing and administering Fabric workspaces.

III. Advanced Analytics
Advanced analytics capabilities in the platform.
"""
    
    # Create pages data
    pages_data = [{
        'page_number': 1,
        'text': test_text,
        'char_count': len(test_text),
        'word_count': len(test_text.split())
    }]
    
    print(f"📄 Test document created")
    
    # Test 1: Using Roman numerals configuration
    print(f"\n🧪 Test 1: Roman Numerals Configuration")
    print("-" * 40)
    
    roman_config = TitleStructureConfig(
        h1_patterns=["I.", "II.", "III.", "IV.", "V."],
        h2_patterns=["1.", "2.", "3.", "4.", "5."],
        h3_patterns=["i.", "ii.", "iii.", "iv.", "v."],
        h4_patterns=["a.", "b.", "c.", "d.", "e."],
        use_auto_detection=False
    )
    
    detector = TitleDetector(custom_config=roman_config)
    title_candidates = detector.detect_titles_in_text(test_text, pages_data)
    
    print(f"Found {len(title_candidates)} titles with Roman config:")
    for title in title_candidates:
        if title.confidence > 0.8:  # Show high-confidence matches
            level_icon = "🔴" if title.level == 1 else "🟡" if title.level == 2 else "🟢" if title.level == 3 else "⚪"
            print(f"  {level_icon} H{title.level}: {title.text} (conf: {title.confidence:.2f})")
    
    # Test 2: Parcours/Module/Unité configuration  
    print(f"\n🧪 Test 2: Educational Configuration")
    print("-" * 40)
    
    educational_config = TitleStructureConfig(
        h1_patterns=["Parcours 1", "Parcours 2", "Parcours 3", "Parcours 4"],
        h2_patterns=["Module 1", "Module 2", "Module 3", "Module 4"],
        h3_patterns=["Unité 1", "Unité 2", "Unité 3", "Unité 4"],
        h4_patterns=[],
        use_auto_detection=True
    )
    
    # Create educational test text
    educational_text = """
Parcours 1: Formation Data Engineering
Présentation du parcours de formation.

Module 1: Bases de données
Introduction aux concepts de base de données.

Unité 1: SQL Fundamentals
Les bases du langage SQL.

Unité 2: NoSQL Concepts
Introduction aux bases NoSQL.

Module 2: Data Processing
Traitement et transformation des données.

Parcours 2: Formation Analytics
Deuxième parcours sur l'analytique.
"""
    
    educational_pages = [{
        'page_number': 1,
        'text': educational_text,
        'char_count': len(educational_text),
        'word_count': len(educational_text.split())
    }]
    
    detector2 = TitleDetector(custom_config=educational_config)
    educational_titles = detector2.detect_titles_in_text(educational_text, educational_pages)
    
    print(f"Found {len(educational_titles)} titles with Educational config:")
    for title in educational_titles:
        if title.confidence > 0.8:
            level_icon = "🔴" if title.level == 1 else "🟡" if title.level == 2 else "🟢" if title.level == 3 else "⚪"
            print(f"  {level_icon} H{title.level}: {title.text} (conf: {title.confidence:.2f})")
    
    # Test 3: Chunk hierarchy with custom config
    print(f"\n🧪 Test 3: Chunk Hierarchy with Custom Config")
    print("-" * 40)
    
    chunks = [chunk.strip() for chunk in test_text.split('\n\n') if chunk.strip()]
    hierarchies = build_chunk_title_hierarchies(chunks, test_text, pages_data, roman_config)
    
    print(f"Built hierarchies for {len(chunks)} chunks:")
    for i, (chunk, hierarchy) in enumerate(zip(chunks[:5], hierarchies[:5])):  # Show first 5
        chunk_preview = (chunk[:50] + "...") if len(chunk) > 50 else chunk
        print(f"\nChunk {i+1}: {chunk_preview}")
        print(f"  H1: {hierarchy.h1_title or 'None'}")
        print(f"  H2: {hierarchy.h2_title or 'None'}")
        print(f"  H3: {hierarchy.h3_title or 'None'}")
        print(f"  Path: {hierarchy.get_full_path() or 'None'}")
    
    # Test 4: Validation of expected behavior
    print(f"\n✅ Validation Tests:")
    print("-" * 30)
    
    # Check if Roman numerals are properly detected
    h1_titles = [t for t in title_candidates if t.level == 1]
    h2_titles = [t for t in title_candidates if t.level == 2] 
    h3_titles = [t for t in title_candidates if t.level == 3]
    
    expected_h1 = ["I. Data Engineering Fundamentals", "II. Microsoft Fabric Overview", "III. Advanced Analytics"]
    expected_h2 = ["1. Data Storage Concepts", "2. Data Processing Patterns", "1. Fabric Components", "2. Fabric Administration"]
    expected_h3 = ["i. Relational Databases", "ii. NoSQL Databases", "i. Batch Processing", "i. Data Factory", "ii. Synapse Data Engineering"]
    
    h1_found = [t.text for t in h1_titles]
    h2_found = [t.text for t in h2_titles]
    h3_found = [t.text for t in h3_titles]
    
    print(f"Expected H1: {len(expected_h1)}, Found: {len(h1_found)}")
    print(f"Expected H2: {len(expected_h2)}, Found: {len(h2_found)}")  
    print(f"Expected H3: {len(expected_h3)}, Found: {len(h3_found)}")
    
    # Check specific matches
    roman_matches = 0
    for expected in expected_h1:
        if any(expected in found for found in h1_found):
            roman_matches += 1
    
    if roman_matches >= 2:
        print("✅ Roman numeral H1 detection working")
    else:
        print("❌ Roman numeral H1 detection needs improvement")
    
    # Check hierarchy persistence
    data_eng_chunks = [h for h in hierarchies if h.h1_title and "Data Engineering" in h.h1_title]
    if len(data_eng_chunks) > 1:
        print("✅ H1 title persistence working")
    else:
        print("❌ H1 title persistence needs improvement")
    
    print(f"\n🎉 Custom title configuration test completed!")


if __name__ == "__main__":
    test_custom_title_config()