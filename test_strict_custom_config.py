#!/usr/bin/env python3
"""
Test strict custom title configuration - no auto detection.
"""

import sys
import os
from pathlib import Path

# Add src to Python path and set up module structure
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

from src.models.schemas import TitleStructureConfig
from src.services.title_detector import TitleDetector


def test_strict_custom_config():
    """Test that only custom patterns are detected when auto_detection=False."""
    print("🔧 Testing Strict Custom Configuration (No Auto Detection)")
    print("=" * 60)
    
    # Create test document with mixed content
    test_text = """
Introduction to Microsoft DP-600
This course covers the complete DP-600 curriculum.

Type: Introduction
This is a type section that should NOT be detected as H1.

Parcours 1: Bien démarrer avec Microsoft Fabric
This should be detected as H1.

Espaces de travail
This should NOT be detected as anything.

Module 1: Bases de données
This should be detected as H2.

Explorer les équipes de données et Microsoft Fabric
This should NOT be detected as anything.

Unité 1: Introduction SQL
This should be detected as H3.

Rôles et défis traditionnels
This should NOT be detected as anything.

I. Traditional Section
This should NOT be detected because it's not in our patterns.

1. Numbered Section  
This should NOT be detected because it's not in our patterns.
"""
    
    # Create pages data
    pages_data = [{
        'page_number': 1,
        'text': test_text,
        'char_count': len(test_text),
        'word_count': len(test_text.split())
    }]
    
    print(f"📄 Test document created")
    
    # Test strict configuration - only our defined patterns
    print(f"\n🧪 Test: Strict Custom Configuration (use_auto_detection=False)")
    print("-" * 50)
    
    strict_config = TitleStructureConfig(
        h1_patterns=["Parcours 1", "Parcours 2", "Parcours 3"],
        h2_patterns=["Module 1", "Module 2", "Module 3"],
        h3_patterns=["Unité 1", "Unité 2", "Unité 3"],
        h4_patterns=[],
        use_auto_detection=False  # STRICT MODE
    )
    
    detector = TitleDetector(custom_config=strict_config)
    title_candidates = detector.detect_titles_in_text(test_text, pages_data)
    
    print(f"Found {len(title_candidates)} titles with strict config:")
    for title in title_candidates:
        level_icon = "🔴" if title.level == 1 else "🟡" if title.level == 2 else "🟢" if title.level == 3 else "⚪"
        print(f"  {level_icon} H{title.level}: {title.text} (conf: {title.confidence:.2f})")
    
    # Validation
    expected_titles = [
        ("Parcours 1: Bien démarrer avec Microsoft Fabric", 1),
        ("Module 1: Bases de données", 2), 
        ("Unité 1: Introduction SQL", 3)
    ]
    
    found_titles = [(t.text, t.level) for t in title_candidates]
    
    print(f"\n✅ Validation:")
    print(f"Expected: {len(expected_titles)} titles")
    print(f"Found: {len(found_titles)} titles")
    
    # Check each expected title
    all_found = True
    for expected_text, expected_level in expected_titles:
        found = any(expected_text in found_text and found_level == expected_level 
                   for found_text, found_level in found_titles)
        if found:
            print(f"  ✅ Found: {expected_text} (H{expected_level})")
        else:
            print(f"  ❌ Missing: {expected_text} (H{expected_level})")
            all_found = False
    
    # Check for unwanted detections
    unwanted_patterns = ["Type:", "Espaces de travail", "Explorer les", "Rôles et défis", "I. Traditional", "1. Numbered"]
    unwanted_found = []
    for title_text, _ in found_titles:
        for unwanted in unwanted_patterns:
            if unwanted in title_text:
                unwanted_found.append(title_text)
                break
    
    if unwanted_found:
        print(f"\n❌ Unwanted titles detected:")
        for unwanted in unwanted_found:
            print(f"  - {unwanted}")
        all_found = False
    else:
        print(f"\n✅ No unwanted titles detected")
    
    if all_found and len(found_titles) == len(expected_titles):
        print(f"\n🎉 STRICT MODE TEST PASSED!")
        print("Only custom patterns were detected, auto-detection was properly disabled.")
    else:
        print(f"\n❌ STRICT MODE TEST FAILED!")
        print("Auto-detection may still be active or pattern matching is incorrect.")
    
    return all_found and len(found_titles) == len(expected_titles)


def test_flexible_config():
    """Test with auto_detection enabled for comparison."""
    print(f"\n🧪 Test: Flexible Configuration (use_auto_detection=True)")
    print("-" * 50)
    
    test_text = """
Parcours 1: Bien démarrer avec Microsoft Fabric
I. Traditional Section
1. Numbered Section
"""
    
    pages_data = [{
        'page_number': 1,
        'text': test_text,
        'char_count': len(test_text),
        'word_count': len(test_text.split())
    }]
    
    flexible_config = TitleStructureConfig(
        h1_patterns=["Parcours 1"],
        h2_patterns=[],
        h3_patterns=[],
        h4_patterns=[],
        use_auto_detection=True  # FLEXIBLE MODE
    )
    
    detector = TitleDetector(custom_config=flexible_config)
    title_candidates = detector.detect_titles_in_text(test_text, pages_data)
    
    print(f"Found {len(title_candidates)} titles with flexible config:")
    for title in title_candidates:
        level_icon = "🔴" if title.level == 1 else "🟡" if title.level == 2 else "🟢" if title.level == 3 else "⚪"
        print(f"  {level_icon} H{title.level}: {title.text} (conf: {title.confidence:.2f})")
    
    print("✅ Flexible mode should detect both custom patterns AND auto-detected patterns")


if __name__ == "__main__":
    success = test_strict_custom_config()
    test_flexible_config()
    
    if success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("Custom configuration system is working correctly.")
    else:
        print(f"\n❌ TESTS FAILED!")
        print("Custom configuration needs debugging.")