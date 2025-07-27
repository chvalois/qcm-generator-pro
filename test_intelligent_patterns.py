#!/usr/bin/env python3
"""
Test intelligent pattern detection.
"""

import sys
import os
from pathlib import Path

# Add src to Python path and set up module structure
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

from src.models.schemas import TitleStructureConfig
from src.services.title_detector import TitleDetector


def test_intelligent_patterns():
    """Test intelligent pattern detection with examples from DP-600."""
    print("🔧 Testing Intelligent Pattern Detection")
    print("=" * 50)
    
    # Create test document with various numbered patterns
    test_text = """
Parcours 1: Bien démarrer avec Microsoft Fabric
Introduction au premier parcours.

Parcours 2: Implémenter un entrepôt de données  
Deuxième parcours sur les entrepôts.

Parcours 4: Implémenter un lakehouse avec Microsoft
Quatrième parcours (on a sauté le 3).

Module 1: Bases de données
Premier module.

Module 2: Data Processing
Deuxième module.

Module 15: Advanced Analytics
Quinzième module.

Unité 1: Introduction SQL
Première unité.

Unité 2: NoSQL Concepts
Deuxième unité.

Unité 27: Machine Learning
Vingt-septième unité.

I. Traditional Section
Section avec chiffres romains majuscules.

II. Another Section
Deuxième section romaine.

XV. Much Later Section
Quinzième section romaine.

i. subsection
Sous-section en romain minuscule.

ii. another subsection
Deuxième sous-section.

viii. later subsection
Huitième sous-section.

1. Numbered item
Item numéroté.

2. Second item
Deuxième item.

25. Much later item
Vingt-cinquième item.

a. Letter item
Item avec lettre.

b. Second letter
Deuxième lettre.

z. Last letter
Dernière lettre.
"""
    
    # Create pages data
    pages_data = [{
        'page_number': 1,
        'text': test_text,
        'char_count': len(test_text),
        'word_count': len(test_text.split())
    }]
    
    print(f"📄 Test document created with various numbered patterns")
    
    # Test intelligent configuration - define only examples, system should generalize
    print(f"\n🧪 Test: Intelligent Pattern Detection")
    print("-" * 40)
    
    intelligent_config = TitleStructureConfig(
        h1_patterns=["Parcours 1"],  # Should match Parcours 1, 2, 4, etc.
        h2_patterns=["Module 1"],    # Should match Module 1, 2, 15, etc.
        h3_patterns=["Unité 1"],     # Should match Unité 1, 2, 27, etc.
        h4_patterns=["1."],          # Should match 1., 2., 25., etc.
        use_auto_detection=False
    )
    
    detector = TitleDetector(custom_config=intelligent_config)
    title_candidates = detector.detect_titles_in_text(test_text, pages_data)
    
    print(f"Found {len(title_candidates)} titles with intelligent patterns:")
    
    # Group by level for better display
    by_level = {}
    for title in title_candidates:
        if title.level not in by_level:
            by_level[title.level] = []
        by_level[title.level].append(title)
    
    for level in sorted(by_level.keys()):
        level_icon = "🔴" if level == 1 else "🟡" if level == 2 else "🟢" if level == 3 else "⚪"
        print(f"\n{level_icon} H{level} Titles:")
        for title in by_level[level]:
            print(f"  • {title.text} (conf: {title.confidence:.2f})")
    
    # Expected matches
    expected_h1 = ["Parcours 1", "Parcours 2", "Parcours 4"]
    expected_h2 = ["Module 1", "Module 2", "Module 15"]
    expected_h3 = ["Unité 1", "Unité 2", "Unité 27"]
    expected_h4 = ["1. Numbered item", "2. Second item", "25. Much later item"]
    
    # Validation
    print(f"\n✅ Validation:")
    
    h1_found = [t.text for t in by_level.get(1, [])]
    h2_found = [t.text for t in by_level.get(2, [])]
    h3_found = [t.text for t in by_level.get(3, [])]
    h4_found = [t.text for t in by_level.get(4, [])]
    
    # Check H1 patterns (Parcours)
    h1_matches = sum(1 for expected in expected_h1 if any(expected in found for found in h1_found))
    print(f"H1 (Parcours): {h1_matches}/{len(expected_h1)} expected patterns found")
    
    # Check H2 patterns (Module)  
    h2_matches = sum(1 for expected in expected_h2 if any(expected in found for found in h2_found))
    print(f"H2 (Module): {h2_matches}/{len(expected_h2)} expected patterns found")
    
    # Check H3 patterns (Unité)
    h3_matches = sum(1 for expected in expected_h3 if any(expected in found for found in h3_found))
    print(f"H3 (Unité): {h3_matches}/{len(expected_h3)} expected patterns found")
    
    # Check H4 patterns (numbered)
    h4_matches = sum(1 for expected in expected_h4 if any(expected in found for found in h4_found))
    print(f"H4 (1.): {h4_matches}/{len(expected_h4)} expected patterns found")
    
    # Overall success
    total_expected = len(expected_h1) + len(expected_h2) + len(expected_h3) + len(expected_h4)
    total_found = h1_matches + h2_matches + h3_matches + h4_matches
    
    print(f"\nOverall: {total_found}/{total_expected} patterns correctly detected")
    
    if total_found >= total_expected * 0.8:  # 80% success rate
        print(f"🎉 INTELLIGENT PATTERN DETECTION WORKING!")
        print("System successfully generalizes from examples to detect similar patterns.")
    else:
        print(f"❌ INTELLIGENT PATTERN DETECTION NEEDS IMPROVEMENT")
        print("System may not be generalizing patterns correctly.")
    
    return total_found >= total_expected * 0.8


def test_regex_conversion():
    """Test the pattern to regex conversion function."""
    print(f"\n🔧 Testing Pattern to Regex Conversion")
    print("-" * 40)
    
    from src.services.title_detector import TitleDetector
    
    detector = TitleDetector()
    
    test_cases = [
        ("Parcours 1", ["Parcours 1: Test", "Parcours 2: Test", "Parcours 15: Test"]),
        ("Module 1", ["Module 1: Test", "Module 2: Test", "Module 27: Test"]),
        ("I.", ["I. Test", "II. Test", "XV. Test"]),
        ("1.", ["1. Test", "2. Test", "25. Test"]),
        ("a.", ["a. Test", "b. Test", "z. Test"])
    ]
    
    for pattern, test_strings in test_cases:
        print(f"\nPattern: '{pattern}'")
        regex = detector._convert_pattern_to_regex(pattern)
        print(f"Regex: '{regex}'")
        
        matches = []
        for test_string in test_strings:
            if detector._matches_custom_pattern(test_string, pattern):
                matches.append(test_string)
        
        print(f"Matches: {matches}")


if __name__ == "__main__":
    success = test_intelligent_patterns()
    test_regex_conversion()
    
    if success:
        print(f"\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n❌ TESTS NEED IMPROVEMENT!")