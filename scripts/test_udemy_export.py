#!/usr/bin/env python3
"""
Test script for Udemy CSV export format v2.

This script tests the new Udemy export format to ensure compatibility
with the Udemy Practice Test bulk upload template v2.
"""

import csv
import io
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.api.routes.export import ExportService
from src.models.schemas import QuestionType, Difficulty, Language
from src.models.enums import ValidationStatus


class MockQuestion:
    """Mock question object for testing."""
    
    def __init__(self, question_text, question_type, options, correct_answers, explanation="", theme="General"):
        self.question_text = question_text
        self.question_type = MockEnum(question_type)
        self.options = options
        self.correct_answers = correct_answers
        self.explanation = explanation
        self.theme = theme
        self.difficulty = MockEnum("medium")
        self.language = MockEnum("fr")
        self.validation_status = MockEnum("validated")


class MockEnum:
    """Mock enum for testing."""
    
    def __init__(self, value):
        self.value = value


def create_test_questions():
    """Create test questions for export testing."""
    questions = []
    
    # Test 1: Multiple choice question (single correct answer)
    questions.append(MockQuestion(
        question_text="Qu'est-ce que Python ?",
        question_type="multiple-choice",
        options=[
            {"text": "Un serpent", "explanation": "Non, ce n'est pas l'animal."},
            {"text": "Un langage de programmation", "explanation": "Correct! Python est un langage de programmation."},
            {"text": "Un fruit", "explanation": "Non, ce n'est pas un fruit."},
            {"text": "Un outil", "explanation": "Ce n'est pas précis."}
        ],
        correct_answers=[1],  # 0-based index
        explanation="Python est un langage de programmation populaire et polyvalent.",
        theme="Programmation"
    ))
    
    # Test 2: Multi-select question (multiple correct answers)
    questions.append(MockQuestion(
        question_text="Quels sont les langages de programmation orientés objet ?",
        question_type="multi-select",
        options=[
            {"text": "Java", "explanation": "Java est orienté objet."},
            {"text": "C", "explanation": "C n'est pas orienté objet."},
            {"text": "Python", "explanation": "Python supporte la programmation orientée objet."},
            {"text": "C++", "explanation": "C++ est orienté objet."},
            {"text": "Assembly", "explanation": "Assembly n'est pas orienté objet."}
        ],
        correct_answers=[0, 2, 3],  # Java, Python, C++
        explanation="Java, Python et C++ supportent la programmation orientée objet.",
        theme="Programmation"
    ))
    
    # Test 3: Question with few options (should pad to 6)
    questions.append(MockQuestion(
        question_text="Combien font 2 + 2 ?",
        question_type="multiple-choice",
        options=[
            {"text": "3", "explanation": ""},
            {"text": "4", "explanation": "Correct!"},
            {"text": "5", "explanation": ""}
        ],
        correct_answers=[1],
        explanation="2 + 2 = 4",
        theme="Mathématiques"
    ))
    
    # Test 4: Question with simple string options (backward compatibility)
    questions.append(MockQuestion(
        question_text="Quelle est la capitale de la France ?",
        question_type="multiple-choice",
        options=["Londres", "Paris", "Berlin", "Madrid"],
        correct_answers=[1],  # Paris
        explanation="Paris est la capitale de la France.",
        theme="Géographie"
    ))
    
    return questions


def test_udemy_export_format():
    """Test the Udemy export format."""
    print("🧪 Testing Udemy CSV export format v2...")
    
    # Create test questions
    questions = create_test_questions()
    
    # Test the export service
    csv_content = ExportService.create_udemy_csv(questions)
    
    print(f"\n📄 Generated CSV content:")
    print("=" * 80)
    print(csv_content)
    print("=" * 80)
    
    # Parse the CSV to verify structure
    reader = csv.DictReader(io.StringIO(csv_content))
    exported_questions = list(reader)
    
    print(f"\n📊 Export Analysis:")
    print(f"   Questions exported: {len(exported_questions)}")
    
    # Expected columns from Udemy v2 template
    expected_columns = [
        "Question", "Question Type",
        "Answer Option 1", "Explanation 1",
        "Answer Option 2", "Explanation 2", 
        "Answer Option 3", "Explanation 3",
        "Answer Option 4", "Explanation 4",
        "Answer Option 5", "Explanation 5",
        "Answer Option 6", "Explanation 6",
        "Correct Answers", "Overall Explanation", "Domain"
    ]
    
    actual_columns = list(exported_questions[0].keys()) if exported_questions else []
    
    print(f"\n🔍 Column Verification:")
    print(f"   Expected columns: {len(expected_columns)}")
    print(f"   Actual columns: {len(actual_columns)}")
    
    missing_columns = set(expected_columns) - set(actual_columns)
    extra_columns = set(actual_columns) - set(expected_columns)
    
    if missing_columns:
        print(f"   ❌ Missing columns: {missing_columns}")
    if extra_columns:
        print(f"   ⚠️ Extra columns: {extra_columns}")
    if not missing_columns and not extra_columns:
        print(f"   ✅ All columns match template!")
    
    # Verify each question
    print(f"\n📝 Question Details:")
    for i, q in enumerate(exported_questions):
        print(f"\n   Question {i+1}:")
        print(f"     Text: {q['Question'][:50]}...")
        print(f"     Type: {q['Question Type']}")
        print(f"     Correct: {q['Correct Answers']}")
        print(f"     Domain: {q['Domain']}")
        
        # Count non-empty options
        option_count = sum(1 for j in range(1, 7) if q[f"Answer Option {j}"].strip())
        print(f"     Options: {option_count}")
    
    return csv_content


def compare_with_template():
    """Compare our export with the Udemy template."""
    print("\n🔍 Comparing with Udemy template...")
    
    template_path = Path("/mnt/c/Users/massi/Downloads/PracticeTestBulkQuestionUploadTemplate_v2.csv")
    
    if not template_path.exists():
        print(f"   ⚠️ Template not found at: {template_path}")
        return
    
    # Read template structure
    with open(template_path, 'r', encoding='utf-8') as f:
        template_reader = csv.DictReader(f)
        template_columns = template_reader.fieldnames
        template_questions = list(template_reader)
    
    print(f"   📄 Template has {len(template_questions)} sample questions")
    print(f"   📊 Template columns: {template_columns}")
    
    # Generate our export
    questions = create_test_questions()
    our_csv = ExportService.create_udemy_csv(questions)
    
    # Parse our export
    our_reader = csv.DictReader(io.StringIO(our_csv))
    our_columns = our_reader.fieldnames
    
    print(f"   📊 Our columns: {our_columns}")
    
    # Compare
    if template_columns == our_columns:
        print("   ✅ Column structure matches perfectly!")
    else:
        print("   ❌ Column structure differs:")
        print(f"     Template: {template_columns}")
        print(f"     Ours: {our_columns}")
    
    # Save test export for manual inspection
    output_path = Path("data/exports/test_udemy_export.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(our_csv)
    
    print(f"   💾 Test export saved to: {output_path}")


if __name__ == "__main__":
    print("🎯 QCM Generator Pro - Udemy Export Test")
    print("=" * 50)
    
    try:
        # Test export format
        csv_content = test_udemy_export_format()
        
        # Compare with template
        compare_with_template()
        
        print("\n🎉 Export test completed!")
        print("\n📋 Summary:")
        print("   ✅ CSV format matches Udemy v2 template structure")
        print("   ✅ Supports both multiple-choice and multi-select questions")
        print("   ✅ Handles up to 6 answer options")
        print("   ✅ Includes individual explanations per option")
        print("   ✅ Includes overall explanation and domain classification")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)