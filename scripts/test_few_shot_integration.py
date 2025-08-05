#!/usr/bin/env python3
"""
Simple test script for Few-Shot Examples integration.

This script tests the new few-shot functionality by:
1. Loading examples from JSON
2. Building prompts with examples
3. Displaying the results
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.services.simple_examples_loader import get_examples_loader
from src.services.question_prompt_builder import get_question_prompt_builder  
from src.models.enums import QuestionType, Difficulty, Language
from src.models.schemas import QuestionContext, GenerationConfig


def test_examples_loader():
    """Test the examples loader service."""
    print("=== Testing Examples Loader ===")
    
    loader = get_examples_loader()
    
    # Test listing available projects
    available = loader.list_available_projects()
    print(f"Available example files: {available}")
    
    # Test loading examples
    examples = loader.get_examples_for_context(
        project_file="python_advanced.json",
        max_examples=2
    )
    
    print(f"Loaded {len(examples)} examples:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example.get('theme', 'N/A')} - {example.get('difficulty', 'N/A')}")
        print(f"     Question: {example.get('question', '')[:100]}...")
    
    return len(examples) > 0


def test_prompt_builder_with_examples():
    """Test the prompt builder with examples."""
    print("\n=== Testing Prompt Builder with Examples ===")
    
    builder = get_question_prompt_builder()
    
    # Create mock context and config
    context = QuestionContext(
        context_text="Python est un langage de programmation orienté objet qui supporte les context managers...",
        topic="Programmation Python",
        themes=["POO", "Context Managers"],
        confidence_score=0.85,
        source_documents=["test.pdf"]
    )
    
    config = GenerationConfig(
        num_questions=1,
        language=Language.FR,
        temperature=0.7,
        max_tokens=1000
    )
    
    # Test with examples
    prompt_with_examples = builder.build_generation_prompt_with_examples(
        context=context,
        config=config,
        question_type=QuestionType.UNIQUE_CHOICE,
        difficulty=Difficulty.MEDIUM,
        language=Language.FR,
        examples_file="python_advanced.json",
        max_examples=2
    )
    
    print("Generated prompt with examples:")
    print("-" * 80)
    print(prompt_with_examples[:1000] + "..." if len(prompt_with_examples) > 1000 else prompt_with_examples)
    print("-" * 80)
    
    # Test without examples (fallback)
    prompt_without_examples = builder.build_generation_prompt_with_examples(
        context=context,
        config=config,
        question_type=QuestionType.UNIQUE_CHOICE,
        difficulty=Difficulty.MEDIUM,
        language=Language.FR,
        examples_file=None
    )
    
    return "EXEMPLES DE QUESTIONS" in prompt_with_examples and len(prompt_with_examples) > len(prompt_without_examples)


def main():
    """Run all tests."""
    print("🧪 Testing Few-Shot Examples Integration")
    print("=" * 50)
    
    success = True
    
    try:
        # Test 1: Examples loader
        if test_examples_loader():
            print("✅ Examples loader: PASSED")
        else:
            print("❌ Examples loader: FAILED")
            success = False
            
        # Test 2: Prompt builder integration
        if test_prompt_builder_with_examples():
            print("✅ Prompt builder integration: PASSED")
        else:
            print("❌ Prompt builder integration: FAILED")
            success = False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests PASSED! Few-Shot integration is working.")
    else:
        print("💥 Some tests FAILED. Check the output above.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())