#!/usr/bin/env python3
"""
Test script for filename uniqueness system.

This script tests the unique filename generation to ensure that:
1. Original filenames are preserved when possible
2. Numeric suffixes are added for duplicates
3. Filenames are sanitized for filesystem safety
"""

import tempfile
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ui.streamlit_app import StreamlitQCMInterface


def test_unique_filename_generation():
    """Test the unique filename generation logic."""
    print("🧪 Testing unique filename generation...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        interface = StreamlitQCMInterface()
        
        # Test 1: Original filename when no conflict
        print("\n📝 Test 1: Original filename preservation")
        filename1 = interface._get_unique_filename(temp_path, "document.pdf")
        print(f"   Original: document.pdf")
        print(f"   Result: {filename1.name}")
        assert filename1.name == "document.pdf"
        
        # Create the file to simulate it exists
        filename1.touch()
        
        # Test 2: Numeric suffix for duplicate
        print("\n📝 Test 2: Duplicate handling with numeric suffix")
        filename2 = interface._get_unique_filename(temp_path, "document.pdf")
        print(f"   Original: document.pdf (exists)")
        print(f"   Result: {filename2.name}")
        assert filename2.name == "document_1.pdf"
        
        # Create the second file
        filename2.touch()
        
        # Test 3: Multiple duplicates
        print("\n📝 Test 3: Multiple duplicates")
        filename3 = interface._get_unique_filename(temp_path, "document.pdf")
        print(f"   Original: document.pdf (exists, _1 exists)")
        print(f"   Result: {filename3.name}")
        assert filename3.name == "document_2.pdf"
        
        # Test 4: Special characters sanitization
        print("\n📝 Test 4: Special characters sanitization")
        filename4 = interface._get_unique_filename(temp_path, "my-file@2024!.pdf")
        print(f"   Original: my-file@2024!.pdf")
        print(f"   Result: {filename4.name}")
        expected = "my-file2024.pdf"  # Special chars removed
        assert filename4.name == expected
        
        # Test 5: No PDF extension
        print("\n📝 Test 5: Missing PDF extension")
        filename5 = interface._get_unique_filename(temp_path, "report")
        print(f"   Original: report")
        print(f"   Result: {filename5.name}")
        assert filename5.name == "report.pdf"
        
        # Test 6: French characters
        print("\n📝 Test 6: French characters")
        filename6 = interface._get_unique_filename(temp_path, "présentationété.pdf")
        print(f"   Original: présentationété.pdf")
        print(f"   Result: {filename6.name}")
        # My filter allows alphanumeric chars, so accented chars are kept
        assert filename6.name == "présentationété.pdf"
        
        print("\n✅ All tests passed!")


def test_real_upload_scenario():
    """Test a more realistic upload scenario."""
    print("\n🚀 Testing realistic upload scenario...")
    
    from src.core.config import settings
    
    # Create test upload directory
    upload_dir = settings.data_dir / "pdfs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    interface = StreamlitQCMInterface()
    
    test_files = [
        "Cours Python.pdf",
        "Cours Python.pdf",  # Same name - should get _1
        "Cours Python.pdf",  # Same name - should get _2
        "cours-python-2024.pdf",  # Different name
        "Guide d'utilisation.pdf",  # French with apostrophe
    ]
    
    generated_files = []
    
    print(f"Upload directory: {upload_dir}")
    
    for i, original_name in enumerate(test_files):
        unique_path = interface._get_unique_filename(upload_dir, original_name)
        print(f"   {original_name} → {unique_path.name}")
        
        # Simulate file creation
        unique_path.touch()
        generated_files.append(unique_path)
    
    print(f"\n📂 Generated files in {upload_dir}:")
    for file_path in upload_dir.glob("*.pdf"):
        print(f"   - {file_path.name}")
    
    # Cleanup generated test files
    for file_path in generated_files:
        try:
            file_path.unlink()
            print(f"   ✅ Cleaned up: {file_path.name}")
        except Exception as e:
            print(f"   ⚠️ Failed to cleanup {file_path.name}: {e}")


if __name__ == "__main__":
    print("🎯 QCM Generator Pro - Filename Uniqueness Test")
    print("=" * 50)
    
    try:
        test_unique_filename_generation()
        test_real_upload_scenario()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📋 Summary:")
        print("   ✅ Original filenames are preserved when possible")
        print("   ✅ Numeric suffixes are added for duplicates")
        print("   ✅ Special characters are sanitized")
        print("   ✅ PDF extension is ensured")
        print("   ✅ Multiple duplicates are handled correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)