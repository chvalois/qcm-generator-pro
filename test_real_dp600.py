#!/usr/bin/env python3
"""
Test the real DP600 PDF with the fixed title detection.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set PYTHONPATH to include the project root
os.environ['PYTHONPATH'] = str(Path(__file__).parent)

# Change to project directory
os.chdir(Path(__file__).parent)

# Now import with absolute imports
from src.services.pdf_processor import PDFProcessor
from src.services.title_detector import build_chunk_title_hierarchies

def test_real_dp600():
    """Test the real DP600 PDF with the fixed title detection."""
    
    # Path to the DP600 PDF (WSL path)
    pdf_path = "/mnt/c/Users/massi/Documents/Perso/Formation Microsoft/DP-600/Formation_DP-600T00_Complete.pdf"
    
    print(f"Testing real DP600 PDF: {pdf_path}")
    
    # Initialize PDF processor
    processor = PDFProcessor()
    
    try:
        # Extract text and get pages data
        print("Extracting text from PDF...")
        text = processor.extract_text(Path(pdf_path))
        pages_data = processor.extract_text_by_pages(Path(pdf_path))
        
        print(f"Extracted {len(text)} characters from {len(pages_data)} pages")
        
        # Create chunks (similar to how the application does it)
        print("Creating text chunks...")
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = text_splitter.split_text(text)
        print(f"Created {len(chunks)} text chunks")
        
        # Build title hierarchies for chunks (now with the fix)
        print("Building title hierarchies for chunks...")
        chunk_hierarchies = build_chunk_title_hierarchies(chunks, text, pages_data)
        
        print(f"Built hierarchies for {len(chunk_hierarchies)} chunks")
        
        # Find chunks that contain Module 3 of Parcours 3 content
        print(f"\nSearching for Module 3 of Parcours 3 chunks...")
        print("=" * 80)
        
        module3_parcours3_chunks = []
        
        for i, (chunk, hierarchy) in enumerate(zip(chunks, chunk_hierarchies)):
            # Check if this chunk has Parcours 3 and Module 3 in its hierarchy
            h1_title = hierarchy.h1_title or ""
            h2_title = hierarchy.h2_title or ""
            
            is_parcours3 = "parcours 3" in h1_title.lower()
            is_module3 = "module 3" in h2_title.lower()
            
            if is_parcours3 and is_module3:
                module3_parcours3_chunks.append({
                    'chunk_index': i,
                    'chunk': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    'hierarchy': hierarchy,
                    'h1_title': hierarchy.h1_title,
                    'h2_title': hierarchy.h2_title,
                    'h3_title': hierarchy.h3_title,
                    'h4_title': hierarchy.h4_title,
                    'full_path': hierarchy.get_full_path()
                })
        
        print(f"Found {len(module3_parcours3_chunks)} chunks correctly assigned to Parcours 3 > Module 3:")
        
        for chunk_data in module3_parcours3_chunks[:5]:  # Show first 5
            print(f"\nChunk {chunk_data['chunk_index']:4d}:")
            print(f"  H1: {chunk_data['h1_title']}")
            print(f"  H2: {chunk_data['h2_title']}")
            print(f"  H3: {chunk_data['h3_title']}")
            print(f"  Full Path: {chunk_data['full_path']}")
            print(f"  Content Preview: {chunk_data['chunk'][:100]}...")
        
        if len(module3_parcours3_chunks) > 5:
            print(f"\n... and {len(module3_parcours3_chunks) - 5} more chunks")
        
        # Success summary
        if len(module3_parcours3_chunks) > 0:
            print(f"\n✅ SUCCESS: Found {len(module3_parcours3_chunks)} chunks correctly assigned to Module 3 of Parcours 3!")
            print("The title detection bug has been fixed.")
            return True
        else:
            print(f"\n❌ ISSUE: No chunks found for Module 3 of Parcours 3")
            return False
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_dp600()
    if success:
        print(f"\n🎉 Title detection is working correctly!")
    else:
        print(f"\n🔧 Further investigation needed.")