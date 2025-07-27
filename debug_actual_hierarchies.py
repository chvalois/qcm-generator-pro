#!/usr/bin/env python3
"""
Debug what hierarchies are actually being assigned to chunks.
"""

import os
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ['PYTHONPATH'] = str(Path(__file__).parent)
os.chdir(Path(__file__).parent)

from src.services.pdf_processor import PDFProcessor
from src.services.title_detector import build_chunk_title_hierarchies

def debug_actual_hierarchies():
    """Debug what hierarchies are actually being assigned."""
    
    pdf_path = "/mnt/c/Users/massi/Documents/Perso/Formation Microsoft/DP-600/Formation_DP-600T00_Complete.pdf"
    
    print(f"Debugging hierarchies for: {pdf_path}")
    
    processor = PDFProcessor()
    
    try:
        text = processor.extract_text(Path(pdf_path))
        pages_data = processor.extract_text_by_pages(Path(pdf_path))
        
        # Create smaller chunks for testing
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for more precise mapping
            chunk_overlap=100,
            length_function=len,
        )
        
        chunks = text_splitter.split_text(text)
        print(f"Created {len(chunks)} text chunks")
        
        # Build hierarchies
        chunk_hierarchies = build_chunk_title_hierarchies(chunks, text, pages_data)
        
        print(f"\nLooking for chunks around Module 3 of Parcours 3...")
        
        # Find chunks containing "Module 3: Utiliser Apache Spark"
        target_chunks = []
        for i, chunk in enumerate(chunks):
            if "module 3" in chunk.lower() and "apache spark" in chunk.lower():
                target_chunks.append(i)
        
        print(f"Found {len(target_chunks)} chunks containing 'Module 3' and 'Apache Spark'")
        
        for chunk_idx in target_chunks:
            hierarchy = chunk_hierarchies[chunk_idx]
            chunk = chunks[chunk_idx]
            
            print(f"\nChunk {chunk_idx}:")
            print(f"  Content: {chunk[:150]}...")
            print(f"  H1: {hierarchy.h1_title}")
            print(f"  H2: {hierarchy.h2_title}")
            print(f"  H3: {hierarchy.h3_title}")
            print(f"  Full Path: {hierarchy.get_full_path()}")
        
        # Also check chunks around these indices (context)
        print(f"\nChecking context chunks around target chunks...")
        
        for chunk_idx in target_chunks:
            start_idx = max(0, chunk_idx - 3)
            end_idx = min(len(chunks), chunk_idx + 4)
            
            print(f"\nContext around chunk {chunk_idx} (range {start_idx}-{end_idx}):")
            
            for i in range(start_idx, end_idx):
                hierarchy = chunk_hierarchies[i]
                chunk = chunks[i]
                marker = " >>> TARGET <<<" if i == chunk_idx else ""
                
                print(f"  Chunk {i:3d}{marker}: {hierarchy.get_full_path()}")
                print(f"    Content: {chunk[:80]}...")
        
        # Check specifically for Parcours 3 chunks
        print(f"\nLooking for ALL Parcours 3 chunks...")
        parcours3_chunks = []
        
        for i, hierarchy in enumerate(chunk_hierarchies):
            if hierarchy.h1_title and "parcours 3" in hierarchy.h1_title.lower():
                parcours3_chunks.append(i)
        
        print(f"Found {len(parcours3_chunks)} chunks assigned to Parcours 3")
        
        if len(parcours3_chunks) > 0:
            print("First 10 Parcours 3 chunks:")
            for i in parcours3_chunks[:10]:
                hierarchy = chunk_hierarchies[i]
                print(f"  Chunk {i:3d}: {hierarchy.get_full_path()}")
        
        # Check if there are any Module 3 chunks at all
        print(f"\nLooking for ALL Module 3 chunks (any Parcours)...")
        module3_chunks = []
        
        for i, hierarchy in enumerate(chunk_hierarchies):
            if hierarchy.h2_title and "module 3" in hierarchy.h2_title.lower():
                module3_chunks.append(i)
        
        print(f"Found {len(module3_chunks)} chunks assigned to any Module 3")
        
        if len(module3_chunks) > 0:
            print("All Module 3 chunks:")
            for i in module3_chunks:
                hierarchy = chunk_hierarchies[i]
                print(f"  Chunk {i:3d}: {hierarchy.get_full_path()}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_actual_hierarchies()