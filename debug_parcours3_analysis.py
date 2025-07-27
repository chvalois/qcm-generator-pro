#!/usr/bin/env python3
"""
Debug script to analyze Parcours 3 content in the DP600 PDF.
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
from src.services.title_detector import TitleDetector, TitleCandidate

def analyze_parcours3():
    """Analyze Parcours 3 content specifically."""
    
    # Path to the DP600 PDF (WSL path)
    pdf_path = "/mnt/c/Users/massi/Documents/Perso/Formation Microsoft/DP-600/Formation_DP-600T00_Complete.pdf"
    
    print(f"Analyzing Parcours 3 content in: {pdf_path}")
    
    # Initialize PDF processor
    processor = PDFProcessor()
    
    try:
        # Extract text and get pages data
        print("Extracting text from PDF...")
        text = processor.extract_text(Path(pdf_path))
        pages_data = processor.extract_text_by_pages(Path(pdf_path))
        
        # Find Parcours 3 start
        parcours3_start_page = None
        parcours4_start_page = None
        
        for page_data in pages_data:
            page_text = page_data.get('text', '').lower()
            page_num = page_data['page_number']
            
            if 'parcours 3' in page_text and parcours3_start_page is None:
                parcours3_start_page = page_num
                print(f"Found Parcours 3 starting at page {page_num}")
            
            if 'parcours 4' in page_text and parcours4_start_page is None:
                parcours4_start_page = page_num
                print(f"Found Parcours 4 starting at page {page_num}")
                break
        
        if parcours3_start_page is None:
            print("Could not find Parcours 3 in the document!")
            return
        
        # Extract pages from Parcours 3
        end_page = parcours4_start_page if parcours4_start_page else len(pages_data)
        
        print(f"\nAnalyzing Parcours 3 content from page {parcours3_start_page} to {end_page-1}")
        print("=" * 80)
        
        # Look for module patterns in Parcours 3 section
        module_patterns_found = []
        
        for page_data in pages_data[parcours3_start_page-1:end_page-1]:
            page_num = page_data['page_number']
            page_text = page_data.get('text', '')
            lines = page_text.split('\n')
            
            for line_num, line in enumerate(lines):
                line_stripped = line.strip()
                if 'module' in line_stripped.lower():
                    module_patterns_found.append({
                        'page': page_num,
                        'line': line_num,
                        'text': line_stripped
                    })
        
        print(f"Found {len(module_patterns_found)} lines containing 'module' in Parcours 3:")
        for pattern in module_patterns_found:
            print(f"Page {pattern['page']:3d} | Line {pattern['line']:3d} | {pattern['text']}")
        
        # Specifically look for "Module 3" in Parcours 3
        module3_patterns = []
        for pattern in module_patterns_found:
            if 'module 3' in pattern['text'].lower():
                module3_patterns.append(pattern)
        
        print(f"\nFound {len(module3_patterns)} 'Module 3' patterns in Parcours 3:")
        for pattern in module3_patterns:
            print(f"Page {pattern['page']:3d} | Line {pattern['line']:3d} | {pattern['text']}")
        
        # Let's also check what the actual structure looks like around page 109
        print(f"\nDetailed content around Parcours 3 start (pages {parcours3_start_page}-{min(parcours3_start_page+5, end_page)}):")
        print("=" * 80)
        
        for page_data in pages_data[parcours3_start_page-1:min(parcours3_start_page+5, end_page)]:
            page_num = page_data['page_number']
            page_text = page_data.get('text', '')
            lines = page_text.split('\n')
            
            print(f"\n--- Page {page_num} ---")
            # Show first 20 non-empty lines
            line_count = 0
            for line in lines:
                if line.strip() and line_count < 20:
                    print(f"{line_count+1:2d}: {line}")
                    line_count += 1
                elif line_count >= 20:
                    print("... (truncated)")
                    break
        
        # Run title detection specifically on Parcours 3 section
        print(f"\nRunning title detection on Parcours 3 section:")
        print("=" * 80)
        
        # Create a subset of pages_data for just Parcours 3
        parcours3_pages = pages_data[parcours3_start_page-1:end_page-1]
        
        # Extract text for just this section
        parcours3_text = ""
        for page_data in parcours3_pages:
            parcours3_text += page_data.get('text', '') + "\n"
        
        # Initialize title detector
        detector = TitleDetector()
        
        # Detect titles in Parcours 3 section only
        title_candidates = detector.detect_titles_in_text(parcours3_text, parcours3_pages)
        
        print(f"Found {len(title_candidates)} title candidates in Parcours 3 section:")
        
        for title in sorted(title_candidates, key=lambda x: (x.page_number, x.line_number)):
            print(f"Page {title.page_number:3d} | L{title.level} | Conf: {title.confidence:.3f} | {title.text}")
            if 'module' in title.text.lower():
                print(f"    *** MODULE DETECTED: {title.text} ***")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_parcours3()