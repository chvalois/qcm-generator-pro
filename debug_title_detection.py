#!/usr/bin/env python3
"""
Debug script to test title detection with the DP600 PDF document.
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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_title_detection():
    """Test title detection with the DP600 PDF."""
    
    # Path to the DP600 PDF (WSL path)
    pdf_path = "/mnt/c/Users/massi/Documents/Perso/Formation Microsoft/DP-600/Formation_DP-600T00_Complete.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return
    
    print(f"Testing title detection with: {pdf_path}")
    
    # Initialize PDF processor
    processor = PDFProcessor()
    
    try:
        # Extract text and get pages data
        print("Extracting text from PDF...")
        text = processor.extract_text(Path(pdf_path))
        metadata = processor.extract_metadata(Path(pdf_path))
        pages_data = processor.extract_text_by_pages(Path(pdf_path))
        
        print(f"Extracted {len(text)} characters from {len(pages_data)} pages")
        
        # Initialize title detector
        detector = TitleDetector()
        
        # Detect titles
        print("Detecting titles...")
        title_candidates = detector.detect_titles_in_text(text, pages_data)
        
        print(f"Found {len(title_candidates)} title candidates")
        
        # Filter and display titles related to "Parcours" and "Module"
        relevant_titles = []
        for title in title_candidates:
            title_lower = title.text.lower()
            if any(keyword in title_lower for keyword in ['parcours', 'module', 'unité']):
                relevant_titles.append(title)
        
        print(f"\nFound {len(relevant_titles)} relevant educational titles:")
        print("=" * 80)
        
        for title in sorted(relevant_titles, key=lambda x: (x.page_number, x.line_number)):
            print(f"Page {title.page_number:3d} | L{title.level} | Conf: {title.confidence:.3f} | {title.text}")
            
            # Check for specific patterns
            if "module 3" in title.text.lower() and "parcours 3" in title.text.lower():
                print("    *** FOUND: Module 3 of Parcours 3 ***")
                print(f"    Features: {title.features}")
        
        # Look specifically for "Module 3" patterns
        print(f"\nSearching specifically for 'Module 3' patterns:")
        print("=" * 80)
        
        module3_titles = []
        for title in title_candidates:
            if "module 3" in title.text.lower():
                module3_titles.append(title)
                print(f"Page {title.page_number:3d} | L{title.level} | Conf: {title.confidence:.3f} | {title.text}")
                print(f"    Features: {title.features}")
        
        if not module3_titles:
            print("No 'Module 3' titles found in candidates!")
            
            # Search for "module 3" in raw text
            print("\nSearching for 'module 3' in raw text...")
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if "module 3" in line.lower():
                    print(f"Line {i:4d}: {line.strip()}")
                    
                    # Find which page this line is on
                    char_pos = sum(len(lines[j]) + 1 for j in range(i))
                    current_page = 1
                    cumulative_chars = 0
                    
                    for page_data in pages_data:
                        page_text = page_data.get('text', '')
                        if cumulative_chars + len(page_text) >= char_pos:
                            current_page = page_data['page_number']
                            break
                        cumulative_chars += len(page_text)
                    
                    print(f"    (Estimated page: {current_page})")
        
        # Test specific patterns that might be in DP600
        print(f"\nTesting specific DP600 patterns:")
        print("=" * 80)
        
        test_lines = [
            "Parcours 3 - Module 3",
            "Module 3 : Administration",
            "Module 3 - Configuration",
            "PARCOURS 3 MODULE 3",
            "3. Module 3",
            "Parcours 3 : Module 3",
        ]
        
        for test_line in test_lines:
            # Create a mock title candidate
            candidate = detector._analyze_title_candidate(
                test_line, 0, 1, [test_line], test_line
            )
            
            if candidate:
                print(f"Pattern: '{test_line}' -> Level {candidate.level}, Confidence: {candidate.confidence:.3f}")
                print(f"    Features: {candidate.features}")
            else:
                print(f"Pattern: '{test_line}' -> NOT DETECTED")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_title_detection()