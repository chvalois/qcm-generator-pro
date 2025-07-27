#!/usr/bin/env python3
"""
Debug title positions and sorting to understand the interference.
"""

import os
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ['PYTHONPATH'] = str(Path(__file__).parent)
os.chdir(Path(__file__).parent)

from src.services.pdf_processor import PDFProcessor
from src.services.title_detector import TitleDetector

def debug_title_positions():
    """Debug title positions and sorting."""
    
    pdf_path = "/mnt/c/Users/massi/Documents/Perso/Formation Microsoft/DP-600/Formation_DP-600T00_Complete.pdf"
    
    print(f"Debugging title positions for: {pdf_path}")
    
    processor = PDFProcessor()
    
    try:
        text = processor.extract_text(Path(pdf_path))
        pages_data = processor.extract_text_by_pages(Path(pdf_path))
        
        print(f"Extracted {len(text)} characters from {len(pages_data)} pages")
        
        # Initialize title detector
        detector = TitleDetector()
        
        # Detect all titles
        title_candidates = detector.detect_titles_in_text(text, pages_data)
        
        print(f"Found {len(title_candidates)} title candidates")
        
        # Focus on Parcours 3 area (around page 120)
        parcours3_titles = []
        for title in title_candidates:
            if hasattr(title, 'page_number') and title.page_number:
                # Look for titles around page 120 (Parcours 3, Module 3 area)
                if 115 <= title.page_number <= 125:
                    parcours3_titles.append(title)
        
        # Also show ALL titles to understand the full picture
        print(f"\nALL title candidates (first 20):")
        print("=" * 80)
        for i, title in enumerate(title_candidates[:20]):
            page_num = getattr(title, 'page_number', 'unknown')
            print(f"{i:2d}. Page {page_num:>3} | L{title.level} | Conf: {title.confidence:.3f} | {title.text[:60]}...")
        
        print(f"\nTitles found around page 120 (Parcours 3 area):")
        print("=" * 80)
        
        for title in sorted(parcours3_titles, key=lambda x: (x.page_number, x.line_number)):
            print(f"Page {title.page_number:3d} | Line {title.line_number:3d} | L{title.level} | Conf: {title.confidence:.3f} | {title.text}")
        
        # Now let's see how these titles get positioned in the algorithm
        print(f"\nTitle positioning analysis:")
        print("=" * 80)
        
        # Build cumulative character positions for pages (same as in the algorithm)
        page_char_positions = {}
        cumulative_chars = 0
        
        for page_data in pages_data:
            page_num = page_data['page_number']
            page_text = page_data.get('text', '')
            page_char_positions[page_num] = cumulative_chars
            cumulative_chars += len(page_text) + 20  # Add buffer for page separators
        
        # Simulate the title positioning logic for these specific titles
        title_positions = []
        
        for title in parcours3_titles:
            title_start = -1
            
            # Use page number information for positioning (same as algorithm)
            if hasattr(title, 'page_number') and title.page_number:
                page_start_pos = page_char_positions.get(title.page_number, 0)
                
                # Search for title within the specific page text
                page_data = next((p for p in pages_data if p['page_number'] == title.page_number), None)
                if page_data:
                    page_text = page_data.get('text', '')
                    local_pos = page_text.find(title.text.strip())
                    
                    if local_pos != -1:
                        title_start = page_start_pos + local_pos
                        print(f"✅ Page {title.page_number} | '{title.text[:50]}...' -> Position {title_start} (local: {local_pos})")
                    else:
                        # Fallback with first few words
                        title_words = title.text.strip().split()
                        if title_words:
                            first_words = ' '.join(title_words[:3])
                            local_pos = page_text.find(first_words)
                            if local_pos != -1:
                                title_start = page_start_pos + local_pos
                                print(f"🔶 Page {title.page_number} | '{title.text[:50]}...' -> Position {title_start} (partial match: '{first_words}')")
                            else:
                                print(f"❌ Page {title.page_number} | '{title.text[:50]}...' -> NOT FOUND in page text")
            
            if title_start == -1:
                print(f"💥 Page {title.page_number} | '{title.text[:50]}...' -> FAILED TO POSITION")
            
            title_positions.append((title_start, title))
        
        # Sort by position (same as algorithm)
        title_positions.sort(key=lambda x: x[0] if x[0] != -1 else float('inf'))
        
        print(f"\nSorted title positions:")
        print("=" * 80)
        
        for pos, title in title_positions:
            if pos != -1:
                print(f"Position {pos:8d} | Page {title.page_number:3d} | L{title.level} | {title.text[:60]}...")
            else:
                print(f"Position {'UNKNOWN':>8s} | Page {title.page_number:3d} | L{title.level} | {title.text[:60]}...")
        
        # Now let's look specifically at what happens around chunk 844 position
        print(f"\nChunk 844 analysis:")
        print("=" * 80)
        
        # Find the approximate position of chunk 844
        # We know chunk 844 starts around page 120
        chunk_844_estimated_pos = page_char_positions.get(120, 0)
        
        print(f"Estimated chunk 844 position: {chunk_844_estimated_pos}")
        
        # Find titles that come before this position
        applicable_titles = []
        for pos, title in title_positions:
            if pos != -1 and pos <= chunk_844_estimated_pos + 1000:  # Small buffer
                applicable_titles.append((pos, title))
        
        print(f"Titles applicable to chunk 844 (before position {chunk_844_estimated_pos + 1000}):")
        
        current_titles = {1: None, 2: None, 3: None, 4: None}
        
        for pos, title in applicable_titles:
            level = title.level
            current_titles[level] = title.text
            
            # Clear lower level titles when a higher or equal level title is found
            for lower_level in range(level + 1, 5):
                current_titles[lower_level] = None
            
            print(f"  After title at {pos:8d} (L{level}): '{title.text[:40]}...'")
            print(f"    Current hierarchy: H1='{current_titles[1] or 'None'}' | H2='{current_titles[2] or 'None'}' | H3='{current_titles[3] or 'None'}'")
        
        print(f"\nFinal hierarchy for chunk 844:")
        print(f"  H1: {current_titles[1]}")
        print(f"  H2: {current_titles[2]}")
        print(f"  H3: {current_titles[3]}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_title_positions()