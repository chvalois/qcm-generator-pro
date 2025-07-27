#!/usr/bin/env python3
"""
Debug the title filtering process to see what's being removed.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ['PYTHONPATH'] = str(Path(__file__).parent)
os.chdir(Path(__file__).parent)

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

from src.services.pdf_processor import PDFProcessor
from src.services.title_detector import TitleDetector

def debug_filtering():
    """Debug the title filtering process."""
    
    pdf_path = "/mnt/c/Users/massi/Documents/Perso/Formation Microsoft/DP-600/Formation_DP-600T00_Complete.pdf"
    
    print(f"Debugging title filtering for: {pdf_path}")
    
    processor = PDFProcessor()
    
    try:
        text = processor.extract_text(Path(pdf_path))
        pages_data = processor.extract_text_by_pages(Path(pdf_path))
        
        # Initialize title detector
        detector = TitleDetector()
        
        # Detect all titles (before filtering)
        title_candidates = detector.detect_titles_in_text(text, pages_data)
        
        print(f"Found {len(title_candidates)} title candidates BEFORE filtering")
        
        # Look for ALL Module 3 titles (not just Apache Spark)
        module3_candidates = []
        for title in title_candidates:
            if "module 3" in title.text.lower():
                module3_candidates.append(title)
        
        print(f"\nFound {len(module3_candidates)} 'Module 3' candidates BEFORE filtering:")
        print("=" * 80)
        
        for title in module3_candidates:
            page_num = getattr(title, 'page_number', 'unknown')
            pattern_score = title.features.get('pattern_score', 0)
            format_score = title.features.get('format_score', 0)
            
            print(f"Page {page_num:3} | Conf: {title.confidence:.3f} | Pattern: {pattern_score:.3f} | Format: {format_score:.3f}")
            print(f"    Text: {title.text}")
            print(f"    Features: {title.features}")
            
            # Test filtering criteria
            keep_title = (
                # Strong pattern recognition (numbered, lettered titles)
                (title.confidence > 0.7 and pattern_score > 0.5) or
                
                # Strong formatting indicators (ALL CAPS, good structure)
                (title.confidence > 0.6 and format_score > 0.6) or
                
                # Moderate confidence with some pattern recognition
                (title.confidence > 0.5 and pattern_score > 0.3) or
                
                # High confidence alone (for well-formatted titles)
                (title.confidence > 0.8)
            )
            
            if keep_title:
                print(f"    ✅ KEPT: Meets filtering criteria")
            else:
                print(f"    ❌ FILTERED OUT: Does not meet criteria")
                print(f"       - Conf > 0.7 and Pattern > 0.5: {title.confidence > 0.7 and pattern_score > 0.5}")
                print(f"       - Conf > 0.6 and Format > 0.6: {title.confidence > 0.6 and format_score > 0.6}")
                print(f"       - Conf > 0.5 and Pattern > 0.3: {title.confidence > 0.5 and pattern_score > 0.3}")
                print(f"       - Conf > 0.8: {title.confidence > 0.8}")
            print()
        
        # Now test the actual filtering as used in the algorithm
        print(f"\nApplying the filtering algorithm:")
        print("=" * 80)
        
        filtered_titles = []
        for title in title_candidates:
            confidence = title.confidence
            pattern_score = title.features.get('pattern_score', 0)
            format_score = title.features.get('format_score', 0)
            
            keep_title = (
                (confidence > 0.7 and pattern_score > 0.5) or
                (confidence > 0.6 and format_score > 0.6) or
                (confidence > 0.5 and pattern_score > 0.3) or
                (confidence > 0.8)
            )
            
            if keep_title:
                filtered_titles.append(title)
        
        print(f"After filtering: {len(filtered_titles)} titles remaining")
        
        # Check if our Module 3 titles survived filtering
        filtered_module3 = []
        for title in filtered_titles:
            if "module 3" in title.text.lower():
                filtered_module3.append(title)
        
        print(f"Module 3 titles that survived filtering: {len(filtered_module3)}")
        
        for title in filtered_module3:
            page_num = getattr(title, 'page_number', 'unknown')
            print(f"  Page {page_num}: {title.text}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_filtering()