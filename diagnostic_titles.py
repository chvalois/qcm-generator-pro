#!/usr/bin/env python3
"""
Script de diagnostic pour analyser la détection des titres
dans le document Formation_DP-600T00_Complete.pdf
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_document_for_missing_titles():
    """Analyser le document pour identifier les titres manqués."""
    
    # Try to find processed document data
    from src.services.document_manager import get_document_manager
    from src.services.title_detector import TitleDetector
    
    doc_manager = get_document_manager()
    
    # Get list of documents to find DP-600
    try:
        stored_docs = doc_manager.list_documents()
        
        dp600_doc = None
        for doc in stored_docs:
            if "DP-600" in doc.get('filename', '') or "Formation" in doc.get('filename', ''):
                dp600_doc = doc
                break
        
        if not dp600_doc:
            print("❌ Document DP-600 non trouvé dans les documents traités")
            print("📋 Documents disponibles:")
            for doc in stored_docs:
                print(f"  - {doc.get('filename', 'N/A')} (ID: {doc.get('id', 'N/A')})")
            return
        
        print(f"✅ Document trouvé: {dp600_doc['filename']} (ID: {dp600_doc['id']})")
        
        # Get document chunks with page info
        chunks_data = doc_manager.get_document_chunks(dp600_doc['id'], include_titles=True)
        
        if not chunks_data:
            print("❌ Aucun chunk trouvé pour ce document")
            return
        
        print(f"📊 {len(chunks_data)} chunks trouvés")
        
        # Search for mentions of "Module 5" and "Unité" in the text
        module5_chunks = []
        unite_mentions = []
        
        for i, chunk in enumerate(chunks_data):
            chunk_text = chunk.get('chunk_text', '')
            
            # Look for Module 5 content
            if re.search(r'module\s*5', chunk_text, re.IGNORECASE):
                module5_chunks.append((i, chunk))
                print(f"🔍 Module 5 trouvé dans chunk {i}")
            
            # Look for Unité mentions
            unite_matches = re.findall(r'unit[ée]\s*\d+[^\\n]*', chunk_text, re.IGNORECASE | re.MULTILINE)
            if unite_matches:
                for match in unite_matches:
                    unite_mentions.append((i, match.strip(), chunk.get('start_page', 'unknown')))
        
        print(f"\\n📋 {len(unite_mentions)} mentions d'unités trouvées:")
        for chunk_idx, mention, page in unite_mentions:
            hierarchy = chunks_data[chunk_idx].get('title_hierarchy', {})
            h3_title = hierarchy.get('h3_title', 'AUCUN')
            print(f"  Chunk {chunk_idx} (page {page}): '{mention}' -> H3: {h3_title}")
        
        # Check specifically for "Unité 7" and "Unité 8" 
        unite7_8_found = [m for m in unite_mentions if re.search(r'unit[ée]\s*[78]', m[1], re.IGNORECASE)]
        
        if unite7_8_found:
            print(f"\\n🎯 Unités 7 et 8 trouvées:")
            for chunk_idx, mention, page in unite7_8_found:
                chunk = chunks_data[chunk_idx]
                hierarchy = chunk.get('title_hierarchy', {})
                print(f"  Page {page}: '{mention}'")
                print(f"    H1: {hierarchy.get('h1_title', 'AUCUN')}")
                print(f"    H2: {hierarchy.get('h2_title', 'AUCUN')}")
                print(f"    H3: {hierarchy.get('h3_title', 'AUCUN')}")
                print(f"    H4: {hierarchy.get('h4_title', 'AUCUN')}")
                print()
        else:
            print("❌ Unités 7 et 8 non trouvées dans le texte des chunks")
        
        # Now check the raw title detection
        print("\\n🔍 Analyse des titres détectés par le système...")
        
        # Get pages data for title detection
        pages_data = []
        for chunk in chunks_data:
            start_page = chunk.get('start_page', 1)
            end_page = chunk.get('end_page', start_page)
            
            for page_num in range(start_page, end_page + 1):
                if not any(p['page_number'] == page_num for p in pages_data):
                    pages_data.append({
                        'page_number': page_num,
                        'text': chunk.get('chunk_text', '')  # Approximation
                    })
        
        # Run title detection
        full_text = '\\n\\n'.join(chunk.get('chunk_text', '') for chunk in chunks_data)
        
        detector = TitleDetector()
        title_candidates = detector.detect_titles_in_text(full_text, pages_data)
        
        print(f"🎯 {len(title_candidates)} candidats de titres détectés")
        
        # Filter for Module 5 related titles
        module5_titles = [t for t in title_candidates if re.search(r'module\s*5', t.text, re.IGNORECASE)]
        unite_titles = [t for t in title_candidates if re.search(r'unit[ée]\s*\d+', t.text, re.IGNORECASE)]
        
        print(f"\\n📋 Titres liés au Module 5: {len(module5_titles)}")
        for title in module5_titles:
            print(f"  H{title.level} (conf: {title.confidence:.3f}): {title.text}")
        
        print(f"\\n📋 Titres d'unités détectés: {len(unite_titles)}")
        for title in unite_titles:
            page_info = f"page {getattr(title, 'page_number', '?')}" if hasattr(title, 'page_number') else "page unknown"
            print(f"  H{title.level} (conf: {title.confidence:.3f}, {page_info}): {title.text}")
        
        # Look specifically for unite 7 and 8
        unite7_8_titles = [t for t in unite_titles if re.search(r'unit[ée]\s*[78]', t.text, re.IGNORECASE)]
        
        if unite7_8_titles:
            print(f"\\n✅ Unités 7 et 8 détectées comme titres:")
            for title in unite7_8_titles:
                print(f"  H{title.level}: {title.text} (conf: {title.confidence:.3f})")
        else:
            print("\\n❌ Unités 7 et 8 NOT détectées comme titres")
            
            # Debug: check what's happening with confidence scores
            print("\\n🔍 Analyse détaillée des mentions d'Unité 7 et 8...")
            
            # Look in raw text for these specific patterns
            unite78_patterns = [
                r'unit[ée]\s*7[^\\n]*',
                r'unit[ée]\s*8[^\\n]*',
                r'7[\.:\s]*[^\\n]*unit[ée][^\\n]*',
                r'8[\.:\s]*[^\\n]*unit[ée][^\\n]*'
            ]
            
            for pattern in unite78_patterns:
                matches = re.findall(pattern, full_text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    print(f"  Pattern '{pattern}' trouvé {len(matches)} fois:")
                    for i, match in enumerate(matches[:5]):  # Show first 5
                        print(f"    {i+1}: {match.strip()}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_document_for_missing_titles()