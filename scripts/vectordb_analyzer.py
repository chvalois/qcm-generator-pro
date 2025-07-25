#!/usr/bin/env python3
"""
Analyseur avancé pour la base vectorielle ChromaDB
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import chromadb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import pandas as pd
from typing import Dict, List, Any
import json


class VectorDBAnalyzer:
    def __init__(self, db_path: str = "./data/vectorstore"):
        """Initialize the analyzer."""
        self.client = chromadb.PersistentClient(path=db_path)
        self.collections = [col.name for col in self.client.list_collections()]
    
    def analyze_chunk_sizes(self, collection_name: str):
        """Analyze the distribution of chunk sizes."""
        print(f"\n📊 Chunk Size Analysis for '{collection_name}'")
        print("-" * 50)
        
        collection = self.client.get_collection(collection_name)
        results = collection.get(include=['documents'])
        
        sizes = [len(doc) for doc in results['documents']]
        
        print(f"📈 Statistics:")
        print(f"   Total chunks: {len(sizes)}")
        print(f"   Average size: {np.mean(sizes):.0f} characters")
        print(f"   Median size: {np.median(sizes):.0f} characters")
        print(f"   Min size: {min(sizes)} characters")
        print(f"   Max size: {max(sizes)} characters")
        print(f"   Std deviation: {np.std(sizes):.0f} characters")
        
        # Distribution
        ranges = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, float('inf'))]
        range_counts = {f"{r[0]}-{r[1] if r[1] != float('inf') else '2000+'}": 0 for r in ranges}
        
        for size in sizes:
            for range_name, (min_val, max_val) in zip(range_counts.keys(), ranges):
                if min_val <= size < max_val:
                    range_counts[range_name] += 1
                    break
        
        print(f"\n📊 Size distribution:")
        for range_name, count in range_counts.items():
            percentage = (count / len(sizes)) * 100
            print(f"   {range_name} chars: {count} chunks ({percentage:.1f}%)")
    
    def analyze_themes(self, collection_name: str):
        """Analyze theme distribution."""
        print(f"\n🎯 Theme Analysis for '{collection_name}'")
        print("-" * 50)
        
        collection = self.client.get_collection(collection_name)
        results = collection.get(include=['metadatas'])
        
        themes = []
        for metadata in results['metadatas']:
            if metadata and 'theme' in metadata:
                themes.append(metadata['theme'])
            else:
                themes.append('Unknown')
        
        theme_counts = Counter(themes)
        
        print(f"📊 Theme distribution:")
        for theme, count in theme_counts.most_common():
            percentage = (count / len(themes)) * 100
            print(f"   {theme}: {count} chunks ({percentage:.1f}%)")
    
    def analyze_documents(self, collection_name: str):
        """Analyze document distribution."""
        print(f"\n📄 Document Analysis for '{collection_name}'")
        print("-" * 50)
        
        collection = self.client.get_collection(collection_name)
        results = collection.get(include=['metadatas'])
        
        doc_info = {}
        for metadata in results['metadatas']:
            if metadata:
                doc_id = metadata.get('document_id', 'unknown')
                filename = metadata.get('filename', 'unknown')
                page = metadata.get('page_number', 'unknown')
                
                if doc_id not in doc_info:
                    doc_info[doc_id] = {
                        'filename': filename,
                        'chunks': 0,
                        'pages': set()
                    }
                
                doc_info[doc_id]['chunks'] += 1
                if page != 'unknown':
                    doc_info[doc_id]['pages'].add(page)
        
        print(f"📊 Document distribution:")
        for doc_id, info in doc_info.items():
            pages_str = f"{len(info['pages'])} pages" if info['pages'] else "unknown pages"
            print(f"   Doc {doc_id} ({info['filename']}): {info['chunks']} chunks, {pages_str}")
    
    def find_similar_chunks(self, collection_name: str, chunk_id: str, n_results: int = 5):
        """Find chunks similar to a specific chunk."""
        print(f"\n🔍 Finding similar chunks to '{chunk_id}' in '{collection_name}'")
        print("-" * 50)
        
        collection = self.client.get_collection(collection_name)
        
        # Get the target chunk
        target = collection.get(
            ids=[chunk_id],
            include=['documents', 'metadatas']
        )
        
        if not target['documents']:
            print(f"❌ Chunk '{chunk_id}' not found")
            return
        
        target_content = target['documents'][0]
        print(f"🎯 Target chunk content: {target_content[:200]}...")
        
        # Find similar chunks
        results = collection.query(
            query_texts=[target_content],
            n_results=n_results + 1,  # +1 because it will include itself
            include=['documents', 'metadatas', 'distances']
        )
        
        print(f"\n📊 Similar chunks:")
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0][1:],  # Skip first (itself)
            results['metadatas'][0][1:],
            results['distances'][0][1:]
        )):
            similarity = 1 - distance
            print(f"\n🔹 Similar chunk {i+1} (similarity: {similarity:.3f}):")
            print(f"   Content: {doc[:150]}...")
            if metadata:
                print(f"   Metadata: {json.dumps(metadata, indent=6, ensure_ascii=False)}")
    
    def export_full_analysis(self, output_file: str = "vectordb_analysis.json"):
        """Export full analysis to JSON."""
        analysis = {
            'collections': {},
            'summary': {}
        }
        
        total_chunks = 0
        
        for collection_name in self.collections:
            collection = self.client.get_collection(collection_name)
            results = collection.get(include=['documents', 'metadatas'])
            
            # Basic stats
            chunks = len(results['documents'])
            total_chunks += chunks
            
            sizes = [len(doc) for doc in results['documents']]
            
            # Theme analysis
            themes = []
            for metadata in results['metadatas']:
                if metadata and 'theme' in metadata:
                    themes.append(metadata['theme'])
                else:
                    themes.append('Unknown')
            
            theme_counts = dict(Counter(themes))
            
            # Document analysis
            doc_counts = {}
            for metadata in results['metadatas']:
                if metadata:
                    doc_id = metadata.get('document_id', 'unknown')
                    doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            
            analysis['collections'][collection_name] = {
                'total_chunks': chunks,
                'size_stats': {
                    'mean': float(np.mean(sizes)),
                    'median': float(np.median(sizes)),
                    'min': int(min(sizes)) if sizes else 0,
                    'max': int(max(sizes)) if sizes else 0,
                    'std': float(np.std(sizes))
                },
                'theme_distribution': theme_counts,
                'document_distribution': doc_counts
            }
        
        analysis['summary'] = {
            'total_collections': len(self.collections),
            'total_chunks': total_chunks,
            'collection_names': self.collections
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Full analysis exported to: {output_file}")
    
    def run_complete_analysis(self):
        """Run complete analysis on all collections."""
        print("🔍 Complete Vector Database Analysis")
        print("=" * 60)
        
        print(f"\n📚 Found {len(self.collections)} collection(s): {', '.join(self.collections)}")
        
        for collection_name in self.collections:
            print(f"\n{'='*60}")
            print(f"Analyzing Collection: {collection_name}")
            print("=" * 60)
            
            self.analyze_chunk_sizes(collection_name)
            self.analyze_themes(collection_name)
            self.analyze_documents(collection_name)
        
        # Export full analysis
        self.export_full_analysis()


def main():
    """Main function."""
    analyzer = VectorDBAnalyzer()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "analyze":
            analyzer.run_complete_analysis()
        elif command == "similar" and len(sys.argv) >= 4:
            collection_name = sys.argv[2]
            chunk_id = sys.argv[3]
            analyzer.find_similar_chunks(collection_name, chunk_id)
        else:
            print("Usage:")
            print("  python vectordb_analyzer.py analyze")
            print("  python vectordb_analyzer.py similar <collection> <chunk_id>")
    else:
        analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()