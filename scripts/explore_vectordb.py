#!/usr/bin/env python3
"""
Script pour explorer et visualiser le contenu de la base vectorielle ChromaDB
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import chromadb
import pandas as pd
from typing import Dict, List, Any
import json


def connect_to_vectordb(db_path: str = "./data/vectorstore") -> chromadb.PersistentClient:
    """Connect to ChromaDB."""
    return chromadb.PersistentClient(path=db_path)


def list_collections(client: chromadb.PersistentClient) -> List[str]:
    """List all collections in the database."""
    collections = client.list_collections()
    return [col.name for col in collections]


def explore_collection(client: chromadb.PersistentClient, collection_name: str):
    """Explore a specific collection."""
    print(f"\n{'='*50}")
    print(f"Collection: {collection_name}")
    print(f"{'='*50}")
    
    try:
        collection = client.get_collection(collection_name)
        
        # Get collection info
        count = collection.count()
        print(f"📊 Total chunks: {count}")
        
        if count == 0:
            print("❌ Collection is empty")
            return
        
        # Get all documents with metadata
        results = collection.get(
            include=['documents', 'metadatas', 'embeddings']
        )
        
        print(f"\n📋 Sample chunks (first 5):")
        print("-" * 50)
        
        for i in range(min(5, len(results['documents']))):
            print(f"\n🔹 Chunk {i+1}:")
            print(f"   ID: {results['ids'][i]}")
            
            # Document content (truncated)
            doc = results['documents'][i]
            doc_preview = doc[:200] + "..." if len(doc) > 200 else doc
            print(f"   Content: {doc_preview}")
            
            # Metadata
            metadata = results['metadatas'][i] if results['metadatas'] else {}
            print(f"   Metadata: {json.dumps(metadata, indent=6, ensure_ascii=False)}")
            
            # Embedding info
            if results['embeddings']:
                embedding = results['embeddings'][i]
                print(f"   Embedding: [{len(embedding)} dimensions]")
                print(f"   Sample values: {embedding[:5]}...")
            
            print("-" * 30)
        
        # Metadata analysis
        if results['metadatas']:
            print(f"\n📈 Metadata Analysis:")
            metadata_keys = set()
            for meta in results['metadatas']:
                if meta:
                    metadata_keys.update(meta.keys())
            
            print(f"   Available metadata fields: {list(metadata_keys)}")
            
            # Count by document
            doc_counts = {}
            theme_counts = {}
            
            for meta in results['metadatas']:
                if meta:
                    doc_id = meta.get('document_id', 'unknown')
                    doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
                    
                    theme = meta.get('theme', 'unknown')
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
            
            print(f"\n   📄 Chunks by document:")
            for doc_id, count in doc_counts.items():
                print(f"      - Document {doc_id}: {count} chunks")
            
            print(f"\n   🎯 Chunks by theme:")
            for theme, count in theme_counts.items():
                print(f"      - {theme}: {count} chunks")
        
    except Exception as e:
        print(f"❌ Error exploring collection '{collection_name}': {e}")


def search_in_collection(client: chromadb.PersistentClient, collection_name: str, query: str, n_results: int = 3):
    """Search for similar chunks in a collection."""
    print(f"\n🔍 Searching in '{collection_name}' for: '{query}'")
    print("-" * 50)
    
    try:
        collection = client.get_collection(collection_name)
        
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['documents'] or not results['documents'][0]:
            print("❌ No results found")
            return
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"\n🔹 Result {i+1} (similarity: {1-distance:.3f}):")
            print(f"   Content: {doc[:300]}...")
            print(f"   Metadata: {json.dumps(metadata, indent=6, ensure_ascii=False)}")
            print("-" * 30)
            
    except Exception as e:
        print(f"❌ Error searching in collection '{collection_name}': {e}")


def export_collection_to_csv(client: chromadb.PersistentClient, collection_name: str, output_file: str):
    """Export collection data to CSV."""
    try:
        collection = client.get_collection(collection_name)
        results = collection.get(include=['documents', 'metadatas'])
        
        data = []
        for i, doc in enumerate(results['documents']):
            row = {
                'id': results['ids'][i],
                'content': doc,
                'content_length': len(doc)
            }
            
            # Add metadata fields
            if results['metadatas'] and results['metadatas'][i]:
                row.update(results['metadatas'][i])
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ Collection '{collection_name}' exported to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error exporting collection '{collection_name}': {e}")


def main():
    """Main function."""
    print("🔍 ChromaDB Vector Database Explorer")
    print("=" * 50)
    
    # Connect to database
    try:
        client = connect_to_vectordb()
        print("✅ Connected to ChromaDB")
    except Exception as e:
        print(f"❌ Failed to connect to ChromaDB: {e}")
        return
    
    # List collections
    collections = list_collections(client)
    print(f"\n📚 Found {len(collections)} collection(s):")
    for i, col in enumerate(collections, 1):
        print(f"   {i}. {col}")
    
    if not collections:
        print("❌ No collections found")
        return
    
    # Explore each collection
    for collection_name in collections:
        explore_collection(client, collection_name)
    
    # Interactive mode
    print(f"\n{'='*50}")
    print("🔍 Interactive Search Mode")
    print("Enter 'quit' to exit, 'export <collection>' to export")
    print("=" * 50)
    
    while True:
        try:
            command = input("\n> Enter search query or command: ").strip()
            
            if command.lower() == 'quit':
                break
            
            if command.startswith('export '):
                col_name = command[7:].strip()
                if col_name in collections:
                    output_file = f"vectordb_export_{col_name}.csv"
                    export_collection_to_csv(client, col_name, output_file)
                else:
                    print(f"❌ Collection '{col_name}' not found")
                continue
            
            if command:
                # Search in all collections
                for col_name in collections:
                    search_in_collection(client, col_name, command)
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()