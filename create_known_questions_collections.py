#!/usr/bin/env python3
"""
Create a Qdrant vector store from known_questions.json.

This script:
1. Loads known_questions.json
2. Creates searchable text from each template (issue_category, description, keywords, text)
3. Generates embeddings using the configured embedding model
4. Stores templates in Qdrant with full template data as payload
"""

import json
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from uuid import uuid4

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "tools"))
from tools.vector_database_tools import async_embed_with_fallback
from langchain_openai import OpenAIEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
import config
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def create_searchable_text(template: Dict[str, Any]) -> str:
    """
    Create a searchable text representation of a template.
    
    Combines the most relevant fields for semantic search:
    - issue_category: The category name
    - description: Detailed description
    - keywords: Relevant keywords (joined)
    - text: Summary text
    """
    parts = []
    
    # Add issue category
    issue_category = template.get("issue_category", "")
    if issue_category:
        parts.append(f"Issue Category: {issue_category}")
    
    # Add description
    description = template.get("description", "")
    if description:
        parts.append(f"Description: {description}")
    
    # Add keywords
    keywords = template.get("keywords", [])
    if keywords:
        keywords_str = ", ".join(keywords)
        parts.append(f"Keywords: {keywords_str}")
    
    # Add summary text
    text = template.get("text", "")
    if text:
        parts.append(f"Summary: {text}")
    
    return "\n".join(parts)


async def ingest_known_questions(
    json_path: str = None,
    collection_name: str = None,
    recreate_collection: bool = False
) -> Dict[str, Any]:
    """
    Load known_questions.json and create Qdrant collection with embeddings.
    
    Args:
        json_path: Path to known_questions.json (default: ./known_questions.json)
        collection_name: Qdrant collection name (default: from config)
        recreate_collection: If True, delete and recreate the collection
    
    Returns:
        Dictionary with success status and statistics
    """
    # Resolve paths
    if json_path:
        questions_path = Path(json_path).expanduser().resolve()
    else:
        questions_path = Path(__file__).parent / "known_questions.json"
    
    if not questions_path.exists():
        return {
            "success": False,
            "error": f"File not found: {questions_path}"
        }
    
    # Load known questions
    print(f"Loading known questions from: {questions_path}")
    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            templates = json.load(f)
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Failed to parse JSON: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error reading file: {e}"
        }
    
    if not templates or not isinstance(templates, list):
        return {
            "success": False,
            "error": "Invalid format: expected a list of templates"
        }
    
    print(f"Loaded {len(templates)} templates")
    
    # Get collection name
    collection = collection_name or config.KNOWN_QUESTIONS_COLLECTION_NAME
    
    # Initialize embedding model
    openai_embedder = None
    if not config.USE_OLLAMA:
        openai_embedder = OpenAIEmbeddings(
            model=config.EMBEDDINGS_MODEL_NAME,
            openai_api_key=config.OPENAI_API_KEY
        )
    
    # Generate first embedding to determine vector size
    first_template = templates[0]
    first_searchable_text = create_searchable_text(first_template)
    
    print(f"Generating first embedding to determine vector size...")
    first_embedding = await async_embed_with_fallback(
        query=first_searchable_text,
        ollama_model=config.EMBEDDINGS_MODEL_NAME if config.USE_OLLAMA else None,
        ollama_base_url=config.OLLAMA_BASE_URL if config.USE_OLLAMA else None,
        openai_embedder=openai_embedder,
        timeout=30.0
    )
    
    if not first_embedding:
        return {
            "success": False,
            "error": "Failed to generate embedding for first template"
        }
    
    vector_size = len(first_embedding)
    print(f"Vector size: {vector_size} dimensions")
    
    # Prepare Qdrant client
    qdrant_url = f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}"
    print(f"Connecting to Qdrant at: {qdrant_url}")
    qdrant_client = AsyncQdrantClient(url=qdrant_url)
    
    try:
        # Create or recreate collection
        vector_params = rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE
        )
        
        if recreate_collection:
            print(f"Recreating collection '{collection}'...")
            await qdrant_client.recreate_collection(
                collection_name=collection,
                vectors_config=vector_params
            )
            print(f"Collection '{collection}' recreated")
        else:
            try:
                await qdrant_client.get_collection(collection_name=collection)
                print(f"Collection '{collection}' already exists, adding/updating points...")
            except Exception:
                print(f"Creating new collection '{collection}'...")
                await qdrant_client.create_collection(
                    collection_name=collection,
                    vectors_config=vector_params
                )
                print(f"Collection '{collection}' created")
        
        # Process templates and create points
        points: List[rest.PointStruct] = []
        stored_count = 0
        failed_templates: List[Dict] = []
        
        print(f"\nProcessing {len(templates)} templates...")
        
        for index, template in enumerate(templates):
            # Create searchable text
            searchable_text = create_searchable_text(template)
            
            if not searchable_text.strip():
                failed_templates.append({
                    "index": index,
                    "reason": "empty searchable text",
                    "template_id": template.get("section_id", f"template_{index}")
                })
                continue
            
            # Generate embedding
            if index == 0:
                embedding = first_embedding
            else:
                embedding = await async_embed_with_fallback(
                    query=searchable_text,
                    ollama_model=config.EMBEDDINGS_MODEL_NAME if config.USE_OLLAMA else None,
                    ollama_base_url=config.OLLAMA_BASE_URL if config.USE_OLLAMA else None,
                    openai_embedder=openai_embedder,
                    timeout=30.0
                )
            
            if not embedding:
                failed_templates.append({
                    "index": index,
                    "reason": "embedding_failed",
                    "template_id": template.get("section_id", f"template_{index}")
                })
                continue
            
            # Create point ID - Qdrant only accepts unsigned integers or UUIDs
            # Always use UUID to avoid issues with section_id formats like "1.1"
            point_id = str(uuid4())
            
            # Create payload with full template data
            # Store section_id in payload for reference
            payload = {
                "searchable_text": searchable_text,
                "template_index": index,  # Store original index for reference
                **template  # Include all template fields
            }
            
            # Create point
            point = rest.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            
            points.append(point)
            
            # Progress update
            if (index + 1) % 10 == 0:
                print(f"  Processed {index + 1}/{len(templates)} templates...")
        
        # Upsert all points
        if points:
            print(f"\nUpserting {len(points)} points to Qdrant...")
            await qdrant_client.upsert(
                collection_name=collection,
                points=points
            )
            stored_count = len(points)
            print(f"✓ Successfully stored {stored_count} templates")
        
        if failed_templates:
            print(f"⚠ Failed to process {len(failed_templates)} templates")
            if config.DEBUG:
                for failed in failed_templates[:5]:  # Show first 5 failures
                    print(f"  - Index {failed['index']}: {failed['reason']}")
        
        return {
            "success": True,
            "collection_name": collection,
            "vector_size": vector_size,
            "total_templates": len(templates),
            "stored_templates": stored_count,
            "failed_templates": len(failed_templates),
            "failed_details": failed_templates if config.DEBUG else None
        }
    
    finally:
        await qdrant_client.close()


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create Qdrant vector store from known_questions.json"
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Path to known_questions.json (default: ./known_questions.json)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help=f"Qdrant collection name (default: {config.KNOWN_QUESTIONS_COLLECTION_NAME})"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the collection (deletes existing data)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Known Questions Vector Store Creator")
    print("=" * 80)
    print(f"Embedding model: {config.EMBEDDINGS_MODEL_NAME}")
    print(f"Using Ollama: {config.USE_OLLAMA}")
    if config.USE_OLLAMA:
        print(f"Ollama URL: {config.OLLAMA_BASE_URL}")
    print(f"Qdrant: {config.QDRANT_HOST}:{config.QDRANT_PORT}")
    print(f"Collection: {args.collection or config.KNOWN_QUESTIONS_COLLECTION_NAME}")
    print("=" * 80)
    print()
    
    result = await ingest_known_questions(
        json_path=args.json_path,
        collection_name=args.collection,
        recreate_collection=args.recreate
    )
    
    if result.get("success"):
        print("\n" + "=" * 80)
        print("✓ SUCCESS")
        print("=" * 80)
        print(f"Collection: {result['collection_name']}")
        print(f"Vector size: {result['vector_size']} dimensions")
        print(f"Total templates: {result['total_templates']}")
        print(f"Stored templates: {result['stored_templates']}")
        if result.get("failed_templates", 0) > 0:
            print(f"Failed templates: {result['failed_templates']}")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("✗ FAILED")
        print("=" * 80)
        print(f"Error: {result.get('error', 'Unknown error')}")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

