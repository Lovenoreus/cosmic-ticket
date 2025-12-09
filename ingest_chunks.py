#!/usr/bin/env python3
"""
Script to ingest chunks into Qdrant vector database.
Invokes the ingest_chunks_into_qdrant function from tools.vector_database_tools
Configuration is read from config.py (or .env file)
"""

import asyncio
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import config
from tools.vector_database_tools import ingest_chunks_into_qdrant


async def main():
    """Main function to run the ingestion process"""
    
    # Read configuration from config file
    json_path = config.INGEST_JSON_PATH
    collection_name = config.COSMIC_DATABASE_COLLECTION_NAME
    batch_size = config.INGEST_BATCH_SIZE
    recreate_collection = config.INGEST_RECREATE_COLLECTION
    
    print("=" * 70)
    print("Qdrant Chunk Ingestion Tool")
    print("=" * 70)
    print(f"JSON Path:       {json_path or 'Using default from vector_database_tools'}")
    print(f"Collection:      {collection_name}")
    print(f"Batch Size:      {batch_size}")
    print(f"Recreate:        {recreate_collection}")
    print("=" * 70)
    print()
    
    if recreate_collection:
        response = input("WARNING: This will DELETE all existing data in the collection. Continue? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Aborted.")
            return
    
    try:
        print("Starting ingestion process...")
        print()
        
        # Create progress bar - will be updated dynamically
        progress_bar = tqdm(
            total=None,  # Unknown initially, will be set when we know total
            desc="Processing chunks",
            unit="chunk",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] | Stored: {postfix[0]}, Failed: {postfix[1]}",
            postfix=["0", "0"],
            dynamic_ncols=True
        )
        
        # Track progress state
        progress_state = {"last_current": 0, "total": 0}
        
        def update_progress(current: int, total: int, stored: int, failed: int):
            """Callback function to update progress bar"""
            # Update total if not set yet
            if progress_bar.total != total:
                progress_bar.total = total
            
            # Calculate how many to increment
            increment = current - progress_state["last_current"]
            if increment > 0:
                progress_bar.update(increment)
            
            progress_state["last_current"] = current
            progress_state["total"] = total
            
            # Update postfix with stored and failed counts
            progress_bar.set_postfix_str(f"Stored: {stored}, Failed: {failed}")
        
        result = await ingest_chunks_into_qdrant(
            json_path=json_path,
            collection_name=collection_name,
            batch_size=batch_size,
            recreate_collection=recreate_collection,
            progress_callback=update_progress
        )
        
        # Close progress bar
        progress_bar.close()
        
        print()
        print("=" * 70)
        print("Ingestion Results")
        print("=" * 70)
        
        if result.get("success"):
            print(f"✓ Success!")
            print(f"  File Path:          {result.get('file_path', 'N/A')}")
            print(f"  Collection Name:    {result.get('collection_name', 'N/A')}")
            print(f"  Vector Size:        {result.get('vector_size', 'N/A')}")
            print(f"  Total Chunks:       {result.get('total_chunks', 0)}")
            print(f"  Stored Chunks:      {result.get('stored_chunks', 0)}")
            print(f"  Failed Chunks:      {len(result.get('failed_chunks', []))}")
            print(f"  Recreated:          {result.get('recreated_collection', False)}")
            
            if result.get('processed_files'):
                print(f"  Processed Files:    {len(result.get('processed_files', []))}")
                for file in result.get('processed_files', []):
                    print(f"    - {file}")
            
            if result.get('failed_chunks'):
                print(f"\n  Failed Chunks ({len(result.get('failed_chunks', []))}):")
                for failed in result.get('failed_chunks', [])[:10]:  # Show first 10
                    print(f"    - Index {failed.get('index')}: {failed.get('reason')}")
                if len(result.get('failed_chunks', [])) > 10:
                    print(f"    ... and {len(result.get('failed_chunks', [])) - 10} more")
            
            if result.get('load_errors'):
                print(f"\n  Load Errors ({len(result.get('load_errors', []))}):")
                for error in result.get('load_errors', [])[:5]:  # Show first 5
                    print(f"    - {error.get('file')}: {error.get('error')}")
            
            if result.get('empty_files'):
                print(f"\n  Empty Files ({len(result.get('empty_files', []))}):")
                for file in result.get('empty_files', [])[:5]:  # Show first 5
                    print(f"    - {file}")
        else:
            print(f"✗ Failed!")
            print(f"  File Path:          {result.get('file_path', 'N/A')}")
            print(f"  Message:            {result.get('message', 'N/A')}")
            
            if result.get('load_errors'):
                print(f"\n  Load Errors:")
                for error in result.get('load_errors', []):
                    print(f"    - {error.get('file')}: {error.get('error')}")
            
            if result.get('empty_files'):
                print(f"\n  Empty Files:")
                for file in result.get('empty_files', []):
                    print(f"    - {file}")
        
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

