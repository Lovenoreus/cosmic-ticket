#!/usr/bin/env python3
"""Query Qdrant with hybrid (vector + keyword) search and return top chunks with IMPROVED reconstruction."""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Add parent directory to path to import root config
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_core.runnables.utils import ConfigurableField
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import config 
import httpx

# Inject truststore for SSL certificate handling (self-signed certificates) - only if using Ollama
if config.USE_OLLAMA:
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        # truststore not installed, skip SSL injection
        pass


# FIXED: Increased padding significantly
SECTION_PADDING = 5   # was 2, now 5 - gives ~11 chunks instead of 5
MAX_CONTEXT_CHUNKS = 15  # Safety limit to prevent token overflow

EMBEDDING_MODEL = config.EMBEDDINGS_MODEL_NAME
DEFAULT_COLLECTION = config.COSMIC_DATABASE_COLLECTION_NAME
DEFAULT_QDRANT_HOST = config.QDRANT_HOST
DEFAULT_QDRANT_PORT = config.QDRANT_PORT
DEFAULT_LIMIT = 1


@dataclass
class HybridResult:
    point_id: str
    score: float
    payload: Dict


def embed_query_openai(client: OpenAI, query: str) -> List[float]:
    """Generate embedding using OpenAI"""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    embedding = response.data[0].embedding
    if not embedding:
        raise RuntimeError("Received empty embedding from OpenAI")
    return embedding


def embed_query_ollama(query: str, model_name: str, ollama_base_url: str) -> List[float]:
    """Generate embedding using Ollama"""
    import requests
    payload = {
        "model": model_name,
        "prompt": query
    }
    
    # Prepare headers with JWT token if available
    headers = {"Content-Type": "application/json"}
    jwt_token = os.getenv("OLLAMA_JWT_TOKEN")
    if jwt_token:
        headers["Authorization"] = f"Bearer {jwt_token}"
    
    try:
        response = requests.post(
            f"{ollama_base_url}/api/embeddings",
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        embedding = result.get("embedding")
        if not embedding:
            raise RuntimeError("Received empty embedding from Ollama")
        return embedding
    except Exception as e:
        raise RuntimeError(f"Ollama embedding failed: {e}")


def embed_query(query: str, openai_client: OpenAI = None, ollama_model: str = None, ollama_base_url: str = None) -> List[float]:
    """Generate embedding using Ollama or OpenAI based on provided parameters"""
    # Prefer Ollama if model and URL are provided, otherwise use OpenAI
    if ollama_model and ollama_base_url:
        return embed_query_ollama(query, ollama_model, ollama_base_url)
    elif openai_client:
        return embed_query_openai(openai_client, query)
    else:
        raise RuntimeError("Either OpenAI client or Ollama configuration must be provided")


def vector_search(
    client: QdrantClient,
    collection: str,
    embedding: List[float],
    limit: int,
) -> Iterable[HybridResult]:
    hits = client.search(
        collection_name=collection,
        query_vector=embedding,
        limit=limit * 2,
        with_payload=True,
        with_vectors=False,
    )
    for hit in hits:
        yield HybridResult(point_id=str(hit.id), score=float(hit.score or 0.0), payload=hit.payload or {})


def keyword_search(
    client: QdrantClient,
    collection: str,
    query: str,
    limit: int,
) -> Iterable[HybridResult]:
    keyword_filter = rest.Filter(
        should=[
            rest.FieldCondition(key="text", match=rest.MatchText(text=query)),
            rest.FieldCondition(key="metadata.context_summary", match=rest.MatchText(text=query)),
            rest.FieldCondition(key="metadata.keywords", match=rest.MatchText(text=query)),
        ]
    )

    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=keyword_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    base_score = 1.5
    decay = 0.01
    for index, point in enumerate(points):
        score = base_score - index * decay
        yield HybridResult(point_id=str(point.id), score=score, payload=point.payload or {})


def merge_results(
    *sources: Iterable[HybridResult],
    limit: int,
) -> List[HybridResult]:
    merged: Dict[str, HybridResult] = {}
    for source in sources:
        for result in source:
            existing = merged.get(result.point_id)
            if existing is None or result.score > existing.score:
                merged[result.point_id] = result

    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:limit]


def fetch_surrounding_chunks(
    client: QdrantClient,
    collection: str,
    source_file: str,
    section_id: int,
    padding: int = SECTION_PADDING
) -> List[Dict]:
    """Fetch chunks from same document around a given section_id with IMPROVED logic."""

    # FIXED: More aggressive padding
    min_id = max(1, section_id - padding)
    max_id = section_id + padding

    section_filter = rest.Filter(
        must=[
            rest.FieldCondition(
                key="source_file",
                match=rest.MatchValue(value=source_file)
            ),
            rest.FieldCondition(
                key="section_id",
                range=rest.Range(
                    gte=min_id,
                    lte=max_id
                )
            ),
        ]
    )

    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=section_filter,
        limit=200,
        with_payload=True,
        with_vectors=False,
    )

    chunks = []
    for p in points:
        payload = p.payload or {}
        sid = payload.get("section_id")
        if sid is not None:
            chunks.append(payload)

    # Order by section_id to reconstruct the document piece
    chunks.sort(key=lambda c: c.get("section_id", 0))
    
    # FIXED: Limit to prevent token overflow but ensure we have enough context
    if len(chunks) > MAX_CONTEXT_CHUNKS:
        # Keep chunks centered around the target section_id
        target_idx = next((i for i, c in enumerate(chunks) if c.get("section_id") == section_id), len(chunks) // 2)
        half_window = MAX_CONTEXT_CHUNKS // 2
        start = max(0, target_idx - half_window)
        end = min(len(chunks), target_idx + half_window + 1)
        chunks = chunks[start:end]
    
    return chunks


def run_query(
    client: QdrantClient,
    openai_client: OpenAI = None,
    collection: str = None,
    query: str = None,
    limit: int = None,
    ollama_model: str = None,
    ollama_base_url: str = None,
) -> Dict[str, Any]:
    embedding = embed_query(query, openai_client, ollama_model, ollama_base_url)
    vector_results = vector_search(client, collection, embedding, limit)
    keyword_results = keyword_search(client, collection, query, limit)
    merged = merge_results(vector_results, keyword_results, limit=limit)

    if not merged:
        return {"results": [], "reconstructed_context": ""}

    # Take the best matching chunk
    top = merged[0]
    payload = top.payload or {}

    source_file = payload.get("source_file")
    section_id = payload.get("section_id")

    # FIXED: Better fallback handling
    if source_file is None or section_id is None:
        # Fallback: use the matched chunk's text directly
        fallback_text = payload.get("text", "")
        return {
            "results": merged,
            "reconstructed_context": fallback_text,
            "source_file": source_file or "unknown",
            "anchor_section_id": section_id,
            "chunk_count": 1,
            "reconstruction_method": "fallback_single_chunk"
        }

    # Fetch document neighbors
    surrounding = fetch_surrounding_chunks(
        client,
        collection,
        source_file,
        section_id,
        padding=SECTION_PADDING,
    )

    # FIXED: Better handling when reconstruction returns nothing
    if not surrounding:
        fallback_text = payload.get("text", "")
        return {
            "results": merged,
            "reconstructed_context": fallback_text,
            "source_file": source_file,
            "anchor_section_id": section_id,
            "chunk_count": 1,
            "reconstruction_method": "fallback_empty_surrounding"
        }

    # Build unified ordered context with chunk markers for better LLM understanding
    context_parts = []
    for i, chunk in enumerate(surrounding):
        text = chunk.get("text", "").strip()
        if text:
            chunk_id = chunk.get("section_id", i)
            # Mark the target chunk clearly
            marker = " [MOST RELEVANT]" if chunk_id == section_id else ""
            context_parts.append(f"--- Chunk {chunk_id}{marker} ---\n{text}")
    
    full_context = "\n\n".join(context_parts)

    return {
        "results": merged,
        "reconstructed_context": full_context,
        "source_file": source_file,
        "anchor_section_id": section_id,
        "chunk_count": len(surrounding),
        "reconstruction_method": "full_reconstruction"
    }


def ask_question(
    question: str,
    *,
    collection: Optional[str] = None,
    limit: Optional[int] = None,
    qdrant_host: Optional[str] = None,
    qdrant_port: Optional[int] = None,
    openai_api_key: Optional[str] = None,
    ollama_model: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
) -> Dict[str, Any]:

    collection_name = collection or os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    top_k = limit or DEFAULT_LIMIT
    host = qdrant_host or os.getenv("QDRANT_HOST", DEFAULT_QDRANT_HOST)
    port = qdrant_port or int(os.getenv("QDRANT_PORT", DEFAULT_QDRANT_PORT))

    load_dotenv()

    # Determine which provider to use based on passed parameters or flag
    use_ollama = config.USE_OLLAMA or (ollama_model is not None and ollama_base_url is not None)
    
    openai_client = None
    if not use_ollama:
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in environment or .env file")
        openai_client = OpenAI(api_key=api_key)
    
    if use_ollama:
        ollama_model = ollama_model or config.EMBEDDINGS_MODEL_NAME
        ollama_base_url = ollama_base_url or config.OLLAMA_BASE_URL
        if not ollama_model or not ollama_base_url:
            raise EnvironmentError("Ollama model and base URL must be provided when using Ollama")

    qdrant_client = QdrantClient(host=host, port=port)

    try:
        return run_query(
            qdrant_client,
            openai_client,
            collection_name,
            question,
            max(1, top_k),
            ollama_model,
            ollama_base_url,
        )
    finally:
        qdrant_client.close()


def format_result(result: HybridResult, index: int) -> str:
    payload = result.payload or {}
    text = (payload.get("text") or "").strip()
    snippet = text[:400] + ("…" if len(text) > 400 else "")
    source = payload.get("source_file") or payload.get("metadata", {}).get("source_file")
    lines = [
        f"Result {index + 1} (score={result.score:.3f})",
        snippet or "[No text content]",
    ]
    if source:
        lines.append(f"Source: {source}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions against a Qdrant vector store")
    parser.add_argument("query", nargs="?", help="Question to ask; enter interactive mode if omitted")
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", DEFAULT_QDRANT_HOST))
    parser.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", DEFAULT_QDRANT_PORT)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def handle_single_query(prompt: str) -> None:
        if not prompt.strip():
            print("Query cannot be empty", file=sys.stderr)
            return

        try:
            response = ask_question(
                prompt,
                collection=args.collection,
                limit=max(1, args.limit),
                qdrant_host=args.qdrant_host,
                qdrant_port=args.qdrant_port,
            )
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return

        if not response:
            print("No matching chunks found")
            return

        results = response.get("results", [])
        reconstructed = response.get("reconstructed_context", "")
        method = response.get("reconstruction_method", "unknown")

        if not results:
            print("No matching chunks found")
            return

        # Print top-k ranked chunks
        for idx, result in enumerate(results):
            print(format_result(result, idx))
            print("-" * 80)

        # Print reconstructed document slice
        print(f"\nReconstructed context (method: {method}):")
        print("----------------------------------------")
        if reconstructed:
            snippet = reconstructed[:1000]
            print(snippet + ("..." if len(reconstructed) > 1000 else ""))
        else:
            print("[No reconstructed context available]")
        print()

    if args.query:
        handle_single_query(args.query)
        return

    # Interactive mode
    print("Interactive Qdrant search. Press Ctrl+C or enter empty query to exit.")
    try:
        while True:
            prompt = input("Question> ").strip()
            if not prompt:
                break
            handle_single_query(prompt)
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()