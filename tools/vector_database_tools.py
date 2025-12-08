import sys
import json
import openai
import os
import asyncio
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from uuid import uuid4
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
from langchain_openai import OpenAIEmbeddings
import traceback
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import httpx
import requests
from pathlib import Path

# Add parent directory and tools directory to path
tools_dir = Path(__file__).parent
parent_dir = tools_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(tools_dir))

import config

# Import tools modules - now they're in the path
import ask_qdrant
from ask_qdrant import ask_question, HybridResult
import rag
from rag import generate_answer

load_dotenv(find_dotenv())

# Inject truststore for SSL certificate handling (self-signed certificates) - only if using Ollama
if config.USE_OLLAMA:
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        # truststore not installed, skip SSL injection
        pass


DEFAULT_CHUNKS_FILENAME = "chunks_Cosmic_manual_block_handling_v10_0.pdf.json"
DEFAULT_CHUNKS_PATH = (Path(__file__).resolve().parent / DEFAULT_CHUNKS_FILENAME).resolve()
INGEST_EMBED_TIMEOUT = 30.0


def embed_query_using_ollama_embedding_model(query: str, model_name: str, ollama_url: str):
    """Generate embedding using Nomic model via Ollama"""
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
            f"{ollama_url}/api/embeddings",
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result["embedding"]

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Ollama request failed: {e}")
        raise
    except KeyError as e:
        print(f"ERROR: Unexpected response format: {e}")
        print(f"Response: {response.text}")
        raise
    except Exception as e:
        print(f"ERROR: Nomic embedding failed: {e}")
        raise

async def async_embed_with_fallback(
        query: str,
        ollama_model: str = None,
        ollama_base_url: str = None,
        openai_embedder: Optional[OpenAIEmbeddings] = None,
        timeout: float = 10.0
) -> List[float]:
    """Async embedding with Ollama or OpenAI based on USE_OLLAMA flag in the config.json"""

    # Use Ollama if flag is set and configured
    if config.USE_OLLAMA and ollama_base_url and ollama_model:
        try:
            # Prepare headers with JWT token if available
            headers = {"Content-Type": "application/json"}
            jwt_token = os.getenv("OLLAMA_JWT_TOKEN")
            if jwt_token:
                headers["Authorization"] = f"Bearer {jwt_token}"
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{ollama_base_url}/api/embeddings",
                    json={"model": ollama_model, "prompt": query},
                    headers=headers
                )

                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get('embedding')
                    if embedding and len(embedding) > 0:
                        if config.DEBUG:
                            print(f"Ollama embedding success: {len(embedding)} dimensions")
                        return embedding

        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError) as e:
            if config.DEBUG:
                print(f"Ollama embedding failed: {type(e).__name__} - {e}")
        except Exception as e:
            if config.DEBUG:
                print(f"Ollama unexpected error: {e}")

    # Use OpenAI if flag is set or as fallback
    if not config.USE_OLLAMA and openai_embedder:
        try:
            if config.DEBUG:
                print("Falling back to OpenAI embedding")
            embedding = await asyncio.to_thread(openai_embedder.embed_query, query)
            if embedding and len(embedding) > 0:
                if config.DEBUG:
                    print(f"OpenAI embedding success: {len(embedding)} dimensions")
                return embedding
        except Exception as e:
            if config.DEBUG:
                print(f"OpenAI embedding failed: {e}")

    return []


async def ingest_chunks_into_qdrant(
        json_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        batch_size: int = 32,
        recreate_collection: bool = False
) -> dict:
    """Load chunked manual JSON file and upsert embeddings into Qdrant."""
    print(f"CHUNKS_PATH: {json_path}")
    data_path = DEFAULT_CHUNKS_PATH if not json_path else _resolve_chunks_path(json_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Chunks file not found at {data_path}")

    collection = collection_name or config.COSMIC_DATABASE_COLLECTION_NAME
    batch_size = max(1, batch_size)

    if data_path.is_dir():
        json_files = sorted([p for p in data_path.glob("*.json") if p.is_file()])
    else:
        json_files = [data_path]

    if not json_files:
        return {
            "success": False,
            "file_path": str(data_path),
            "message": "No JSON chunk files found"
        }

    chunks: List[dict] = []
    chunk_sources: List[str] = []
    file_errors: List[dict] = []
    empty_files: List[str] = []

    for file_path in json_files:
        try:
            with file_path.open("r", encoding="utf-8") as fp:
                document = json.load(fp)
            print(f"PROCESSING DOCUMENT: {file_path}")
            
        except json.JSONDecodeError as exc:
            file_errors.append({
                "file": str(file_path),
                "error": f"Failed to parse JSON chunks file: {exc}"
            })
            continue
        except Exception as exc:  # pylint: disable=broad-except
            file_errors.append({
                "file": str(file_path),
                "error": f"Unexpected error reading file: {exc}"
            })
            continue

        file_chunks = document.get("chunks", []) if isinstance(document, dict) else []
        if not file_chunks:
            empty_files.append(str(file_path))
            continue

        chunks.extend(file_chunks)
        chunk_sources.extend([str(file_path)] * len(file_chunks))

    if not chunks:
        message = "No chunks found in the provided directory" if data_path.is_dir() else "No chunks found in the provided file"
        return {
            "success": False,
            "file_path": str(data_path),
            "message": message,
            "load_errors": file_errors,
            "empty_files": empty_files
        }

    openai_embedder = None
    if not config.USE_OLLAMA:
        openai_embedder = OpenAIEmbeddings(
            model=config.EMBEDDINGS_MODEL_NAME,
            openai_api_key=config.OPENAI_API_KEY
        )

    # Prime first embedding to determine vector size and validate access
    first_chunk_index = next((idx for idx, chunk in enumerate(chunks) if (chunk.get("text") or "").strip()), None)
    if first_chunk_index is None:
        raise ValueError("First chunk is missing 'text' content")

    first_chunk = chunks[first_chunk_index]

    first_embedding = await async_embed_with_fallback(
        query=first_chunk["text"],
        ollama_model=config.EMBEDDINGS_MODEL_NAME if config.USE_OLLAMA else None,
        ollama_base_url=config.OLLAMA_BASE_URL if config.USE_OLLAMA else None,
        openai_embedder=openai_embedder,
        timeout=INGEST_EMBED_TIMEOUT
    )

    if not first_embedding:
        raise RuntimeError("Unable to generate embedding for the first chunk")

    vector_size = len(first_embedding)

    # Prepare Qdrant client
    qdrant_url = f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}"
    qdrant_client = AsyncQdrantClient(url=qdrant_url)

    try:
        vector_params = rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE
        )

        if recreate_collection:
            if config.DEBUG:
                print(f"Recreating Qdrant collection '{collection}' with vector size {vector_size}")
            await qdrant_client.recreate_collection(
                collection_name=collection,
                vectors_config=vector_params
            )
        else:
            try:
                await qdrant_client.get_collection(collection_name=collection)
            except Exception:
                if config.DEBUG:
                    print(f"Creating Qdrant collection '{collection}' with vector size {vector_size}")
                await qdrant_client.create_collection(
                    collection_name=collection,
                    vectors_config=vector_params
                )

        points_batch: List[rest.PointStruct] = []
        stored_count = 0
        failed_chunks: List[dict] = []

        async def _embed_chunk_text(text: str) -> List[float]:
            return await async_embed_with_fallback(
                query=text,
                ollama_model=config.EMBEDDINGS_MODEL_NAME if config.USE_OLLAMA else None,
                ollama_base_url=config.OLLAMA_BASE_URL if config.USE_OLLAMA else None,
                openai_embedder=openai_embedder,
                timeout=INGEST_EMBED_TIMEOUT
            )

        def _build_payload(chunk: dict) -> dict:
            metadata = {k: v for k, v in chunk.items() if k != "text"}
            return {"text": chunk.get("text", ""), **metadata}

        def _build_point(chunk: dict, embedding: List[float]) -> rest.PointStruct:
            point_id = chunk.get("text_id") or chunk.get("section_id") or str(uuid4())
            return rest.PointStruct(
                id=point_id,
                vector=embedding,
                payload=_build_payload(chunk)
            )

        for index, chunk in enumerate(chunks):
            chunk["text_id"] = str(uuid4())
            text_content = (chunk.get("text") or "").strip()
            if not text_content:
                failed_chunks.append({
                    "index": index,
                    "reason": "empty text",
                    "chunk_id": chunk.get("text_id"),
                    "source_file": chunk.get("source_file") or chunk_sources[index]
                })
                continue

            if index == first_chunk_index:
                embedding = first_embedding
            else:
                embedding = await _embed_chunk_text(text_content)

            if not embedding:
                failed_chunks.append({
                    "index": index,
                    "reason": "embedding_failed",
                    "chunk_id": chunk.get("text_id"),
                    "source_file": chunk.get("source_file") or chunk_sources[index]
                })
                continue

            points_batch.append(_build_point(chunk, embedding))

            if len(points_batch) >= batch_size:
                await qdrant_client.upsert(
                    collection_name=collection,
                    points=points_batch
                )
                stored_count += len(points_batch)
                points_batch = []

        if points_batch:
            await qdrant_client.upsert(
                collection_name=collection,
                points=points_batch
            )
            stored_count += len(points_batch)

        return {
            "success": True,
            "file_path": str(data_path),
            "collection_name": collection,
            "vector_size": vector_size,
            "total_chunks": len(chunks),
            "stored_chunks": stored_count,
            "failed_chunks": failed_chunks,
            "recreated_collection": recreate_collection,
            "processed_files": [str(path) for path in json_files],
            "load_errors": file_errors,
            "empty_files": empty_files
        }

    finally:
        await qdrant_client.close()


def _resolve_chunks_path(path_value: str) -> Path:
    """Resolve user-supplied path for chunks JSON."""

    candidate = Path(path_value).expanduser()
    if candidate.exists():
        return candidate.resolve()

    module_relative = (Path(__file__).resolve().parent / candidate).resolve()
    if module_relative.exists():
        return module_relative

    return candidate


async def cosmic_database_tool2(
        query: str,
        *,
        limit: Optional[int] = None,
        min_score: Optional[float] = None,
        collection_name: Optional[str] = None
) -> dict:
    """RAG-powered variant that aligns with cosmic_database_tool response format."""

    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return {
            "success": False,
            "query": query,
            "results": [],
            "error": "Query must not be empty"
        }

    result_limit = limit if isinstance(limit, int) and limit > 0 else config.QDRANT_RESULT_LIMIT
    result_limit = max(1, min(result_limit, 20))

    score_threshold = 0.6 if min_score is None else float(min_score)
    score_threshold = max(0.0, min(score_threshold, 1.0))

    collection = (collection_name or config.COSMIC_DATABASE_COLLECTION_NAME).strip()

    if config.DEBUG:
        print(f"[RAG] Searching cosmic database for query: {cleaned_query}")
        print(f"[RAG] Using collection: {collection}")
        print(f"[RAG] Result limit: {result_limit} | Min score: {score_threshold}")

    openai_api_key = None
    if not config.USE_OLLAMA:
        openai_api_key = (getattr(config, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY") or "").strip()
        if not openai_api_key:
            return {
                "success": False,
                "query": cleaned_query,
                "results": [],
                "error": "OpenAI API key is required when using OpenAI provider"
            }

    try:
        retrieval_response: Dict[str, Any] = await asyncio.to_thread(
            ask_question,
            cleaned_query,
            collection=collection,
            limit=result_limit,
            qdrant_host=config.QDRANT_HOST,
            qdrant_port=config.QDRANT_PORT,
            openai_api_key=openai_api_key,
            ollama_model=config.EMBEDDINGS_MODEL_NAME if config.USE_OLLAMA else None,
            ollama_base_url=config.OLLAMA_BASE_URL if config.USE_OLLAMA else None
        )

        # Extract results and reconstructed_context from ask_question response
        retrieval_results: List[HybridResult] = retrieval_response.get("results", [])
        reconstructed_context: str = retrieval_response.get("reconstructed_context", "")

        if config.DEBUG:
            method = retrieval_response.get("reconstruction_method", "unknown")
            chunk_count = retrieval_response.get("chunk_count", 0)
            context_length = len(reconstructed_context) if reconstructed_context else 0
            print(f"[RAG] Retrieved {len(retrieval_results)} results, reconstruction method: {method}, chunks: {chunk_count}")
            print(f"[RAG] Reconstructed context length: {context_length} characters")
            if not reconstructed_context or not reconstructed_context.strip():
                print("[RAG] WARNING: Reconstructed context is empty!")

        filtered_hits = [
            hit for hit in retrieval_results
            if isinstance(hit.score, (int, float)) and hit.score >= score_threshold
        ]

        if config.DEBUG:
            print(f"[RAG] Filtered hits after score threshold ({score_threshold}): {len(filtered_hits)}")

        if not filtered_hits:
            return {
                "success": False,
                "query": cleaned_query,
                "collection_name": collection,
                "min_score": score_threshold,
                "result_limit": result_limit,
                "result_count": 0,
                "results": [],
                "sources": [],
                "message": "No matching content found for the given query"
            }

        formatted_results = []
        sources = set()

        for hit in filtered_hits[:result_limit]:
            payload = hit.payload or {}
            text_content = (payload.get("text") or "").strip()
            metadata = {k: v for k, v in payload.items() if k != "text"}
            metadata.setdefault("id", hit.point_id)

            if text_content:
                formatted_results.append({
                    "score": float(hit.score),
                    "text": text_content,
                    "metadata": metadata
                })

            source_file = metadata.get("source_file")
            if source_file:
                sources.add(source_file)

        if config.DEBUG:
            print(f"[RAG] Generating answer with context length: {len(reconstructed_context)} characters")

        def _generate_rag_answer() -> Optional[str]:
            try:
                answer = generate_answer(
                    cleaned_query, 
                    reconstructed_context,
                    use_ollama=config.USE_OLLAMA,
                    openai_api_key=openai_api_key,
                    ollama_base_url=config.OLLAMA_BASE_URL if config.USE_OLLAMA else None,
                    agent_model_name=config.AGENT_MODEL_NAME if config.USE_OLLAMA else None
                )
                if config.DEBUG:
                    answer_length = len(answer) if answer else 0
                    print(f"[RAG] Generated answer length: {answer_length} characters")
                return answer
            except Exception as exc:  # pylint: disable=broad-except
                if config.DEBUG:
                    print(f"[RAG] Failed to generate answer: {exc}")
                    import traceback
                    print(f"[RAG] Traceback: {traceback.format_exc()}")
                return None

        rag_answer = await asyncio.to_thread(_generate_rag_answer)

        if config.DEBUG:
            if rag_answer:
                print(f"[RAG] Answer generated successfully: {rag_answer[:100]}..." if len(rag_answer) > 100 else f"[RAG] Answer generated successfully: {rag_answer}")
            else:
                print("[RAG] WARNING: No answer was generated (rag_answer is None or empty)")

        return {
            "success": True,
            "query": cleaned_query,
            "collection_name": collection,
            "min_score": score_threshold,
            "result_limit": result_limit,
            "result_count": len(formatted_results),
            "results": formatted_results,
            "sources": sorted(sources),
            "message": rag_answer
        }

    except Exception as e:
        if config.DEBUG:
            print(f"[RAG] Error searching cosmic database: {e}")
            print(f"[RAG] Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "query": cleaned_query,
            "results": [],
            "error": f"Error searching cosmic database with RAG: {str(e)}"
        }