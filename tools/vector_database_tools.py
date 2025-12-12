import sys
import json
import openai
import os
import asyncio
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from uuid import uuid4
import logging

# Add parent directory to path to import root config BEFORE other imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
# Also add tools directory to path for importing ask_qdrant and rag
tools_dir = Path(__file__).parent
sys.path.insert(0, str(tools_dir))

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
from langchain_openai import OpenAIEmbeddings
import traceback
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

from ask_qdrant import ask_question, HybridResult
from rag import generate_answer
import httpx
import requests
import config

load_dotenv(find_dotenv())

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

logger = logging.getLogger(__name__)

# Set appropriate log levels for noisy libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('qdrant_client').setLevel(logging.WARNING)

logger.info("=" * 80)
logger.info("VECTOR DATABASE TOOLS INITIALIZED")
logger.info("=" * 80)
logger.info(f"USE_OLLAMA: {config.USE_OLLAMA}")
logger.info(f"EMBEDDINGS_MODEL: {config.EMBEDDINGS_MODEL_NAME}")
logger.info(f"QDRANT_HOST: {config.QDRANT_HOST}:{config.QDRANT_PORT}")
logger.info(f"COLLECTION: {config.COSMIC_DATABASE_COLLECTION_NAME}")
logger.info("=" * 80)

# Inject truststore for SSL certificate handling (self-signed certificates) - only if using Ollama
if config.USE_OLLAMA:
    try:
        import truststore

        truststore.inject_into_ssl()
        logger.info("Truststore injected for SSL handling (Ollama)")
    except ImportError:
        logger.warning("Truststore not installed, skipping SSL injection")
        # truststore not installed, skip SSL injection
        pass

DEFAULT_CHUNKS_FILENAME = "chunks_Cosmic_manual_block_handling_v10_0.pdf.json"
DEFAULT_CHUNKS_PATH = (Path(__file__).resolve().parent / DEFAULT_CHUNKS_FILENAME).resolve()
INGEST_EMBED_TIMEOUT = 30.0

logger.info(f"Default chunks path: {DEFAULT_CHUNKS_PATH}")


def embed_query_using_ollama_embedding_model(query: str, model_name: str, ollama_url: str):
    """Generate embedding using Nomic model via Ollama"""
    logger.debug(f"Embedding query with Ollama - Model: {model_name}, URL: {ollama_url}")
    logger.debug(f"Query: {query[:100]}...")

    payload = {
        "model": model_name,
        "prompt": query
    }

    # Prepare headers with JWT token if available
    headers = {"Content-Type": "application/json"}
    jwt_token = os.getenv("OLLAMA_JWT_TOKEN")
    if jwt_token:
        headers["Authorization"] = f"Bearer {jwt_token}"
        logger.debug("JWT token added to request headers")

    try:
        logger.debug(f"Sending request to {ollama_url}/api/embeddings")
        response = requests.post(
            f"{ollama_url}/api/embeddings",
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        embedding = result["embedding"]
        logger.info(f"Ollama embedding generated successfully - Dimensions: {len(embedding)}")
        return embedding

    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama request failed: {e}", exc_info=True)
        raise
    except KeyError as e:
        logger.error(f"Unexpected response format: {e}")
        logger.error(f"Response: {response.text}")
        raise
    except Exception as e:
        logger.error(f"Ollama embedding failed: {e}", exc_info=True)
        raise


async def async_embed_with_fallback(
        query: str,
        ollama_model: str = None,
        ollama_base_url: str = None,
        openai_embedder: Optional[OpenAIEmbeddings] = None,
        timeout: float = 10.0
) -> List[float]:
    """Async embedding with Ollama or OpenAI based on USE_OLLAMA flag in the config.json"""

    logger.debug(f"Async embedding - Query length: {len(query)}, Timeout: {timeout}s")
    logger.debug(f"USE_OLLAMA: {config.USE_OLLAMA}")

    # Use Ollama if flag is set and configured
    if config.USE_OLLAMA and ollama_base_url and ollama_model:
        logger.info(f"Attempting Ollama embedding with model: {ollama_model}")
        try:
            # Prepare headers with JWT token if available
            headers = {"Content-Type": "application/json"}
            jwt_token = os.getenv("OLLAMA_JWT_TOKEN")
            if jwt_token:
                headers["Authorization"] = f"Bearer {jwt_token}"
                logger.debug("JWT token added to headers")

            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.debug(f"Sending async request to {ollama_base_url}/api/embeddings")
                response = await client.post(
                    f"{ollama_base_url}/api/embeddings",
                    json={"model": ollama_model, "prompt": query},
                    headers=headers
                )

                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get('embedding')
                    if embedding and len(embedding) > 0:
                        logger.info(f"Ollama embedding success: {len(embedding)} dimensions")
                        return embedding
                else:
                    logger.warning(f"Ollama returned status code: {response.status_code}")

        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError) as e:
            logger.warning(f"Ollama embedding failed: {type(e).__name__} - {e}")
        except Exception as e:
            logger.error(f"Ollama unexpected error: {e}", exc_info=True)

    # Use OpenAI if flag is set or as fallback
    if not config.USE_OLLAMA and openai_embedder:
        logger.info("Using OpenAI embedding")
        try:
            embedding = await asyncio.to_thread(openai_embedder.embed_query, query)
            if embedding and len(embedding) > 0:
                logger.info(f"OpenAI embedding success: {len(embedding)} dimensions")
                return embedding
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}", exc_info=True)

    logger.error("All embedding methods failed - returning empty list")
    return []


async def ingest_chunks_into_qdrant(
        json_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        batch_size: int = 32,
        recreate_collection: bool = False,
        progress_callback=None
) -> dict:
    """Load chunked manual JSON file and upsert embeddings into Qdrant."""
    logger.info("=" * 80)
    logger.info("INGESTING CHUNKS INTO QDRANT")
    logger.info("=" * 80)

    data_path = DEFAULT_CHUNKS_PATH if not json_path else _resolve_chunks_path(json_path)
    logger.info(f"Chunks path: {data_path}")

    if not data_path.exists():
        logger.error(f"Chunks file not found at {data_path}")
        raise FileNotFoundError(f"Chunks file not found at {data_path}")

    collection = collection_name or config.COSMIC_DATABASE_COLLECTION_NAME
    batch_size = max(1, batch_size)

    logger.info(f"Collection: {collection}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Recreate collection: {recreate_collection}")

    if data_path.is_dir():
        json_files = sorted([p for p in data_path.glob("*.json") if p.is_file()])
        logger.info(f"Processing directory with {len(json_files)} JSON files")
    else:
        json_files = [data_path]
        logger.info("Processing single JSON file")

    if not json_files:
        logger.error("No JSON chunk files found")
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
        logger.info(f"Processing file: {file_path}")
        try:
            with file_path.open("r", encoding="utf-8") as fp:
                document = json.load(fp)
            logger.debug(f"Successfully loaded JSON from {file_path}")

        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse JSON from {file_path}: {exc}")
            file_errors.append({
                "file": str(file_path),
                "error": f"Failed to parse JSON chunks file: {exc}"
            })
            continue
        except Exception as exc:
            logger.error(f"Unexpected error reading {file_path}: {exc}", exc_info=True)
            file_errors.append({
                "file": str(file_path),
                "error": f"Unexpected error reading file: {exc}"
            })
            continue

        file_chunks = document.get("chunks", []) if isinstance(document, dict) else []
        if not file_chunks:
            logger.warning(f"No chunks found in {file_path}")
            empty_files.append(str(file_path))
            continue

        logger.info(f"Loaded {len(file_chunks)} chunks from {file_path}")
        chunks.extend(file_chunks)
        chunk_sources.extend([str(file_path)] * len(file_chunks))

    logger.info(f"Total chunks loaded: {len(chunks)}")

    if not chunks:
        message = "No chunks found in the provided directory" if data_path.is_dir() else "No chunks found in the provided file"
        logger.error(message)
        return {
            "success": False,
            "file_path": str(data_path),
            "message": message,
            "load_errors": file_errors,
            "empty_files": empty_files
        }

    openai_embedder = None
    if not config.USE_OLLAMA:
        logger.info("Initializing OpenAI embedder")
        openai_embedder = OpenAIEmbeddings(
            model=config.EMBEDDINGS_MODEL_NAME,
            openai_api_key=config.OPENAI_API_KEY
        )

    # Prime first embedding to determine vector size and validate access
    logger.info("Generating first embedding to determine vector size")
    first_chunk_index = next((idx for idx, chunk in enumerate(chunks) if (chunk.get("text") or "").strip()), None)
    if first_chunk_index is None:
        logger.error("First chunk is missing 'text' content")
        raise ValueError("First chunk is missing 'text' content")

    first_chunk = chunks[first_chunk_index]
    logger.debug(f"First chunk index: {first_chunk_index}")
    logger.debug(f"First chunk text preview: {first_chunk['text'][:100]}...")

    first_embedding = await async_embed_with_fallback(
        query=first_chunk["text"],
        ollama_model=config.EMBEDDINGS_MODEL_NAME if config.USE_OLLAMA else None,
        ollama_base_url=config.OLLAMA_BASE_URL if config.USE_OLLAMA else None,
        openai_embedder=openai_embedder,
        timeout=INGEST_EMBED_TIMEOUT
    )

    if not first_embedding:
        logger.error("Unable to generate embedding for the first chunk")
        raise RuntimeError("Unable to generate embedding for the first chunk")

    vector_size = len(first_embedding)
    logger.info(f"Vector size determined: {vector_size} dimensions")

    # Prepare Qdrant client
    qdrant_url = f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}"
    logger.info(f"Connecting to Qdrant at {qdrant_url}")
    qdrant_client = AsyncQdrantClient(url=qdrant_url)

    try:
        vector_params = rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE
        )

        if recreate_collection:
            logger.warning(f"Recreating Qdrant collection '{collection}' with vector size {vector_size}")
            await qdrant_client.recreate_collection(
                collection_name=collection,
                vectors_config=vector_params
            )
            logger.info(f"Collection '{collection}' recreated successfully")
        else:
            try:
                await qdrant_client.get_collection(collection_name=collection)
                logger.info(f"Using existing collection '{collection}'")
            except Exception:
                logger.info(f"Creating new Qdrant collection '{collection}' with vector size {vector_size}")
                await qdrant_client.create_collection(
                    collection_name=collection,
                    vectors_config=vector_params
                )
                logger.info(f"Collection '{collection}' created successfully")

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

        total_chunks = len(chunks)
        logger.info(f"Starting ingestion of {total_chunks} chunks")

        for index, chunk in enumerate(chunks):
            if (index + 1) % 10 == 0 or index == 0:
                logger.info(f"Processing chunk {index + 1}/{total_chunks}")

            chunk["text_id"] = str(uuid4())
            text_content = (chunk.get("text") or "").strip()

            if not text_content:
                logger.warning(f"Chunk {index} has empty text, skipping")
                failed_chunks.append({
                    "index": index,
                    "reason": "empty text",
                    "chunk_id": chunk.get("text_id"),
                    "source_file": chunk.get("source_file") or chunk_sources[index]
                })
                if progress_callback:
                    progress_callback(index + 1, total_chunks, stored_count, len(failed_chunks))
                continue

            if index == first_chunk_index:
                embedding = first_embedding
                logger.debug(f"Using pre-generated embedding for chunk {index}")
            else:
                embedding = await _embed_chunk_text(text_content)

            if not embedding:
                logger.error(f"Failed to generate embedding for chunk {index}")
                failed_chunks.append({
                    "index": index,
                    "reason": "embedding_failed",
                    "chunk_id": chunk.get("text_id"),
                    "source_file": chunk.get("source_file") or chunk_sources[index]
                })
                if progress_callback:
                    progress_callback(index + 1, total_chunks, stored_count, len(failed_chunks))
                continue

            points_batch.append(_build_point(chunk, embedding))

            if len(points_batch) >= batch_size:
                logger.debug(f"Upserting batch of {len(points_batch)} points")
                await qdrant_client.upsert(
                    collection_name=collection,
                    points=points_batch
                )
                stored_count += len(points_batch)
                logger.info(f"Upserted {stored_count}/{total_chunks} chunks")
                points_batch = []

            if progress_callback:
                progress_callback(index + 1, total_chunks, stored_count, len(failed_chunks))

        if points_batch:
            logger.debug(f"Upserting final batch of {len(points_batch)} points")
            await qdrant_client.upsert(
                collection_name=collection,
                points=points_batch
            )
            stored_count += len(points_batch)
            logger.info(f"Final upsert complete. Total stored: {stored_count}/{total_chunks}")
            if progress_callback:
                progress_callback(total_chunks, total_chunks, stored_count, len(failed_chunks))

        logger.info("=" * 80)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total chunks: {len(chunks)}")
        logger.info(f"Stored chunks: {stored_count}")
        logger.info(f"Failed chunks: {len(failed_chunks)}")
        logger.info("=" * 80)

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
        logger.debug("Qdrant client closed")


def _resolve_chunks_path(path_value: str) -> Path:
    """Resolve user-supplied path for chunks JSON."""
    logger.debug(f"Resolving chunks path: {path_value}")

    candidate = Path(path_value).expanduser()
    if candidate.exists():
        logger.debug(f"Path exists: {candidate.resolve()}")
        return candidate.resolve()

    module_relative = (Path(__file__).resolve().parent / candidate).resolve()
    if module_relative.exists():
        logger.debug(f"Module-relative path exists: {module_relative}")
        return module_relative

    logger.warning(f"Path not found, returning candidate: {candidate}")
    return candidate


async def cosmic_database_tool2(
        query: str,
        *,
        limit: Optional[int] = None,
        min_score: Optional[float] = None,
        collection_name: Optional[str] = None
) -> dict:
    """RAG-powered variant that aligns with cosmic_database_tool response format."""

    logger.info("=" * 80)
    logger.info("COSMIC DATABASE TOOL 2 - RAG SEARCH")
    logger.info("=" * 80)

    cleaned_query = (query or "").strip()
    if not cleaned_query:
        logger.warning("Empty query received")
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

    logger.info(f"Query: {cleaned_query}")
    logger.info(f"Collection: {collection}")
    logger.info(f"Result limit: {result_limit}")
    logger.info(f"Min score threshold: {score_threshold}")

    openai_api_key = None
    if not config.USE_OLLAMA:
        openai_api_key = (getattr(config, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY") or "").strip()
        if not openai_api_key:
            logger.error("OpenAI API key is required but not found")
            return {
                "success": False,
                "query": cleaned_query,
                "results": [],
                "error": "OpenAI API key is required when using OpenAI provider"
            }
        logger.debug("OpenAI API key found")

    try:
        logger.info("Calling ask_question for retrieval")
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

        method = retrieval_response.get("reconstruction_method", "unknown")
        chunk_count = retrieval_response.get("chunk_count", 0)
        context_length = len(reconstructed_context) if reconstructed_context else 0

        logger.info("=" * 80)
        logger.info("RETRIEVAL RESULTS")
        logger.info("=" * 80)
        logger.info(f"Retrieved results: {len(retrieval_results)}")
        logger.info(f"Reconstruction method: {method}")
        logger.info(f"Chunks in context: {chunk_count}")
        logger.info(f"Context length: {context_length} characters")
        logger.info("=" * 80)

        # Log each retrieved result in detail
        logger.info("DETAILED RETRIEVAL HITS:")
        logger.info("=" * 80)
        for idx, hit in enumerate(retrieval_results, 1):
            logger.info(f"HIT #{idx}")
            logger.info(f"  Score: {hit.score}")
            logger.info(f"  Point ID: {hit.point_id}")
            if hit.payload:
                text_preview = hit.payload.get("text", "")[:200]
                logger.info(f"  Text preview: {text_preview}...")
                logger.info(f"  Metadata keys: {[k for k in hit.payload.keys() if k != 'text']}")
                for key, value in hit.payload.items():
                    if key != "text":
                        logger.info(f"    {key}: {value}")
            logger.info("-" * 80)

        if not reconstructed_context or not reconstructed_context.strip():
            logger.warning("Reconstructed context is EMPTY!")
        else:
            logger.info(f"Reconstructed context preview (first 500 chars):")
            logger.info(reconstructed_context[:500])
            logger.info("=" * 80)

        filtered_hits = [
            hit for hit in retrieval_results
            if isinstance(hit.score, (int, float)) and hit.score >= score_threshold
        ]

        logger.info(f"Filtered hits (score >= {score_threshold}): {len(filtered_hits)}")

        if not filtered_hits:
            logger.warning("No matching content found after filtering by score threshold")
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

        logger.info("=" * 80)
        logger.info("FORMATTED RESULTS (AFTER FILTERING)")
        logger.info("=" * 80)

        for idx, hit in enumerate(filtered_hits[:result_limit], 1):
            payload = hit.payload or {}
            text_content = (payload.get("text") or "").strip()
            metadata = {k: v for k, v in payload.items() if k != "text"}
            metadata.setdefault("id", hit.point_id)

            if text_content:
                formatted_result = {
                    "score": float(hit.score),
                    "text": text_content,
                    "metadata": metadata
                }
                formatted_results.append(formatted_result)

                logger.info(f"RESULT #{idx}")
                logger.info(f"  Score: {hit.score}")
                logger.info(f"  Text length: {len(text_content)} chars")
                logger.info(f"  Text preview: {text_content[:200]}...")
                logger.info(f"  Metadata: {metadata}")
                logger.info("-" * 80)

            source_file = metadata.get("source_file")
            if source_file:
                sources.add(source_file)

        logger.info("=" * 80)
        logger.info(f"Total formatted results: {len(formatted_results)}")
        logger.info(f"Unique sources: {len(sources)}")
        if sources:
            logger.info(f"Sources: {sorted(sources)}")
        logger.info("=" * 80)

        logger.info("Generating RAG answer")
        logger.info(f"Context length for generation: {len(reconstructed_context)} characters")

        def _generate_rag_answer() -> Optional[str]:
            try:
                logger.debug("Calling generate_answer function")
                answer = generate_answer(
                    cleaned_query,
                    reconstructed_context,
                    use_ollama=config.USE_OLLAMA,
                    openai_api_key=openai_api_key,
                    ollama_base_url=config.OLLAMA_BASE_URL if config.USE_OLLAMA else None,
                    agent_model_name=config.AGENT_MODEL_NAME if config.USE_OLLAMA else None
                )
                answer_length = len(answer) if answer else 0
                logger.info(f"Answer generated - Length: {answer_length} characters")
                if answer:
                    logger.info(f"Answer preview: {answer[:200]}...")
                return answer
            except Exception as exc:
                logger.error(f"Failed to generate answer: {exc}", exc_info=True)
                return None

        rag_answer = await asyncio.to_thread(_generate_rag_answer)

        logger.info("=" * 80)
        if rag_answer:
            logger.info("RAG ANSWER GENERATED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Answer length: {len(rag_answer)} characters")
            logger.info("Full answer:")
            logger.info(rag_answer)
            logger.info("=" * 80)
        else:
            logger.error("RAG ANSWER GENERATION FAILED - Answer is None or empty")
            logger.info("=" * 80)

        result = {
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

        logger.info("FINAL RESULT SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Success: {result['success']}")
        logger.info(f"Query: {result['query']}")
        logger.info(f"Result count: {result['result_count']}")
        logger.info(f"Sources count: {len(result['sources'])}")
        logger.info(f"Has answer: {bool(result['message'])}")
        logger.info("=" * 80)

        return result

    except Exception as e:
        logger.error("=" * 80)
        logger.error("ERROR IN COSMIC DATABASE TOOL")
        logger.error("=" * 80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Traceback:", exc_info=True)
        logger.error("=" * 80)

        return {
            "success": False,
            "query": cleaned_query,
            "results": [],
            "error": f"Error searching cosmic database with RAG: {str(e)}"
        }