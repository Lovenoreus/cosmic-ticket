#!/usr/bin/env python3
"""Interactive RAG helper with IMPROVED prompting and temperature."""

import sys
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path to import root config
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from ask_qdrant import ask_question, HybridResult
import config

# Inject truststore for SSL certificate handling (self-signed certificates) - only if using Ollama
if config.USE_OLLAMA:
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        # truststore not installed, skip SSL injection
        pass


DEFAULT_MODEL = "gpt-4o-mini"

# OPTIMIZED SYSTEM PROMPT for semantic LLM-based chunking
SYSTEM_PROMPT = """You are an expert assistant answering questions about healthcare documentation and procedures in Swedish.

You will receive reconstructed document context containing multiple chunks from the source document. These chunks are semantically organized and arranged in sequence. The chunk marked "[MOST RELEVANT]" best matches the user's question.

ANSWER STRATEGY:

1. **Locate the core answer**: Start with the [MOST RELEVANT] chunk - this likely contains the primary answer.

2. **Extract key information**: Identify the essential facts, definitions, or steps that directly answer the question.

3. **Check surrounding chunks**: Look at neighboring chunks ONLY if they add necessary context or missing details.

4. **Be concise**: Provide only the information needed to answer the question completely. Avoid including tangential details from surrounding chunks.

5. **Format appropriately**:
   - For "what/why/when" questions: Provide a clear, direct paragraph
   - For "how" or procedure questions: Use numbered steps
   - For definitions: Give the definition, then one example if available

6. **Stay accurate**: Never add information not present in the chunks. If something is stated as an example in the chunks, present it as an example.

WHEN TO SAY "I'M NOT SURE":
Only if:
- The reconstructed context is empty or contains no text
- The chunks are completely unrelated to the question (wrong document entirely)
- The specific detail is genuinely absent from all provided chunks

CONCISENESS RULES:
- If the [MOST RELEVANT] chunk fully answers the question, use primarily that chunk
- Don't repeat information already stated
- Don't add procedural steps not explicitly mentioned
- Don't expand examples beyond what's in the source
- Focus on what the user asked, not everything the chunks contain

EXAMPLE RESPONSES:

**Question:** "Vad är syftet med X?"
**Good answer:** "Syftet är att ge en överblick över patientgrupper som är anslutna till en enhet men inte inskrivna i slutenvård, och att underlätta planering samt förbättra lagefterlevnad."
**Bad answer:** "Syftet är att ge en funktion som möjliggör en överblick över patienter... Denna översikt är filtrerbar på olika kriterier... [adds filtering details, team info, etc.]"

**Question:** "Hur öppnas fönstret X?"
**Good answer:** "Fönstret öppnas via huvudmenyn eller med kortkommandot Ctrl + Skift + A."
**Bad answer:** "För att öppna fönstret följer du dessa steg: 1. Gå till huvudmenyn... När fönstret är öppet kan du filtrera... [adds filtering capabilities not asked about]"

Your goal: Provide accurate, complete, and CONCISE answers that directly address what the user asked.
""".strip()


def generate_answer(
    query: str,
    reconstructed_context: str,
    *,
    use_ollama: bool = False,
    openai_api_key: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    agent_model_name: Optional[str] = None,
    model: str | None = None,
    temperature: float = 0.7,  # FIXED: Increased from 0.3 to 0.7
) -> str:
    """Generate answer with improved prompting using Ollama or OpenAI."""
    
    # FIXED: Better handling of empty context
    if not reconstructed_context or reconstructed_context.strip() == "":
        return "I'm not sure based on the provided documentation."

    user_prompt = f"""User Question:
{query}

Reconstructed Document Context:
{reconstructed_context}

Provide a helpful answer based on the context above.
""".strip()

    # Use Ollama if flag is set
    if use_ollama:
        ollama_base_url = ollama_base_url or config.OLLAMA_BASE_URL
        agent_model_name = agent_model_name or config.AGENT_MODEL_NAME
        
        if not ollama_base_url or not agent_model_name:
            raise ValueError("Ollama base URL and model name must be provided when using Ollama")
        
        # Prepare headers with JWT token if available
        ollama_headers = {}
        jwt_token = os.getenv("OLLAMA_JWT_TOKEN")
        if jwt_token:
            ollama_headers["Authorization"] = f"Bearer {jwt_token}"
        
        ollama_client = ChatOllama(
            model=agent_model_name,
            base_url=ollama_base_url,
            temperature=temperature,
            headers=ollama_headers if ollama_headers else None
        )
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        response = ollama_client.invoke(messages)
        return response.content.strip()
    
    # Use OpenAI
    else:
        api_key = openai_api_key or config.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key must be provided when using OpenAI")
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model or DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


def main() -> None:
    load_dotenv()
    client = OpenAI()

    print("RAG assistant ready. Press Ctrl+C or submit empty query to exit.")

    try:
        while True:
            query = input("Query> ").strip()
            if not query:
                break

            start = time.perf_counter()

            # Retrieve from Qdrant
            try:
                response = ask_question(query)
            except Exception as exc:
                print(f"Error during retrieval: {exc}", file=sys.stderr)
                continue

            if not response:
                print("No records found.")
                continue

            results = response.get("results", [])
            reconstructed_context = response.get("reconstructed_context", "")
            method = response.get("reconstruction_method", "unknown")

            if not results:
                print("No records found.")
                continue

            # Generate answer using reconstructed doc piece
            try:
                answer = generate_answer(
                    query, 
                    reconstructed_context,
                    use_ollama=config.USE_OLLAMA
                )
            except Exception as exc:
                print(f"Error generating answer: {exc}", file=sys.stderr)
                continue

            if not answer:
                print("No answer generated.")
                continue

            print("\n--- Answer ---")
            print(answer)
            print("--------------\n")

            # Print hybrid search results
            print(f"Top retrieved chunks (method: {method}):\n")
            for idx, result in enumerate(results, start=1):
                payload = result.payload or {}
                source = payload.get("source_file") or "unknown"
                text = (payload.get("text") or "").strip()
                print(f"[{idx}] Source: {source} | Score: {result.score:.3f} | ID: {result.point_id}")
                print(text[:200] + ("..." if len(text) > 200 else ""))
                print("-" * 80)

            # Show reconstructed context preview
            print("\nReconstructed context used:\n")
            snippet = reconstructed_context[:1200]
            print(snippet + ("..." if len(reconstructed_context) > 1200 else ""))
            print("------------------------------------------------------------\n")

            elapsed = time.perf_counter() - start
            print(f"⏱️ Response generated in {elapsed:.2f}s\n")

    except (KeyboardInterrupt, EOFError):
        print()
        return


if __name__ == "__main__":
    main()