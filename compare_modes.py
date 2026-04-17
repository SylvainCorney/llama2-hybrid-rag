"""
Run this script from your llama2-rag-env to generate 3-mode comparison results.
It will print RAG-only, LLM-only, and Hybrid responses for 3 test queries.

Usage:
    python compare_modes.py
"""

import requests
import sys

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama2:7b"

TEST_QUERIES = [
    "What training data was used for Llama2?",
    "How does Llama2 handle safety and alignment?",
    "Explain the RLHF process used in Llama2 and its impact on helpfulness vs safety tradeoffs."
]

def query_ollama(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    })
    return response.json().get("response", "ERROR")

def get_rag_context(vectorstore, query, k=4):
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])

def mode_llm_only(query):
    """No context, just the LLM answering from its training knowledge."""
    prompt = f"""You are an AI research assistant. Answer the following question about Llama2 from your knowledge:

Question: {query}

Answer:"""
    return query_ollama(prompt)

def mode_rag_only(context, query):
    """Context provided but with a generic base model prompt (no fine-tuning framing)."""
    prompt = f"""Answer the following question using ONLY the context provided below. 
Do not use any outside knowledge.

Context:
{context}

Question: {query}

Answer based strictly on context:"""
    return query_ollama(prompt)

def mode_hybrid(context, query):
    """Full hybrid: RAG context + LLM knowledge synthesis."""
    prompt = f"""You are an AI research assistant specializing in large language models.

Context from the Llama2 research paper:
{context}

Question: {query}

Provide a detailed, accurate answer combining the context above with your knowledge:"""
    return query_ollama(prompt)

if __name__ == "__main__":
    print("Loading vector store...")
    try:
        import sys
        sys.path.insert(0, ".")
        from rag_pipeline import build_vector_store, retrieve_context
        vectorstore = build_vector_store()
        print("Vector store ready.\n")
    except Exception as e:
        print(f"Could not load vector store: {e}")
        print("Running LLM-only mode only.\n")
        vectorstore = None

    for i, query in enumerate(TEST_QUERIES, 1):
        print("=" * 70)
        print(f"QUERY {i}: {query}")
        print("=" * 70)

        context = ""
        if vectorstore:
            context = retrieve_context(vectorstore, query)

        print("\n--- MODE 1: LLM-Only (no RAG context) ---")
        r1 = mode_llm_only(query)
        print(r1[:600], "..." if len(r1) > 600 else "")

        if vectorstore:
            print("\n--- MODE 2: RAG-Only (context, no LLM knowledge framing) ---")
            r2 = mode_rag_only(context, query)
            print(r2[:600], "..." if len(r2) > 600 else "")

            print("\n--- MODE 3: Hybrid (RAG + LLM synthesis) ---")
            r3 = mode_hybrid(context, query)
            print(r3[:600], "..." if len(r3) > 600 else "")

        print()