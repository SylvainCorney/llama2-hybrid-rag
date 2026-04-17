"""
llama_model.py — Llama2 Model Interface via Ollama
====================================================
Provides a callable interface to the locally-running aiResearcher fine-tuned model
served by Ollama's REST API.

Why Ollama instead of loading weights directly through transformers?
  - No GPU required: Ollama manages quantization (4-bit GGUF) internally
  - No HuggingFace authentication: model is already pulled locally
  - Shared model: the same Ollama instance can serve other tools on the machine
  - Simpler dependency graph: no torch/accelerate/bitsandbytes version conflicts

Prerequisites:
  - Ollama installed and running:  ollama serve
  - Llama2 model pulled:          ollama pull llama2:7b  # or: ollama create aiResearcher -f Modelfile

Course: Leveraging Llama2 for Advanced AI Solutions
Author: Sylvain Corney | Nokia Canada
"""

import requests

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"   # Default Ollama local server address
DEFAULT_MODEL   = "llama2:7b"               # Ollama model tag (must match `ollama list`)
TIMEOUT_SECONDS = 120                        # Max wait time for a single inference call


def load_model(model_name: str = DEFAULT_MODEL):
    """
    Return a callable that sends prompts to the Ollama-served Llama2 model.

    This factory function follows the same interface pattern as a HuggingFace
    pipeline — it returns a callable rather than a model object, so it integrates
    cleanly with Streamlit's @st.cache_resource decorator.

    Args:
        model_name: The Ollama model tag to use (e.g. "aiResearcher:latest").
                    Must match exactly what `ollama list` shows.

    Returns:
        generate(prompt: str) -> str
            A function that accepts a prompt string and returns the model's
            response as a plain string.

    Example:
        pipe = load_model()
        answer = pipe("Explain RLHF in simple terms.")
    """

    def generate(prompt: str) -> str:
        """
        Send a prompt to the Ollama REST API and return the generated text.

        Ollama's /api/generate endpoint accepts a JSON payload and returns a
        JSON response. With stream=False, the full response is returned in one
        request rather than as a token stream.

        Args:
            prompt: The complete prompt string, including any retrieved context.

        Returns:
            The model's response text as a plain string.

        Raises:
            requests.exceptions.ConnectionError: if Ollama is not running.
            KeyError: if the model name does not match a pulled model.
        """
        response = requests.post(
            url=f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model_name,        # Which model to use
                "prompt": prompt,           # The full prompt with RAG context
                "stream": False,            # Return complete response, not a stream
            },
            timeout=TIMEOUT_SECONDS,
        )

        # Raise an HTTP error if Ollama returned a non-200 status code
        response.raise_for_status()

        # The response JSON contains a "response" key with the generated text.
        # If the model name is wrong, Ollama returns {"error": "model not found"}
        # which will cause a KeyError here — check `ollama list` if this occurs.
        return response.json()["response"]

    return generate