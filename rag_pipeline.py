"""
rag_pipeline.py — Retrieval-Augmented Generation Pipeline
==========================================================
Handles all document ingestion and retrieval logic:
  1. Loads the Llama2 research paper PDF using LangChain's PyPDFLoader
  2. Splits it into overlapping text chunks
  3. Embeds each chunk using a local sentence-transformer model
  4. Persists the vectors to a ChromaDB collection on disk
  5. Exposes a retrieve_context() function for semantic similarity search

The vector store is built once and cached to ./chroma_db — subsequent
application starts skip re-embedding and load directly from disk.

Course: Leveraging Llama2 for Advanced AI Solutions
Author: Sylvain Corney | Nokia Canada
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ── Configuration constants ───────────────────────────────────────────────────
PDF_PATH = "llama2_paper.pdf"          # Path to the Llama2 research paper PDF

# Local path to the downloaded sentence-transformer embedding model.
# Using a local path avoids SSL certificate issues when pulling from HuggingFace.
EMBEDDING_MODEL_PATH = "./models/all-MiniLM-L6-v2"

CHROMA_PERSIST_DIR = "./chroma_db"    # Directory where ChromaDB stores vectors on disk

CHUNK_SIZE = 500      # Max characters per text chunk fed to the embedding model
CHUNK_OVERLAP = 50    # Characters shared between consecutive chunks to preserve context
TOP_K = 4             # Number of most-relevant chunks to retrieve per query


def build_vector_store(pdf_path: str = PDF_PATH) -> Chroma:
    """
    Load the PDF, split into chunks, embed them, and persist to ChromaDB.

    On first run this processes the entire PDF (~77 pages) and saves the
    resulting vector store to disk. On subsequent runs ChromaDB loads from
    the persisted directory, so this function remains fast.

    Args:
        pdf_path: Filesystem path to the Llama2 research paper PDF.

    Returns:
        A ChromaDB vector store object ready for similarity search.
    """

    # ── Step 1: Load PDF ──────────────────────────────────────────────────────
    # PyPDFLoader reads the PDF page by page, preserving page number metadata.
    # Each page becomes a LangChain Document with .page_content and .metadata.
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # ── Step 2: Split into chunks ─────────────────────────────────────────────
    # RecursiveCharacterTextSplitter attempts to split on paragraph boundaries
    # first, then sentence boundaries, then characters — preserving semantic
    # coherence better than a fixed-size splitter.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    # ── Step 3: Load embedding model ─────────────────────────────────────────
    # all-MiniLM-L6-v2 produces 384-dimensional dense vectors.
    # It runs entirely on CPU, making it compatible with any development machine.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": "cpu"},   # Explicit CPU — change to "mps" on Apple Silicon with PyTorch MPS
        encode_kwargs={"normalize_embeddings": True},  # Cosine similarity requires normalised vectors
    )

    # ── Step 4: Create and persist ChromaDB vector store ─────────────────────
    # Chroma.from_documents() embeds every chunk and stores (vector, text, metadata)
    # tuples in the persist directory. If the directory already exists with data,
    # use Chroma(persist_directory=..., embedding_function=...) to load instead.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    return vectorstore


def retrieve_context(vectorstore: Chroma, query: str, k: int = TOP_K) -> str:
    """
    Perform a semantic similarity search and return the top-k chunks as text.

    The query is embedded using the same model as the documents, then cosine
    similarity is computed against all stored vectors. The top-k most similar
    chunks are returned concatenated as a single string for prompt injection.

    Args:
        vectorstore : The ChromaDB collection to search against.
        query       : The user's natural language question.
        k           : Number of chunks to retrieve (default TOP_K = 4).

    Returns:
        A single string containing the top-k retrieved passages,
        separated by double newlines.
    """
    docs = vectorstore.similarity_search(query, k=k)

    # Concatenate chunk text, separated by double newlines for readability
    # in the prompt context block
    return "\n\n".join([doc.page_content for doc in docs])