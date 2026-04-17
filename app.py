"""
app.py — Llama2 Hybrid Research Assistant
==========================================
Streamlit conversational UI that combines:
  - RAG (ChromaDB + sentence-transformers) for document retrieval
  - Fine-tuned Llama2 7B (via Ollama) for answer generation

Course: Leveraging Llama2 for Advanced AI Solutions
Author: Sylvain Corney | Nokia Canada
"""

import streamlit as st
from rag_pipeline import build_vector_store, retrieve_context
from llama_model import load_model

# ── Page configuration ────────────────────────────────────────────────────────
# Must be the first Streamlit call in the script.
st.set_page_config(
    page_title="Llama2 Research Assistant",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
# Injects styling for a polished, research-tool aesthetic:
# dark navy header bar, card-style chat bubbles, branded accent colour.
st.markdown("""
<style>
/* ── Global font & background ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0f1117;
    color: #e8eaf0;
}

/* ── Header banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2f4a 60%, #0d2137 100%);
    border-bottom: 2px solid #f97316;
    padding: 1.4rem 2rem 1.2rem 2rem;
    border-radius: 0 0 12px 12px;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #f8fafc;
    letter-spacing: -0.5px;
    margin: 0;
}
.hero-subtitle {
    font-size: 0.88rem;
    color: #94a3b8;
    margin-top: 0.3rem;
    letter-spacing: 0.5px;
}
.hero-badge {
    display: inline-block;
    background: #f97316;
    color: #fff;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    margin-top: 0.5rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Chat message formatting ── */
/* Ensure line breaks in AI responses render as paragraphs */
[data-testid="stChatMessageContent"] p {
    margin-bottom: 0.75rem;
    line-height: 1.7;
    font-size: 0.95rem;
}

/* ── Sidebar styling ── */
[data-testid="stSidebar"] {
    background-color: #0d1b2a;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] h3 {
    color: #f97316;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
/* Force all sidebar text to be readable on dark background */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] strong {
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] em {
    color: #94a3b8 !important;
}

/* ── Chat input box ── */
[data-testid="stChatInput"] textarea {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
}

/* ── Spinner text ── */
[data-testid="stSpinner"] p {
    color: #f97316 !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
}

/* ── Clear button ── */
.stButton > button {
    background-color: #1e3a5f;
    color: #94a3b8;
    border: 1px solid #334155;
    border-radius: 6px;
    font-size: 0.8rem;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    background-color: #f97316;
    color: #fff;
    border-color: #f97316;
}
</style>
""", unsafe_allow_html=True)

# ── Avatar with hover-to-edit ─────────────────────────────────────────
# Shows the avatar normally; reveals the file uploader only on hover.
# Uses an HTML/CSS overlay trick combined with a Streamlit expander.

if "avatar_image" in st.session_state:
        # Display the current avatar with a small edit hint below it
        st.markdown("<div style='text-align:center; padding-top:1rem;'>",
                    unsafe_allow_html=True)
        st.image(st.session_state["avatar_image"], width=100)
        st.markdown("""
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.9rem;
                        color:#f8fafc; font-weight:600; margin-top:0.5rem;">
                LLaMA Research Bot
            </div>
            <div style="font-size:0.72rem; color:#475569; margin-top:0.2rem;">
                AI Research Assistant
            </div>
        </div>""", unsafe_allow_html=True)

        # Small collapsed expander acts as the "edit" affordance
        with st.expander("✏️ Change avatar", expanded=False):
            uploaded_avatar = st.file_uploader(
                "Upload new image",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed"
            )
            if uploaded_avatar:
                st.session_state["avatar_image"] = uploaded_avatar
                st.rerun()

else:
        # No avatar set yet — show default emoji + prominent uploader
        st.markdown("""
        <div style='text-align:center; padding-top:1rem;'>
            <div style='font-size:5rem; line-height:1;'>🤖</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.9rem;
                        color:#f8fafc; font-weight:600; margin-top:0.5rem;">
                LLaMA Research Bot
            </div>
            <div style="font-size:0.72rem; color:#64748b; margin-top:0.2rem;">
                AI Research Assistant
            </div>
        </div>""", unsafe_allow_html=True)

        uploaded_avatar = st.file_uploader(
            "🖼️ Set assistant avatar",
            type=["png", "jpg", "jpeg"],
            help="Upload an image to personalise the assistant"
        )
        if uploaded_avatar:
            st.session_state["avatar_image"] = uploaded_avatar
            st.rerun()

    
st.markdown("<hr style='border-color:#1e3a5f; margin: 0.8rem 0;'>",
                unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
# Provides context about the assistant and a conversation reset button.
with st.sidebar:
    # AI researcher profile picture using a large emoji avatar
    # Replace the emoji block below with st.image("avatar.png") if you have
    # a custom image file in your project directory.
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.5rem 0;">
        <div style="font-size: 5rem; line-height:1;">🤖</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.9rem;
                    color:#f8fafc; font-weight:600; margin-top:0.5rem;">
            LLaMA Research Bot
        </div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:0.2rem;">
            AI Research Assistant
        </div>
    </div>
    <hr style="border-color:#1e3a5f; margin: 0.8rem 0;">
    """, unsafe_allow_html=True)

    st.markdown("### About")
    st.markdown("""
    This assistant uses a **hybrid LLM architecture**:

    - 📄 **RAG** retrieves relevant passages from the Llama2 research paper  
    - 🧠 **Llama2 7B** (fine-tuned, via Ollama) synthesizes the final answer  
    - 🔒 **Fully local** — no data leaves your machine
    """)

    st.markdown("### Knowledge Base")
    st.markdown("""
    - Llama2: Open Foundation and Fine-Tuned Chat Models *(arXiv:2307.09288)*
    """)

    st.markdown("### Example Questions")
    # Suggested queries help first-time users understand what to ask
    example_queries = [
        "What training data was used for Llama2?",
        "How does Llama2 handle safety?",
        "Explain RLHF in Llama2",
        "What is Ghost Attention?",
        "How large is the Llama2 70B model?",
    ]
    for q in example_queries:
        st.markdown(f"- *{q}*")

    st.markdown("---")

    # Clear conversation button — resets session state and reruns the app
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div style="font-size:0.7rem; color:#334155; text-align:center; margin-top:1rem;">
        Sylvain Corney · Nokia Canada · 2026
    </div>
    """, unsafe_allow_html=True)

# ── Resource initialisation ───────────────────────────────────────────────────
# @st.cache_resource ensures build_vector_store() and load_model() are called
# only once per session, even across Streamlit reruns triggered by user input.
@st.cache_resource(show_spinner="⚙️ Loading knowledge base and model — please wait...")
def init():
    """
    Initialise the RAG vector store and the Ollama model callable.
    Returns:
        vectorstore : ChromaDB collection loaded with Llama2 paper chunks
        pipe        : callable(prompt) → str  (wraps Ollama REST API)
    """
    vs = build_vector_store()
    pipe = load_model()
    return vs, pipe

vectorstore, pipe = init()

# ── Chat history initialisation ───────────────────────────────────────────────
# st.session_state persists data across Streamlit reruns within the same session.
# Each message is stored as {"role": "user"|"assistant", "content": str}.
if "messages" not in st.session_state:
    st.session_state.messages = []

    # Display a welcome message on first load
    welcome = (
        "Hello! I'm your Llama2 research assistant. "
        "I have access to the full Llama2 research paper and can answer "
        "detailed questions about its architecture, training, safety design, "
        "and benchmarks.\n\n"
        "What would you like to know?"
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome})

# ── Render existing conversation history ─────────────────────────────────────
# Loop through stored messages and render each in the appropriate chat bubble.
# The avatar parameter sets a custom icon for the assistant role.
for msg in st.session_state.messages:
    avatar = "🦙" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        # st.markdown renders newlines as paragraph breaks, preserving formatting
        st.markdown(msg["content"])

# ── Handle new user input ─────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about Llama2 or large language models..."):

    # 1. Store and display the user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Generate the assistant response
    with st.chat_message("assistant", avatar="🦙"):
        with st.spinner("🔍 Searching paper · 🧠 Generating response..."):

            # Step A: Retrieve the top-4 most relevant chunks from ChromaDB
            # using semantic similarity between the query embedding and stored vectors
            context = retrieve_context(vectorstore, prompt, k=2)

            # Step B: Build the hybrid prompt — combines retrieved paper context
            # with an open-ended instruction so Llama2 can synthesise both sources
            full_prompt = f"""You are an expert AI research assistant specialising in large language models, with deep knowledge of the Llama2 paper by Meta AI.

Use the context below from the Llama2 research paper to ground your answer, then expand with your broader knowledge of LLMs where relevant.

--- CONTEXT FROM LLAMA2 PAPER ---
{context}
--- END CONTEXT ---

Question: {prompt}

Provide a clear, well-structured answer. Use paragraphs to separate distinct ideas. Be specific and cite details from the paper where possible."""

            # Step C: Send the prompt to the locally-running Llama2 model via Ollama
            answer = pipe(full_prompt)

            # Step D: Render the response — st.markdown handles paragraph breaks
            # (double newlines in the answer are rendered as <p> tags)
            st.markdown(answer)

    # 3. Persist the assistant response to session history for future reruns
    st.session_state.messages.append({"role": "assistant", "content": answer})