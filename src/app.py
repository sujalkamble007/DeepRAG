# src/app.py
"""
DeepRAG — NLP-Powered Deep-Sea Governance Document Intelligence Assistant
Uses Retrieval-Augmented Generation (RAG) to analyze and interpret complex policy,
research, and environmental documents related to deep-sea governance.
"""

import os
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import datetime
import textwrap

# ─────────────────────────────────────────────────────────────
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
api_key = os.getenv("OPENAI_API_KEY")

if not api_key or not isinstance(api_key, str) or api_key.startswith("sk-REPLACE"):
    st.warning("⚠️ Missing or invalid API key. Please add it to your .env file as OPENAI_API_KEY.")
    st.stop()

# Set environment for OpenAI
os.environ["OPENAI_API_KEY"] = api_key

# Initialize OpenAI client
try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error("OpenAI client initialization failed. Check your API key and network.")
    st.stop()

# ─── Caching embedding model ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    model.max_seq_length = 256
    return model

model = load_embedding_model()

# ─── Helper functions ─────────────────────────────────────────
def chunk_text(txt, size=300):
    words = txt.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def estimate_tokens(text):
    return int(len(text.split()) / 0.75)

def build_faiss_index(embeds: np.ndarray):
    dimension = embeds.shape[1]
    idx = faiss.IndexFlatL2(dimension)
    # Ensure embeddings are float32 and properly shaped
    embeddings_f32 = embeds.astype(np.float32)
    idx.add(embeddings_f32)  # type: ignore
    return idx

def semantic_retrieve(query, index_obj, chunks_list, top_k=3):
    q_vec = model.encode([query]).astype("float32")
    # Ensure q_vec is 2D array for FAISS
    if len(q_vec.shape) == 1:
        q_vec = q_vec.reshape(1, -1)
    distances, indices = index_obj.search(q_vec, min(top_k, len(chunks_list)))
    return [(i, chunks_list[i]) for i in indices[0] if i < len(chunks_list)]

# ─── Page config & Header ─────────────────────────────────────
st.set_page_config(page_title="DeepRAG", layout="wide")

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:16px;">
      <div style="width:64px; height:64px; border-radius:12px; background:linear-gradient(135deg,#0ea5e9,#1e3a8a); display:flex; align-items:center; justify-content:center;">
        <svg width="34" height="34" viewBox="0 0 24 24" fill="white"><path d="M12 2L15 8H9L12 2Z"/><circle cx="12" cy="14" r="6" fill="white" opacity="0.12"/></svg>
      </div>
      <div>
        <h1 style="margin:0; font-size:1.8rem;">DeepRAG</h1>
        <div style="color:gray; font-size:1rem;">NLP Technique for Efficient Deep-Sea Governance Document Analysis</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Top-k retrieved chunks", 1, 8, 3)
    temperature = st.slider("Model temperature", 0.0, 1.0, 0.3, 0.05)
    model_choice = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"], index=2)

    st.markdown("---")
    if st.button("🧹 Clear chat & index"):
        for k in ["chat_history", "chunks", "embeddings", "index", "raw_text"]:
            st.session_state.pop(k, None)
        st.success("Session cleared. Re-upload your document.")
        st.stop()

    st.caption("DeepRAG applies NLP-driven RAG to enhance analysis of marine policy, research, and governance texts.")

# ─── Upload & Parse PDF ───────────────────────────────────────
uploaded = st.file_uploader("📄 Upload a Deep-Sea Governance PDF (policy, research, report)", type="pdf")
if not uploaded:
    st.info("Upload a PDF to begin your analysis.")
    st.stop()

if "raw_text" not in st.session_state:
    try:
        reader = PdfReader(uploaded)
        raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        st.session_state["raw_text"] = raw_text
        st.session_state["pdf_pages"] = len(reader.pages)
    except Exception as e:
        st.error(f"PDF parsing failed: {e}")
        st.stop()

raw_text = st.session_state["raw_text"]
if not raw_text.strip():
    st.warning("No extractable text found in PDF.")
    st.stop()

st.success("✅ Document parsed successfully.")

# ─── Document Overview ───────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Pages", st.session_state.get("pdf_pages", "-"))
col2.metric("Words", len(raw_text.split()))
col3.metric("Tokens (approx.)", estimate_tokens(raw_text))

# ─── Build Embedding Index ────────────────────────────────────
if "chunks" not in st.session_state:
    with st.spinner("Building RAG knowledge base..."):
        chunks = chunk_text(raw_text, 300)
        embeddings = model.encode(chunks).astype("float32")
        st.session_state["chunks"] = chunks
        st.session_state["embeddings"] = embeddings
        st.session_state["index"] = build_faiss_index(embeddings)
st.success("Knowledge base ready for semantic retrieval.")

if st.checkbox("🔍 Show sample chunks"):
    for i, c in enumerate(st.session_state["chunks"][:3]):
        st.markdown(f"**Chunk {i+1}**")
        st.write(textwrap.shorten(c, width=400, placeholder="..."))

# ─── NLP Query Section ───────────────────────────────────────
st.markdown("## 💬 Ask about Deep-Sea Governance insights")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{
        "role": "system",
        "content": (
            "You are DeepRAG — an NLP-powered assistant specialized in analyzing "
            "Deep-Sea Governance, environmental regulations, marine policy documents, "
            "and sustainability reports. Provide insightful, concise, and domain-relevant "
            "answers with references from the given context."
        ),
        # Use timezone-aware UTC timestamp to avoid deprecation of utcnow()
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }]

with st.form("query_form", clear_on_submit=False):
    query = st.text_input("Enter your question related to the document:")
    submit = st.form_submit_button("Ask DeepRAG")

if submit and query:
    index_obj = st.session_state["index"]
    chunks = st.session_state["chunks"]

    with st.spinner("Retrieving and Generating answer..."):
        retrieved = semantic_retrieve(query, index_obj, chunks, top_k)
        context = "\n\n".join([f"[Chunk {i}] {chunk}" for i, chunk in retrieved])
        system_prompt = (
            f"Analyze the question using deep-sea governance and environmental NLP reasoning.\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer with clear explanations, highlighting governance frameworks or research implications."
        )

        st.session_state.chat_history.append({"role": "user", "content": query})

        try:
            # Create messages list with proper typing
            messages = st.session_state.chat_history + [{"role": "user", "content": system_prompt}]
            response = client.chat.completions.create(
                model=model_choice,
                messages=messages,  # type: ignore
                temperature=float(temperature),
            )
            answer = response.choices[0].message.content or "No response generated"
        except Exception as e:
            answer = f"Model request failed: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        st.markdown("### 🧠 DeepRAG Answer")
        st.info(answer)

        st.markdown("### 📚 Retrieved Context (for transparency)")
        for i, chunk in retrieved:
            with st.expander(f"Chunk {i} — Preview"):
                st.write(chunk)

# ─── Chat History ─────────────────────────────────────────────
st.markdown("---")
st.subheader("🗂️ Conversation History")
for msg in st.session_state.get("chat_history", []):
    role = msg["role"]
    if role == "assistant":
        st.markdown("**DeepRAG:**")
        st.info(msg["content"])
    elif role == "user":
        st.markdown("**You:**")
        st.write(msg["content"])

# ─── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.caption("DeepRAG — NLP Technique for Efficient Deep-Sea Governance Document Analysis • Built with RAG and Streamlit.")