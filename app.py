import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import torch

# -----------------------------
# Setup (Torch + Hugging Face)
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    generator = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
    return embedder, generator

model, generator = load_models()

# Load FAISS index + chunks
index = faiss.read_index("data/faiss_index.bin")
with open("data/chunks_list.txt", "r", encoding="utf-8") as f:
    data = f.read().split("---")
chunks = [c.strip() for c in data if c.strip()]

def retrieve_chunks(query, top_k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec).astype('float32'), top_k)
    return [chunks[i] for i in indices[0]]

def generate_answer(query):
    retrieved = retrieve_chunks(query)
    context = "\n".join(retrieved)
    prompt = f"Use the context below to answer:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    output = generator(prompt, max_length=256)
    return output[0]['generated_text'], retrieved

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="GUVI Knowledge Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 GUVI Knowledge Retrieval Chatbot")

query = st.text_input("Ask about GUVI courses, certifications, or learning:")

if st.button("Get Answer") and query.strip():
    with st.spinner("Thinking..."):
        ans, refs = generate_answer(query)
    st.success("✅ Answer:")
    st.write(ans)

    with st.expander("📚 Retrieved Context"):
        for r in refs:
            st.markdown(f"- {r}")
