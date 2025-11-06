import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, os
from openai import OpenAI  # ✅ New client import

# -----------------------------
# Configuration
# -----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # ✅ Correct API setup

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


# Load FAISS index + text chunks
index = faiss.read_index("data/faiss_index.bin")
with open("data/chunks_list.txt", "r", encoding="utf-8") as f:
    data = f.read().split("---")
chunks = [c.strip() for c in data if c.strip()]

# Retrieve similar chunks
def retrieve_relevant_chunks(query, top_k=3):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector).astype('float32'), top_k)
    return [chunks[i] for i in indices[0]]

# Generate answer via NEW OpenAI client
def generate_answer(query):
    retrieved = retrieve_relevant_chunks(query)
    context = "\n".join(retrieved)
    prompt = f"Answer the user's question using the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = client.chat.completions.create(   # ✅ New syntax (not openai.ChatCompletion)
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.4,
    )
    return response.choices[0].message.content, retrieved

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="GUVI Knowledge Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 GUVI Knowledge Retrieval Chatbot")

user_query = st.text_input("Ask anything about GUVI courses, certifications, or learning:")

if st.button("Get Answer") and user_query.strip():
    with st.spinner("Thinking... 💭"):
        answer, refs = generate_answer(user_query)
    st.success("✅ Answer:")
    st.write(answer)

    with st.expander("📚 View Retrieved Context"):
        for r in refs:
            st.markdown(f"- {r}")
