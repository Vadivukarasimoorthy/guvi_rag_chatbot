from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import os   # ✅ ADD THIS LINE

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Load FAISS index and chunks
index = faiss.read_index("data/faiss_index.bin")
with open("data/chunks_list.txt", "r", encoding="utf-8") as f:
    data = f.read().split("---")
chunks = [c.strip() for c in data if c.strip()]

# Step 2: Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_relevant_chunks(query, top_k=3):
    query_vector = model.encode([query])

    distances, indices = index.search(
        np.array(query_vector).astype('float32'),
        top_k
    )

    print("\nQuery:", query)
    print("\nRetrieved Chunks:")

    results = []

    for idx in indices[0]:
        print(chunks[idx])
        print("-" * 50)
        results.append(chunks[idx])

    return results

def generate_answer(query):
    """Generate final answer using retrieved chunks + LLM"""
    retrieved = retrieve_relevant_chunks(query)
    context = "\n".join(retrieved)
    
    prompt = f"Use the below GUVI data to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.4,
    )
    return response["choices"][0]["message"]["content"]

# Example Test
if _name_ == "_main_":
    user_q = input("Ask a question: ")
    ans = generate_answer(user_q)
    print("\n💬 Chatbot Answer:\n", ans)