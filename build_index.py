from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load cleaned text chunks
with open("data/cleaned_chunks.txt", "r", encoding="utf-8") as f:
    chunks = [c.strip() for c in f.read().split("---") if c.strip()]

# Encode chunks
embeddings = model.encode(chunks)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype("float32"))

# Save index and chunks list
faiss.write_index(index, "data/faiss_index.bin")
with open("data/chunks_list.txt", "w", encoding="utf-8") as f:
    f.write("\n---\n".join(chunks))

print("✅ FAISS index rebuilt successfully. Chunks:", len(chunks))
