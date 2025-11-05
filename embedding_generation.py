from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1 – Load cleaned chunks
with open("data/cleaned_chunks.txt", "r", encoding="utf-8") as f:
    data = f.read().split("---")
chunks = [c.strip() for c in data if c.strip()]

print(f"📄 {len(chunks)} text chunks loaded")

# Step 2 – Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3 – Create embeddings
embeddings = model.encode(chunks, show_progress_bar=True)

# Step 4 – Convert to FAISS index
d = embeddings.shape[1]  # vector dimension
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings).astype('float32'))

# Step 5 – Save index and chunks
faiss.write_index(index, "data/faiss_index.bin")
with open("data/chunks_list.txt", "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(c + "\n---\n")

print("✅ Embedding generation complete!")
print(f"Stored {len(chunks)} vectors in data/faiss_index.bin")
