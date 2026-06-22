from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# --------------------------
# Paths
# --------------------------
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

files = ["guvi_courses.txt", "guvi_blogs.txt", "guvi_faqs.txt"]

# --------------------------
# Step 1 — Combine all text files
# --------------------------
print("🔄 Combining GUVI text files...")
texts = ""
for file in files:
    with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
        texts += f.read() + "\n"

# --------------------------
# Step 2 — Chunk the text
# --------------------------
print("✂️ Splitting text into chunks...")

chunks = []

for file in files:
    with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
        content = f.read()

        # Split by blank lines
        file_chunks = [c.strip() for c in content.split("\n\n") if c.strip()]

        chunks.extend(file_chunks)

print(f"✅ Total Chunks: {len(chunks)}")

with open(os.path.join(data_dir, "cleaned_chunks.txt"), "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(c.strip() + "\n---\n")

print(f"✅ Total Chunks: {len(chunks)}")

# --------------------------
# Step 3 — Create FAISS embeddings
# --------------------------
print("⚙️ Generating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
embeddings = model.encode(chunks, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")

# --------------------------
# Step 4 — Build FAISS index
# --------------------------
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, os.path.join(data_dir, "faiss_index.bin"))

# --------------------------
# Step 5 — Save chunk list
# --------------------------
with open(os.path.join(data_dir, "chunks_list.txt"), "w", encoding="utf-8") as f:
    f.write("\n---\n".join(chunks))

print("🎯 FAISS index and chunks rebuilt successfully!")
