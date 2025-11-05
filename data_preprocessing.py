import os

# folder path
data_folder = "data"
cleaned_texts = []

# Step 1: Read all .txt files
for file_name in os.listdir(data_folder):
    if file_name.endswith(".txt"):
        with open(os.path.join(data_folder, file_name), "r", encoding="utf-8") as f:
            text = f.read()
            # Step 2: Basic cleaning
            text = text.replace("\n", " ").replace("  ", " ")
            cleaned_texts.append(text)

# Step 3: Combine all text into one string
full_text = " ".join(cleaned_texts)

# Step 4: Split into small chunks (every 500 characters)
chunk_size = 500
chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

# Step 5: Save chunks into a new file
os.makedirs("data", exist_ok=True)
with open("data/cleaned_chunks.txt", "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(c.strip() + "\n---\n")

print(f"✅ Data cleaning done! {len(chunks)} chunks created and saved in data/cleaned_chunks.txt")
