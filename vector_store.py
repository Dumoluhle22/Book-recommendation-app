import chromadb
import pandas as pd
import ast
import os
import shutil

# 1. Config
DB_PATH = "./my_books_db"
INPUT_CSV = "books_with_embeddings.csv"
COLLECTION_NAME = "african_literature"

# 2. Clean start: delete old database to avoid stale/duplicate entries
if os.path.exists(DB_PATH):
    print("Removing old database folder...")
    shutil.rmtree(DB_PATH)

# 3. Initialize ChromaDB
print("Initializing ChromaDB...")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.create_collection(name=COLLECTION_NAME)

# 4. Load embedded data
if not os.path.exists(INPUT_CSV):
    raise SystemExit(
        f"ERROR: '{INPUT_CSV}' not found. Run main.py first to generate embeddings."
    )

df = pd.read_csv(INPUT_CSV)
print(f"Loading {len(df)} books from CSV...")

# 5. Build the lists Chroma needs
ids = []
embeddings = []
metadatas = []
documents = []

for i, row in df.iterrows():
    ids.append(str(i))
    embeddings.append(ast.literal_eval(row['embedding']))

    # Only include metadata fields that actually exist in the CSV,
    # so we don't fabricate values (like the old hardcoded trope).
    metadata = {
        "title": row['title'],
        "author": row['author'],
    }
    if 'genre' in df.columns and pd.notna(row.get('genre')):
        metadata["genre"] = row['genre']

    metadatas.append(metadata)
    documents.append(row['desc'])

# 6. Upload to ChromaDB in batches (Chroma has a max batch size per call)
BATCH_SIZE = 500
print("Uploading to ChromaDB...")
for start in range(0, len(ids), BATCH_SIZE):
    end = start + BATCH_SIZE
    collection.add(
        ids=ids[start:end],
        embeddings=embeddings[start:end],
        metadatas=metadatas[start:end],
        documents=documents[start:end],
    )
    print(f"  Uploaded {min(end, len(ids))}/{len(ids)}")

print(f"\nSuccess! {collection.count()} books are now in the database.")