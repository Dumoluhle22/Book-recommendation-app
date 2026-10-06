import chromadb
import pandas as pd
import ast
import os
import shutil

# 1. Config
DB_PATH = "./my_books_db"
INPUT_CSV = "books_with_embeddings.csv"
COLLECTION_NAME = "african_literature"
STREAM_BATCH_SIZE = 2000  # rows read from CSV and uploaded per chunk

# 2. Clean start: delete old database to avoid stale/duplicate entries
if os.path.exists(DB_PATH):
    print("Removing old database folder...")
    shutil.rmtree(DB_PATH)

# 3. Initialize ChromaDB
print("Initializing ChromaDB...")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.create_collection(name=COLLECTION_NAME)

# 4. Check file exists
if not os.path.exists(INPUT_CSV):
    raise SystemExit(
        f"ERROR: '{INPUT_CSV}' not found. Run main.py first to generate embeddings."
    )

# 5. Stream the CSV in chunks instead of loading it all into memory at once
print("Loading and uploading books from CSV in streams...")

row_counter = 0  # used to generate unique IDs across chunks

for chunk in pd.read_csv(INPUT_CSV, chunksize=STREAM_BATCH_SIZE):
    # Drop rows with missing title, author, desc, or embedding —
    # Chroma rejects NaN documents, and a few rows can end up incomplete
    # if main.py was interrupted/resumed partway through.
    before = len(chunk)
    chunk = chunk.dropna(subset=['title', 'author', 'desc', 'embedding'])
    skipped = before - len(chunk)
    if skipped:
        print(f"  Skipped {skipped} row(s) with missing data in this chunk.")
    if len(chunk) == 0:
        continue

    # Convert the embedding column from string back to actual lists
    current_embeddings = [ast.literal_eval(x) for x in chunk['embedding'].tolist()]

    # Generate sequential IDs (your CSV has no 'id' column)
    current_ids = [str(i) for i in range(row_counter, row_counter + len(chunk))]
    row_counter += len(chunk)

    current_documents = chunk['desc'].tolist()

    # Metadata: only include columns that actually exist
    metadata_cols = ['title', 'author']
    if 'genre' in chunk.columns:
        metadata_cols.append('genre')
    current_metadatas = chunk[metadata_cols].to_dict(orient='records')

    collection.add(
        ids=current_ids,
        embeddings=current_embeddings,
        metadatas=current_metadatas,
        documents=current_documents,
    )

    print(f"  Inserted a batch of {len(chunk)} books ({row_counter} total so far).")

print(f"\nSuccess! Total books now in database: {collection.count()}")