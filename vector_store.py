import chromadb
import pandas as pd
import ast
import os
import shutil

# 1. Clean start: Delete old database to avoid conflicts
db_path = "./my_books_db"
if os.path.exists(db_path):
    print("Removing old database folder...")
    shutil.rmtree(db_path)

# 2. Initialize
print("Initializing ChromaDB...")
client = chromadb.PersistentClient(path=db_path)
collection = client.create_collection(name="african_literature")

# 3. Load data
file_path = 'books_with_embeddings_test.csv'
df = pd.read_csv(file_path)
print(f"Loading {len(df)} books from CSV...")

# 4. Add to DB
ids = []
embeddings = []
metadatas = []
documents = []

for i, row in df.iterrows():
    ids.append(str(i))
    embeddings.append(ast.literal_eval(row['embedding']))
    metadatas.append({"title": row['title'], "author": row['author'], "genre": row['genre'], "trope": "enemies to lovers"})
    documents.append(row['desc'])

print("Uploading to ChromaDB...")
collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

print(f"Success! {collection.count()} books are now in the database.")