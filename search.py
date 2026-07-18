import pandas as pd
import numpy as np
import faiss
import ast
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Load the embedded data
print("Loading book data...")
df = pd.read_csv('books_with_embeddings_test.csv')

# 2. THE GOTCHA FIX: Convert stringified lists back to actual Python lists, then to NumPy arrays
df['embedding'] = df['embedding'].apply(ast.literal_eval)
embeddings = np.array(df['embedding'].tolist()).astype('float32')

# 3. Build the FAISS Vector Index
dimension = embeddings.shape[1] # This is usually 1536 for OpenAI's small model
index = faiss.IndexFlatL2(dimension) # L2 measures the exact distance between vectors
index.add(embeddings)
print(f"Vector engine spun up with {index.ntotal} books ready to search!\n")

# 4. The Query Function
def search_books(query, top_k=2):
    print(f"Searching for: '{query}'...")
    
    # Step A: Convert the user's plain text search into a vector using the exact same model
    response = client.embeddings.create(input=query, model="text-embedding-3-small")
    query_vector = np.array([response.data[0].embedding]).astype('float32')
    
    # Step B: Ask FAISS to find the closest matches
    distances, indices = index.search(query_vector, top_k)
    
    # Step C: Display the results
    print("\n--- TOP MATCHES ---")
    for i, idx in enumerate(indices[0]):
        book_title = df.iloc[idx]['title']
        book_desc = df.iloc[idx]['desc']
        # The lower the distance score, the closer the match
        print(f"{i+1}. {book_title} (Distance: {distances[0][i]:.4f})")
        print(f"   Snippet: {book_desc[:100]}...\n")

# 5. Let's test it!
# Even with only 5 books, the AI will try to find the semantic closest match.
search_books("A story about war and conflict")