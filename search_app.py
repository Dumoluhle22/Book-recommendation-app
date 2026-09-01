import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./my_books_db")
collection = chroma_client.get_collection(name="african_literature")


def query_books(user_prompt, n_results=3):
    # 1. Turn the user's prompt into an embedding
    response = client.embeddings.create(
        input=user_prompt, 
        model="text-embedding-3-small"
    )
    query_vector = response.data[0].embedding
    
    # 2. Let Chroma find the closest matches
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )
    
    # 3. Print the results
    for i in range(len(results['ids'][0])):
        print(f"Match: {results['metadatas'][0][i]['title']}")
        print(f"Snippet: {results['documents'][0][i][:100]}...\n")

query_books("A story about historical conflict and identity")