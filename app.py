import streamlit as st
import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv

# --- Functions ---

def expand_query(user_input):
    prompt = f"Rewrite this book search query to be descriptive and identify specific tropes or themes: '{user_input}'"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    # Fixed the typo here: changed 'contenty' to 'content'
    return response.choices[0].message.content

def get_reranked_results(user_query, results):
    books_text = ""
    # Safely iterate through the results
    for i in range(len(results['metadatas'][0])):
        title = results['metadatas'][0][i]['title']
        desc = results['documents'][0][i]
        books_text += f"{i+1}. {title}: {desc}\n"

    prompt = f"""
    The user is looking for books matching this request: '{user_query}'
    
    Here are the retrieved books from our database:
    {books_text}
    
    Please rank these books based on how well they fit the user's request. 
    Return ONLY the top 3 book numbers, and a very brief reason why they fit.
    Format: "1. [Title]: [Reason]"
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- Setup ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./my_books_db")
collection = chroma_client.get_or_create_collection(name="african_literature")

# --- UI ---
st.title("📚 Bartech African Lit Recommender")
user_query = st.text_input("What kind of story are you looking for?")

if st.button("Search"):
    if user_query:
        with st.spinner("Refining your request..."):
            refined_query = expand_query(user_query)
            st.info(f"Searching for: *{refined_query}*")
        
        response = client.embeddings.create(
            input=refined_query, 
            model="text-embedding-3-small"
        )
        query_vector = response.data[0].embedding
        
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=10
        )
        
        st.subheader("AI-Curated Recommendations:")
        curated_list = get_reranked_results(refined_query, results)
        st.write(curated_list)
        
