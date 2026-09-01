import streamlit as st
import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv

DB_PATH = "./my_books_db"
COLLECTION_NAME = "african_literature"

# --- Functions ---

def expand_query(user_input):
    prompt = (
        f"Rewrite this book search query to be descriptive and identify "
        f"specific tropes or themes: '{user_input}'"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def get_reranked_results(user_query, results):
    books_text = ""
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
api_key = os.getenv("OPENAI_API_KEY")

st.title("📚 Bartech African Lit Recommender")

if not api_key:
    st.error("OPENAI_API_KEY not found. Add it to your .env file and restart the app.")
    st.stop()

if not os.path.exists(DB_PATH):
    st.error(
        "No book database found. Run main.py then vector_store.py first "
        "to build the search index."
    )
    st.stop()

client = OpenAI(api_key=api_key)
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# --- UI ---
user_query = st.text_input("What kind of story are you looking for?")

if st.button("Search"):
    if not user_query.strip():
        st.warning("Type something to search for first.")
    else:
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

        if not results['ids'][0]:
            st.warning("No matches found. Try rephrasing your search.")
        else:
            st.subheader("AI-Curated Recommendations:")
            with st.spinner("Ranking the best matches..."):
                curated_list = get_reranked_results(refined_query, results)
            st.write(curated_list)