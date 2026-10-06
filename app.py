import streamlit as st
import chromadb
import requests
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

DB_PATH = "./my_books_db"
COLLECTION_NAME = "african_literature"

st.set_page_config(
    page_title="Novella",
    page_icon="📖",
    layout="centered",
)

# --- Custom styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Quicksand:wght@400;500;600;700&display=swap');

:root {
    --espresso: #3e2723;
    --petal: #f4c9d6;
    --ivory: #f5f0e6;
}

.stApp {
    background-color: var(--ivory);
    background-image: radial-gradient(var(--petal) 1.6px, transparent 1.6px);
    background-size: 22px 22px;
    color: var(--espresso);
    font-family: 'Quicksand', sans-serif;
}

.main .block-container {
    max-width: 700px;
    padding-top: 3rem;
}

h1 {
    font-family: 'Fredoka', sans-serif !important;
    font-weight: 700 !important;
    color: var(--espresso) !important;
    font-size: 2.8rem !important;
    letter-spacing: -0.01em;
    margin-bottom: 0.1rem !important;
}

.subtitle {
    color: var(--espresso);
    opacity: 0.65;
    font-weight: 500;
    font-size: 0.95rem;
    margin-bottom: 2.2rem;
}

div[data-testid="stTextInput"] input {
    background-color: var(--ivory) !important;
    color: var(--espresso) !important;
    border: 2px solid var(--petal) !important;
    border-radius: 14px !important;
    padding: 0.7rem 1rem !important;
    font-size: 1rem !important;
    font-family: 'Quicksand', sans-serif !important;
    font-weight: 500 !important;
}
div[data-testid="stTextInput"] input:focus {
    border: 2px solid var(--espresso) !important;
    box-shadow: 0 0 0 1px var(--espresso) !important;
}
div[data-testid="stTextInput"] label {
    color: var(--espresso) !important;
    opacity: 0.75;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

div.stButton > button {
    background-color: var(--espresso) !important;
    color: var(--petal) !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Fredoka', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.8rem !important;
    margin-top: 0.6rem;
    transition: opacity 0.15s ease;
}
div.stButton > button:hover {
    opacity: 0.85;
}

div[data-testid="stAlert"] {
    background-color: #ffffffaa !important;
    border: 2px solid var(--petal) !important;
    border-radius: 14px !important;
    color: var(--espresso) !important;
}

.results-heading {
    font-family: 'Fredoka', sans-serif;
    font-weight: 600;
    font-size: 1.4rem;
    color: var(--espresso);
    margin-top: 2.2rem;
    margin-bottom: 1rem;
}

.book-card {
    background-color: #ffffffcc;
    border: 2px solid var(--petal);
    border-radius: 16px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
}
.book-card .book-title {
    font-family: 'Fredoka', sans-serif;
    font-weight: 600;
    font-size: 1.15rem;
    color: var(--espresso);
    margin-bottom: 0.3rem;
}
.book-card .book-reason {
    font-family: 'Quicksand', sans-serif;
    font-weight: 500;
    font-size: 0.92rem;
    color: var(--espresso);
    opacity: 0.8;
    line-height: 1.5;
    margin-bottom: 0.5rem;
}
.book-card .book-link a {
    color: var(--espresso);
    background-color: var(--petal);
    border-radius: 10px;
    padding: 0.25rem 0.7rem;
    font-size: 0.82rem;
    font-weight: 700;
    text-decoration: none;
}
.book-card .book-link a:hover {
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)


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

    Pick the top 3 books that best fit the user's request.

    Respond with ONLY valid JSON, no other text, in exactly this format:
    {{
      "recommendations": [
        {{"title": "Book Title", "reason": "brief reason why it fits"}},
        {{"title": "Book Title", "reason": "brief reason why it fits"}},
        {{"title": "Book Title", "reason": "brief reason why it fits"}}
      ]
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(response.choices[0].message.content)
        return parsed.get("recommendations", [])
    except (json.JSONDecodeError, AttributeError):
        return []


def find_book_link(title, author=""):
    """Find a page for the book using Google Books or Open Library."""

    try:
        query = f"intitle:{title}"
        if author:
            query += f"+inauthor:{author}"

        resp = requests.get(
            "https://www.googleapis.com/books/v1/volumes",
            params={
                "q": query,
                "maxResults": 1
            },
            timeout=5
        )

        data = resp.json()
        items = data.get("items")

        if items:
            info = items[0].get("volumeInfo", {})

            link = (
                info.get("previewLink")
                or info.get("infoLink")
                or info.get("canonicalVolumeLink")
            )

            if link:
                return link

    except requests.RequestException:
        pass

    try:
        resp = requests.get(
            "https://openlibrary.org/search.json",
            params={
                "title": title,
                "author": author,
                "limit": 1
            },
            timeout=5
        )

        data = resp.json()
        docs = data.get("docs")

        if docs:
            key = docs[0].get("key")

            if key:
                return f"https://openlibrary.org{key}"

    except requests.RequestException:
        pass

    return None


# --- Setup ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.markdown("<h1>Novella</h1>", unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Tell me the story you\'re chasing — I\'ll find it in the shelves.</div>',
    unsafe_allow_html=True,
)

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
            with st.spinner("Ranking the best matches..."):
                recommendations = get_reranked_results(refined_query, results)

            if not recommendations:
                st.warning("Couldn't generate recommendations this time — try searching again.")
            else:
                st.markdown('<div class="results-heading">Recommended for you</div>', unsafe_allow_html=True)
                with st.spinner("Finding where to read them..."):
                    for book in recommendations:
                        title = book.get("title", "Untitled")
                        reason = book.get("reason", "")

                        link = find_book_link(title)

                        # If no direct book link was found, create a search link
                        if not link:
                            from urllib.parse import quote
                            link = f"https://www.google.com/search?q={quote(title + ' book')}"

                        st.markdown(
                            f"""
                            <div class="book-card">
                                <div class="book-title">{title}</div>
                                <div class="book-reason">{reason}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Always show a clickable link
                        st.markdown(
                            f"""
                            <a href="{link}" target="_blank"
                               style="
                                   display: inline-block;
                                   background-color: #f4c9d6;
                                   color: #3e2723;
                                   padding: 8px 14px;
                                   border-radius: 10px;
                                   font-family: 'Quicksand', sans-serif;
                                   font-weight: 700;
                                   text-decoration: none;
                                   margin-bottom: 16px;
                               ">
                                📖 View book →
                            </a>
                            """,
                            unsafe_allow_html=True,
                        )