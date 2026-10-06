import streamlit as st
import chromadb
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

DB_PATH = "./my_books_db"
COLLECTION_NAME = "african_literature"

st.set_page_config(
    page_title="Bartech — African Lit Recommender",
    page_icon="📖",
    layout="centered",
)

# --- Custom styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,500;0,600;0,700;1,500&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #14171f;
    --surface: #1e2330;
    --gold: #c99a3e;
    --coral: #c1443b;
    --text: #ede8dd;
    --muted: #9da3b4;
}

.stApp {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    max-width: 700px;
    padding-top: 3rem;
}

/* Title */
h1 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    font-style: italic;
    color: var(--text) !important;
    font-size: 2.4rem !important;
    letter-spacing: -0.01em;
    margin-bottom: 0.1rem !important;
}

.subtitle {
    color: var(--muted);
    font-size: 0.95rem;
    margin-bottom: 2.2rem;
}

/* Text input */
div[data-testid="stTextInput"] input {
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid #333a4d !important;
    border-radius: 8px !important;
    padding: 0.7rem 1rem !important;
    font-size: 1rem !important;
}
div[data-testid="stTextInput"] input:focus {
    border: 1px solid var(--gold) !important;
    box-shadow: 0 0 0 1px var(--gold) !important;
}
div[data-testid="stTextInput"] label {
    color: var(--muted) !important;
    font-size: 0.9rem !important;
}

/* Button */
div.stButton > button {
    background-color: var(--gold) !important;
    color: #14171f !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.6rem !important;
    margin-top: 0.6rem;
    transition: opacity 0.15s ease;
}
div.stButton > button:hover {
    opacity: 0.85;
}

/* Info / warning boxes */
div[data-testid="stAlert"] {
    background-color: var(--surface) !important;
    border: 1px solid #333a4d !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
}

/* Section heading above results */
.results-heading {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.3rem;
    color: var(--text);
    margin-top: 2.2rem;
    margin-bottom: 1rem;
}

/* Book result card */
.book-card {
    background-color: var(--surface);
    border-left: 3px solid var(--gold);
    border-radius: 6px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.9rem;
}
.book-card .book-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.1rem;
    color: var(--text);
    margin-bottom: 0.3rem;
}
.book-card .book-reason {
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    color: var(--muted);
    line-height: 1.5;
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


# --- Setup ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.markdown("<h1>Bartech</h1>", unsafe_allow_html=True)
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
                for book in recommendations:
                    title = book.get('title', 'Untitled')
                    reason = book.get('reason', '')
                    st.markdown(
                        f"""<div class="book-card">
                            <div class="book-title">{title}</div>
                            <div class="book-reason">{reason}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )