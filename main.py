import pandas as pd
import re
import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Authenticate with OpenAI
load_dotenv()

# --- DIAGNOSTIC TEST ---
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    print(" ERROR: Python cannot find the OPENAI_API_KEY. Check your .env file name!")
else:
    print(f" Key found! It starts with: {api_key[:12]}...")
# -----------------------

client = OpenAI(api_key=api_key)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. Load the dataset
df = pd.read_csv('books.csv')
print(f"Loaded {len(df)} total books from the dataset.")

# 3. Clean the data - UPDATED TO MATCH YOUR CSV
columns_we_need = ['title', 'author', 'desc'] 
df = df.dropna(subset=columns_we_need)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Strip weird characters
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'-]', '', text)
    # Fix spacing
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

print("Cleaning text...")
df['title'] = df['title'].apply(clean_text)
df['author'] = df['author'].apply(clean_text)
df['desc'] = df['desc'].apply(clean_text)

# 4. Create the rich text - UPDATED TO MATCH YOUR CSV
df['combined_text'] = (
    "Title: " + df['title'] + ". " +
    "Author: " + df['author'] + ". " +
    "Synopsis: " + df['desc']
)

# 5. The Embedding Function
def get_embedding(text):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# 6. TEST RUN: Isolate the first 5 books
print("\n--- Starting Embedding Generation (Test Batch) ---")
df_test = df.head(5).copy()

# 7. Generate vectors and save
print("Sending text to OpenAI... this should only take a few seconds.")
df_test['embedding'] = df_test['combined_text'].apply(get_embedding)

df_test.to_csv('books_with_embeddings_test.csv', index=False)
print("\nSuccess! Saved to 'books_with_embeddings_test.csv'")
print("Here is a sneak peek at the data:")
print(df_test[['title', 'embedding']].head())