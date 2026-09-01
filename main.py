import pandas as pd
import re
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# --- Config ---
SOURCE_CSV = "books.csv"
OUTPUT_CSV = "books_with_embeddings.csv"
CHECKPOINT_EVERY = 5      # save progress every N batches
BATCH_SIZE = 100          # books embedded per API call
EMBED_MODEL = "text-embedding-3-small"

# 1. Authenticate with OpenAI
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise SystemExit("ERROR: OPENAI_API_KEY not found. Check your .env file.")
client = OpenAI(api_key=api_key)

# 2. Load the dataset
df = pd.read_csv(SOURCE_CSV)
print(f"Loaded {len(df)} total books from the dataset.")

# 3. Clean the data
columns_we_need = ['title', 'author', 'desc']
df = df.dropna(subset=columns_we_need).reset_index(drop=True)
print(f"{len(df)} books remain after dropping rows missing title/author/desc.")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)          # strip HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'-]', '', text)  # strip weird chars
    text = re.sub(r'\s+', ' ', text)               # fix spacing
    return text.strip()


print("Cleaning text...")
df['title'] = df['title'].apply(clean_text)
df['author'] = df['author'].apply(clean_text)
df['desc'] = df['desc'].apply(clean_text)

# 4. Build the rich text used for embedding
df['combined_text'] = (
    "Title: " + df['title'] + ". " +
    "Author: " + df['author'] + ". " +
    "Synopsis: " + df['desc']
)

# 5. Rough cost estimate before spending any money
# text-embedding-3-small ~ $0.02 per 1M tokens. ~4 chars/token is a safe rough estimate.
total_chars = df['combined_text'].str.len().sum()
approx_tokens = total_chars / 4
approx_cost = (approx_tokens / 1_000_000) * 0.02
print(f"\nApprox tokens: {approx_tokens:,.0f}  |  Estimated cost: ${approx_cost:.4f}\n")

# 6. Resume support: if a partial output file exists, pick up where we left off
if os.path.exists(OUTPUT_CSV):
    done_df = pd.read_csv(OUTPUT_CSV)
    done_titles = set(done_df['title'])
    print(f"Found existing progress file with {len(done_df)} books already embedded. Resuming...")
else:
    done_df = pd.DataFrame(columns=list(df.columns) + ['embedding'])
    done_titles = set()

remaining_df = df[~df['title'].isin(done_titles)].reset_index(drop=True)
print(f"{len(remaining_df)} books left to embed.\n")


def embed_batch(texts):
    """Embed a list of strings in a single API call. Returns list of vectors."""
    response = client.embeddings.create(input=texts, model=EMBED_MODEL)
    # response.data is returned in the same order as the input list
    return [item.embedding for item in response.data]


# 7. Process in batches, saving progress periodically
all_rows = [done_df] if len(done_df) else []
batch_rows = []

num_batches = (len(remaining_df) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_num in range(num_batches):
    start = batch_num * BATCH_SIZE
    end = start + BATCH_SIZE
    batch_df = remaining_df.iloc[start:end]

    print(f"Embedding batch {batch_num + 1}/{num_batches} ({len(batch_df)} books)...")

    try:
        embeddings = embed_batch(batch_df['combined_text'].tolist())
    except Exception as e:
        print(f"  Error on batch {batch_num + 1}: {e}")
        print("  Retrying once after a short pause...")
        time.sleep(5)
        try:
            embeddings = embed_batch(batch_df['combined_text'].tolist())
        except Exception as e2:
            print(f"  Retry failed too: {e2}")
            print("  Stopping here — rerun the script later, it will resume automatically.")
            break

    batch_df = batch_df.copy()
    batch_df['embedding'] = embeddings
    batch_rows.append(batch_df)

    # Checkpoint: write progress to disk every CHECKPOINT_EVERY batches
    if (batch_num + 1) % CHECKPOINT_EVERY == 0 or (batch_num + 1) == num_batches:
        combined = pd.concat(all_rows + batch_rows, ignore_index=True)
        combined.to_csv(OUTPUT_CSV, index=False)
        print(f"  Checkpoint saved: {len(combined)} books embedded so far.\n")

print(f"\nDone. Final file saved to '{OUTPUT_CSV}'.")