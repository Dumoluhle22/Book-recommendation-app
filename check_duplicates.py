import pandas as pd

df = pd.read_csv("books_with_embeddings.csv")
total = len(df)
unique = df['title'].nunique()

print(f"Total rows: {total}")
print(f"Unique titles: {unique}")

if total == unique:
    print("No duplicate titles — looks like a genuinely large dataset.")
else:
    dupes = total - unique
    print(f"{dupes} duplicate rows found. Example duplicates:")
    print(df[df.duplicated(subset='title', keep=False)].sort_values('title')['title'].head(10))