import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

# Function To Split Text Into Overlapping Chunks Based On Word Count
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    words = text.split()

    if not words:
        return []

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]

        if not chunk_words:
            continue

        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        if i + chunk_size >= len(words):
            break

    return chunks


# Function To Chunk All Cleaned Wikipedia Articles And Save Them As A New Parquet File
def chunk_articles(chunk_size: int = 200, overlap: int = 50):
    print(f"Chunking articles with chunk_size={chunk_size}, overlap={overlap}...")

    df = pd.read_parquet("data/processed/wiki_clean.parquet")

    chunked_rows = []

    for _, row in df.iterrows():
        article_id = row["id"]
        title = row["title"]
        text = row["clean_text"]

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for chunk_id, chunk in enumerate(chunks):
            chunked_rows.append({
                "article_id": article_id,
                "title": title,
                "chunk_id": chunk_id,
                "chunk_text": chunk,
                "chunk_word_count": len(chunk.split())
            })

    chunk_df = pd.DataFrame(chunked_rows)
    output_path = "data/processed/wiki_chunks.parquet"
    chunk_df.to_parquet(output_path, index=False)

    print(f"Saved {len(chunk_df)} Chunks To {output_path}")
    print(chunk_df.head())


# Main
if __name__ == "__main__":
    chunk_articles()