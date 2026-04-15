import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

os.makedirs("data/processed", exist_ok=True)


# Function To Generate Embeddings For Chunked Wikipedia Articles
def generate_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    print(f"Loading embedding model: {model_name}")

    df = pd.read_parquet("data/processed/wiki_chunks.parquet")

    print(f"Loaded {len(df)} chunks.")

    model = SentenceTransformer(model_name)

    texts = df["chunk_text"].tolist()

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    metadata = df[["article_id", "title", "chunk_id", "chunk_text", "chunk_word_count"]].copy()

    metadata_output = "data/processed/chunk_metadata.parquet"
    embeddings_output = "data/processed/chunk_embeddings.npy"

    metadata.to_parquet(metadata_output, index=False)
    np.save(embeddings_output, embeddings)

    print(f"Saved metadata to {metadata_output}")
    print(f"Saved embeddings to {embeddings_output}")
    print(f"Embeddings shape: {embeddings.shape}")


# Main
if __name__ == "__main__":
    generate_embeddings()