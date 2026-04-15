import os
import numpy as np
import pandas as pd
import faiss

os.makedirs("data/processed", exist_ok=True)


# Function To Build A FAISS Index From Precomputed Chunk Embeddings
def build_faiss_index():
    print("Loading embeddings and metadata...")

    embeddings_path = "data/processed/chunk_embeddings.npy"
    metadata_path = "data/processed/chunk_metadata.parquet"

    embeddings = np.load(embeddings_path)
    metadata = pd.read_parquet(metadata_path)

    print(f"Loaded embeddings with shape: {embeddings.shape}")
    print(f"Loaded metadata rows: {len(metadata)}")

    # FAISS expects float32
    embeddings = embeddings.astype("float32")

    # Get embedding dimension
    dimension = embeddings.shape[1]

    print(f"Building FAISS IndexFlatL2 with dimension {dimension}...")

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    index_output = "data/processed/wiki_faiss.index"
    faiss.write_index(index, index_output)

    print(f"Saved FAISS index to {index_output}")
    print(f"Total vectors indexed: {index.ntotal}")


# Main
if __name__ == "__main__":
    build_faiss_index()