## Step 1: Download Wikipedia Articles, Process The Text To Clean It And Remove Unwanted Articles, And Load Vocabulary Lists For Different Proficiency Levels 
from scripts.download_wiki import download_wiki
from scripts.preprocess_wiki import preprocess_wiki
from scripts.analyze_articles import analyze_articles
from scripts.chunk_articles import chunk_articles
from scripts.generate_embeddings import generate_embeddings
from scripts.build_faiss_index import build_faiss_index

# Download Wikipedia Articles
download_wiki()

# Clean and Process The Wikapedia Articles
preprocess_wiki()

# Analyze The Cleaned Articles With Different Vocabulary Lists
analyze_articles("beginner")
analyze_articles("intermediate")
analyze_articles("advanced")

# Chunk The Cleaned Articles Into Overlapping Chunks Based On Word Count
chunk_articles()

# Generate Embeddings For The Chunked Articles Using A Pretrained Sentence Transformer Model
generate_embeddings()

# Build A FAISS Index From The Precomputed Chunk Embeddings For Fast Similarity Search
build_faiss_index()