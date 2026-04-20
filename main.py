## Step 1: Download Wikipedia Articles, Process The Text To Clean It And Remove Unwanted Articles, And Load Vocabulary Lists For Different Proficiency Levels 
from scripts.download_wiki import download_wiki
from scripts.preprocess_wiki import preprocess_wiki
from scripts.analyze_articles import analyze_articles

# Download Wikipedia Articles
download_wiki()

# Clean and Process The Wikipedia Articles
preprocess_wiki()

# Analyze The Cleaned Articles With Different Vocabulary Lists
# Outputs outputs/article_stats.jsonl in the format expected by the RAG pipeline
analyze_articles()

## Step 2: Build The FAISS Index For The RAG Pipeline
import subprocess
import sys

print("Building FAISS index from outputs/article_stats.jsonl ...")
subprocess.run(
    [sys.executable, "scripts/build_index.py",
     "--articles", "outputs/article_stats.jsonl",
     "--index-dir", "data/faiss_index"],
    check=True,
)