import pandas as pd
import re
import os
import textstat
import json

os.makedirs("outputs", exist_ok=True)

# Function That Loads a Vocabulary File And Returns A Set Of Lowercase Words
def load_word_list(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


# Function To Tokenize Text Into Lowercase Words Using Regex
def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


# Function To Compute Flesch-Kincaid Grade Level Using textstat
def flesch_kincaid_grade(text: str) -> float:
    return textstat.flesch_kincaid_grade(text)


# Function To Analyze Articles And Save Results As JSON Lines
# Output format matches the RAG pipeline's expected input:
#   { "id", "title", "text",
#     "coverage_ratio": {"beginner": float, "intermediate": float, "advanced": float},
#     "new_words":      {"beginner": [...], "intermediate": [...], "advanced": [...]},
#     "readability_score": float }
def analyze_articles():
    print("Running analysis for all levels...")

    vocab_paths = {
        "beginner": "data/vocab/beginner_1000.txt",
        "intermediate": "data/vocab/intermediate_3000.txt",
        "advanced": "data/vocab/advanced_6000.txt",
    }

    df = pd.read_parquet("data/processed/wiki_clean.parquet")

    output_path = "outputs/article_stats.jsonl"

    # Pre-load all vocab sets once
    vocab_sets = {level: load_word_list(path) for level, path in vocab_paths.items()}

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            text = row["clean_text"]
            tokens = tokenize_words(text)

            if not tokens:
                continue

            readability = flesch_kincaid_grade(text)

            coverage_ratio = {}
            new_words = {}

            for level, known_words in vocab_sets.items():
                known_count = sum(1 for w in tokens if w in known_words)
                coverage_ratio[level] = known_count / len(tokens)
                new_words[level] = sorted(set(w for w in tokens if w not in known_words))

            record = {
                "id": row["id"],
                "title": row["title"],
                "text": text,
                "coverage_ratio": coverage_ratio,
                "new_words": new_words,
                "readability_score": readability,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved JSONL output to {output_path}")


# Main
if __name__ == "__main__":
    analyze_articles()