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
def analyze_articles():
    print("Running analysis for all levels...")

    vocab_paths = {
        "Beginner": "data/vocab/beginner_1000.txt",
        "Intermediate": "data/vocab/intermediate_3000.txt",
        "Advanced": "data/vocab/advanced_6000.txt"
    }

    df = pd.read_parquet("data/processed/wiki_clean.parquet")

    output_path = "outputs/article_stats.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for level, vocab_path in vocab_paths.items():
            print(f"Processing {level} level...")

            known_words = load_word_list(vocab_path)

            for _, row in df.iterrows():
                text = row["clean_text"]
                tokens = tokenize_words(text)

                if not tokens:
                    continue

                total_words = len(tokens)
                unique_words = len(set(tokens))
                known_count = sum(1 for w in tokens if w in known_words)
                coverage_ratio = known_count / total_words

                article_new_words = sorted(set(w for w in tokens if w not in known_words))
                new_word_count = len(article_new_words)

                readability = flesch_kincaid_grade(text)

                is_candidate = 0.90 <= coverage_ratio <= 0.97

                record = {
                    "ID": row["id"],
                    "Title": row["title"],
                    "Level": level,
                    "Total_Words": total_words,
                    "Unique_Words": unique_words,
                    "Coverage_Ratio": coverage_ratio,
                    "New_Word_Count": new_word_count,
                    "New_Words": ", ".join(article_new_words[:30]),
                    "Flesch_Kincaid_Grade": readability,
                    "Candidate": is_candidate
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved JSONL output to {output_path}")


# Main
if __name__ == "__main__":
    analyze_articles()