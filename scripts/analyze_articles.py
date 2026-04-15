import pandas as pd
import re
import os
import textstat

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


# Function To Analyze Articles
def analyze_articles(level="intermediate"): # Defaults To Intermediate Level If No Level Is Specified
    print(f"Running analysis for {level} level...") # Print Message To Double Check If It's Running

    # Defines The Paths To The Vocabulary Files For Each Level
    vocab_paths = {
        "beginner": "data/vocab/beginner_1000.txt",
        "intermediate": "data/vocab/intermediate_3000.txt",
        "advanced": "data/vocab/advanced_6000.txt"
    }

    # If The Level Isn't Beginner, Intermediate, Or Advanced, Raise An Error
    if level not in vocab_paths:
        raise ValueError(f"Invalid level: {level}. Choose from beginner, intermediate, advanced.")

    # Load The Known Words For The Specified Level
    known_words = load_word_list(vocab_paths[level])

    # Load The Cleaned Wikipedia Articles From The Processed Parquet File
    df = pd.read_parquet("data/processed/wiki_clean.parquet")

    # Creates An Empty List
    results = []

    # Iterates Over Each Row In The DataFrame To Analyze Each Article
    for _, row in df.iterrows():
        text = row["clean_text"]
        tokens = tokenize_words(text)
        if not tokens:
            continue

        total_words = len(tokens)
        unique_words = len(set(tokens))
        known_count = sum(1 for w in tokens if w in known_words)
        coverage_ratio = known_count / total_words

        new_words = sorted(set(w for w in tokens if w not in known_words))
        new_word_count = len(new_words)

        # Calculates A Readability Score Using The Flesch-Kincaid Formula
        readability = flesch_kincaid_grade(text)

        is_candidate = 0.90 <= coverage_ratio <= 0.97 # Must Have Between 90-97% Of Words Known To Be A Candidate Article For The Given Level

        # Adds A Dictionary With The Article's Statistics To The Results List
        results.append({
            "ID": row["id"], # ID Of The Article
            "Title": row["title"], # The Title Of The Article
            "Level": level, # The Level Of The Article (Beginner, Intermediate, Advanced)
            "Total_Words": total_words, # Total Number Of Words
            "Unique_Words": unique_words, # Total Number Of Unique Words 
            "Coverage_Ratio": coverage_ratio, # Ratio Of Known Words To Total Words
            "New_Word_Count": new_word_count, # Number Of New Words Not In The Known Vocabulary For The Given Level
            "New_Words": ", ".join(new_words[:30]), # Joins The First 30 New Words Into A Comma-Separated String
            "Flesch_Kincaid_Grade": readability, # Readability Score
            "Candidate": is_candidate, # If It's A Candidate Article
        })

    # Converts The Results List Of Dictionaries Into A DataFrame And Saves It As A CSV File For The Given Level
    out = pd.DataFrame(results)
    out.to_csv(f"outputs/article_stats_{level}.csv", index=False)

    # Prints The First Few Rows Of The Output
    print(out.head())
    print(f"{level} Candidate articles: {out['Candidate'].sum()}") # Prints The Number Of Candidate Articles For The Given Level


# Main
if __name__ == "__main__":
    analyze_articles("beginner")
    analyze_articles("intermediate")
    analyze_articles("advanced")