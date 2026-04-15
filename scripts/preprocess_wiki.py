import pandas as pd
import re
import unicodedata
import os

os.makedirs("data/processed", exist_ok=True)

# Function To Clean Text By Normalizing Unicode, Removing Extra Whitespace, And Stripping Leading/Trailing Spaces
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Function To Identify Disambiguation Pages Based On Title And Text Content
def is_disambiguation(title: str, text: str) -> bool:
    t = (title or "").lower()
    x = (text or "").lower()
    return "(disambiguation)" in t or "may refer to:" in x


# Function To Identify Stub Articles Based On Word Count (Simple Heuristic)
def is_stub(text: str) -> bool:
    return len((text or "").split()) < 150


# Main Function To Preprocess Wikipedia Data By Cleaning Text, Removing Disambiguation Pages And Stubs, And Saving The Cleaned Data
def preprocess_wiki():
    print("Running Preprocessing...") # Print Message To Double Check That The Function Is Being Called

    df = pd.read_parquet("data/raw/wiki_sample.parquet")

    df["clean_text"] = df["text"].apply(clean_text)

    df = df[
        (~df.apply(lambda row: is_disambiguation(row["title"], row["clean_text"]), axis=1)) &
        (~df["clean_text"].apply(is_stub))
    ].copy()

    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

    df.to_parquet("data/processed/wiki_clean.parquet", index=False)
    print(f"Saved {len(df)} cleaned articles.")

# Main
if __name__ == "__main__":
    preprocess_wiki()