from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data/raw", exist_ok = True)

# Function To Download A Sample Of Wikipedia Articles Using The Hugging Face Datasets Library And Save It As A Parquet File
def download_wiki(sample_size=500):
    print("Downloading Wikipedia dataset...") # Print Message To Double Check That The Function Is Being Called 
    
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    rows = []

    for i, item in enumerate(dataset):
        rows.append({
            "id": item.get("id"),
            "title": item.get("title"),
            "text": item.get("text")
        })

        if i + 1 >= sample_size:
            break

    df = pd.DataFrame(rows)
    df.to_parquet("data/raw/wiki_sample.parquet", index=False)

    print(f"Downloaded {len(df)} samples from Wikipedia and saved to data/raw/wiki_sample.parquet")

# Main
if __name__ == "__main__":
    download_wiki()