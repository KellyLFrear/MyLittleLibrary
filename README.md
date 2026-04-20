# Intelligent Vocabulary Recommender System

CSUSM Final Sprint Challenge project for building a vocabulary-aware reading recommender over a Wikipedia corpus.

This repository now has a working **Deliverable 1 data pipeline** for:

1. Building configurable known-word lists
2. Downloading a Wikipedia subset
3. Preprocessing the articles
4. Tokenizing cleaned articles with an LLM-compatible subword tokenizer
5. Running the vocabulary analysis module
6. Producing candidate article rows by vocabulary level

---

## 1. Project goal

Given a student vocabulary level or known-word list, the system should identify Wikipedia articles that are:

- mostly made of words the student already knows
- include a small set of learnable new words
- readable for the student's current level

Current runtime vocabulary presets:

- **Beginner**: 1,000 total known words
- **Intermediate**: 3,000 total known words
- **Advanced**: 6,000 total known words

These runtime vocab files are **cumulative**:

- `beginner_1000.txt` = beginner words only
- `intermediate_3000.txt` = beginner + intermediate add-on words
- `advanced_6000.txt` = beginner + intermediate + advanced add-on words

Internal exclusive add-on bands used during construction:

- beginner add-on = 1,000
- intermediate add-on = 2,000
- advanced add-on = 3,000

This keeps the runtime totals aligned with the rubric while still modeling cumulative reader knowledge.

---

## 2. Current repo structure

```text
MyLittleLibrary/
├── data/
│   ├── raw/                          # raw Wikipedia parquet files
│   ├── processed/                    # cleaned + tokenized Wikipedia parquet files
│   └── vocab/                        # runtime vocab text files used by analysis
│       ├── beginner_1000.txt
│       ├── intermediate_3000.txt
│       └── advanced_6000.txt
├── outputs/
│   ├── article_stats.jsonl
│   ├── article_stats_50k.jsonl
│   └── article_stats_100k.jsonl
├── scripts/
│   ├── build_vocab.py
│   ├── build_vocab_lists.py
│   ├── download_wiki.py
│   ├── preprocess_wiki.py
│   ├── tokenize_articles.py
│   ├── analyze_articles.py
│   └── check_article_stats.py
├── vocab_sources/                    # source CSV/TXT vocab files
├── vocab_output/                     # scored vocab CSVs + summary.json
├── main.py
├── README.md
└── requirements.txt
```

---

## 3. What each script does

### `scripts/build_vocab_lists.py`
Builds vocabulary bands from source files in `vocab_sources/`.

Input:
- source vocab CSV/TXT files
- optional `banned_words.txt`

Recommended source CSV schema:

```csv
word,source_grade_hint,normalized_band,source_list,rank,count,academic,pos,notes
```

Outputs:
- `master_vocab_scored.csv`
- exclusive add-on band CSVs:
  - `beginner_band_1000.csv`
  - `intermediate_band_2000.csv`
  - `advanced_band_3000.csv`
- cumulative runtime vocab files:
  - `data/vocab/beginner_1000.txt`
  - `data/vocab/intermediate_3000.txt`
  - `data/vocab/advanced_6000.txt`
- `summary.json`

### `scripts/build_vocab.py`
Quick verification script that checks the sizes of:

- `data/vocab/beginner_1000.txt`
- `data/vocab/intermediate_3000.txt`
- `data/vocab/advanced_6000.txt`

### `scripts/download_wiki.py`
Downloads a Wikipedia subset and saves it as a parquet file in `data/raw/`.

Current patched behavior:
- supports `--sample-size`
- supports `--output`
- uses streaming for smaller debug-friendly pulls

### `scripts/preprocess_wiki.py`
Reads a raw wiki parquet, cleans text, removes disambiguation/stub-like pages, and writes a cleaned parquet.

Current patched behavior:
- supports `--input`
- supports `--output`
- supports `--min-words`

### `scripts/tokenize_articles.py`
Reads the cleaned wiki parquet and tokenizes each article with a Hugging Face / LLaMA-compatible tokenizer.

Current intended behavior:
- supports `--input`
- supports `--output`
- supports `--tokenizer`
- supports `--max-length`
- supports `--batch-size`

Expected tokenization outputs in the parquet:
- `llm_tokenizer_name`
- `llm_input_ids_json`
- `llm_attention_mask_json`
- `llm_token_count`

This script is the new Deliverable 1 step that prepares article text for actual LLM-style subword input instead of relying only on regex word tokenization.

### `scripts/analyze_articles.py`
Reads the tokenized wiki parquet and the three runtime vocab files, then computes:

- total word count
- unique word count
- coverage ratio
- new-word count/list
- readability score
- LLM token count if present
- tokenizer name if present
- word-to-LLM-token ratio if present
- candidate flag based on a coverage window

Current patched behavior:
- supports `--input`
- supports `--output`
- supports optional custom vocab file paths
- supports `--coverage-min`
- supports `--coverage-max`

Important note:
- word-level regex tokenization is still used inside this script for vocabulary coverage because the project vocab files are stored as whole words
- LLM subword tokenization now happens earlier in the pipeline inside `tokenize_articles.py`

### `scripts/check_article_stats.py`
Reads an article-stats JSONL file and prints:

- first few rows
- total rows by level
- candidate rows by level
- coverage statistics
- readability statistics
- simulated candidate counts for alternate windows

Current patched behavior:
- supports `--input`
- supports `--preview-rows`
- supports repeated `--window min:max`

---

## 4. Environment setup

### Windows PowerShell

Create and activate a virtual environment:

```powershell
python -m venv .venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned)
.\.venv\Scripts\Activate.ps1
```

Upgrade pip and install the packages needed for Deliverable 1:

```powershell
python -m pip install --upgrade pip
python -m pip install pandas pyarrow datasets textstat transformers nltk
```

If your selected tokenizer depends on SentencePiece, also install:

```powershell
python -m pip install sentencepiece
```

Verify imports:

```powershell
python -c "import pandas, pyarrow, datasets, textstat, transformers, nltk; print('deps ok')"
```

Because `analyze_articles.py` uses `textstat` for Flesch-Kincaid readability, download the NLTK pronunciation dictionary once before running the analysis step:

```powershell
python -m nltk.downloader cmudict
```

---

## 5. Vocabulary source setup

Place your vocab source files in `vocab_sources/`.

Recommended files:

```text
vocab_sources/
├── beginner_k_to_5.csv
├── intermediate_6_to_8.csv
├── advanced_9_to_12.csv
└── banned_words.txt
```

### Recommended grade-band strategy

Use flexible source labels in the CSV, but normalize internally to these three bands:

- `k-5th grade`
- `6-8`
- `9-12`

Example rows:

```csv
word,source_grade_hint,normalized_band,source_list,rank,count,academic,pos,notes
water,K,K-5,fry_dolch,150,,false,noun,
compare,7,6-8,ngsl,1800,,false,verb,
analyze,10,9-12,coca,4200,,true,verb,
```

---

## 6. Run the project through Deliverable 1 analysis

## Step 1: Build the vocab lists

Build all vocab bands and write cumulative runtime vocab files:

```powershell
python scripts/build_vocab_lists.py ^
  --input vocab_sources ^
  --output vocab_output ^
  --runtime-vocab-dir data/vocab ^
  --only-band all
```

Expected summary pattern:

- exclusive add-on band counts:
  - beginner = 1000
  - intermediate = 2000
  - advanced = 3000
- runtime total counts:
  - beginner = 1000
  - intermediate = 3000
  - advanced = 6000

Expected output files:

```text
vocab_output/
├── master_vocab_scored.csv
├── beginner_band_1000.csv
├── intermediate_band_2000.csv
├── advanced_band_3000.csv
└── summary.json

data/vocab/
├── beginner_1000.txt
├── intermediate_3000.txt
└── advanced_6000.txt
```

### Optional single-band runs

```powershell
python scripts/build_vocab_lists.py --input vocab_sources --output vocab_output --runtime-vocab-dir data/vocab --only-band beginner
python scripts/build_vocab_lists.py --input vocab_sources --output vocab_output --runtime-vocab-dir data/vocab --only-band intermediate
python scripts/build_vocab_lists.py --input vocab_sources --output vocab_output --runtime-vocab-dir data/vocab --only-band advanced
```

If a band is slightly short while testing, you can temporarily allow it:

```powershell
python scripts/build_vocab_lists.py --input vocab_sources --output vocab_output --runtime-vocab-dir data/vocab --only-band intermediate --allow-shortfall
```

---

## Step 2: Verify vocab sizes

```powershell
python scripts/build_vocab.py
```

Expected result:

```text
Beginner Size: 1000
Intermediate Size: 3000
Advanced Size: 6000
```

---

## Step 3: Download Wikipedia data

### Small debug run

```powershell
python scripts/download_wiki.py --sample-size 500 --output data/raw/wiki_sample.parquet
```

### Larger debug run

```powershell
python scripts/download_wiki.py --sample-size 50000 --output data/raw/wiki_50k.parquet
```

### Full Deliverable 1 run

```powershell
python scripts/download_wiki.py --sample-size 100000 --output data/raw/wiki_100k.parquet
```

---

## Step 4: Preprocess the articles

### Small sample

```powershell
python scripts/preprocess_wiki.py --input data/raw/wiki_sample.parquet --output data/processed/wiki_clean.parquet
```

### 50k raw run

```powershell
python scripts/preprocess_wiki.py --input data/raw/wiki_50k.parquet --output data/processed/wiki_50k_clean.parquet
```

### 100k raw run

```powershell
python scripts/preprocess_wiki.py --input data/raw/wiki_100k.parquet --output data/processed/wiki_100k_clean.parquet
```

---

## Step 5: Tokenize cleaned articles for LLM input

This is the new pipeline step added for Deliverable 1 so the corpus is prepared with a real subword tokenizer before analysis.

### Small sample

```powershell
python scripts/tokenize_articles.py --input data/processed/wiki_clean.parquet --output data/processed/wiki_tokenized.parquet --tokenizer meta-llama/Llama-3.1-8B-Instruct --max-length 2048
```

### 50k cleaned run

```powershell
python scripts/tokenize_articles.py --input data/processed/wiki_50k_clean.parquet --output data/processed/wiki_50k_tokenized.parquet --tokenizer meta-llama/Llama-3.2-1B-Instruct --max-length 2048
```

### 100k cleaned run

```powershell
python scripts/tokenize_articles.py --input data/processed/wiki_100k_clean.parquet --output data/processed/wiki_100k_tokenized.parquet --tokenizer models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --max-length 2048
```

Expected effect:
- the cleaned parquet is converted into a tokenized parquet
- each row now includes LLM-oriented subword token data
- `analyze_articles.py` can now report LLM token counts alongside word-level vocabulary coverage

---

## Step 6: Analyze article vocabulary

Current working candidate window for general Wikipedia debug/reporting runs:

- `--coverage-min 0.45`
- `--coverage-max 0.70`

This is lower than the rubric example window of 0.90–0.97, but the window is configurable and this lower range produced usable candidate counts on the current Wikipedia corpus.

### Small sample

```powershell
python scripts/analyze_articles.py --input data/processed/wiki_tokenized.parquet --output outputs/article_stats.jsonl --coverage-min 0.45 --coverage-max 0.70
```

### 50k tokenized run

```powershell
python scripts/analyze_articles.py --input data/processed/wiki_50k_tokenized.parquet --output outputs/article_stats_50k.jsonl --coverage-min 0.45 --coverage-max 0.70
```

### 100k tokenized run

```powershell
python scripts/analyze_articles.py --input data/processed/wiki_100k_tokenized.parquet --output outputs/article_stats_100k.jsonl --coverage-min 0.45 --coverage-max 0.70
```

### What analysis computes

For each article and for each vocabulary level, the analyzer computes:

- `Total_Words`
- `Unique_Words`
- `Coverage_Ratio`
- `New_Word_Count`
- `New_Words`
- `Flesch_Kincaid_Grade`
- `LLM_Token_Count`
- `Tokenizer_Name`
- `Word_to_LLM_Token_Ratio`
- `Candidate`

A row is a **candidate row** when its article coverage ratio falls within the configured coverage window for that level.

---

## Step 7: Check the analysis output

Small sample:

```powershell
python scripts/check_article_stats.py
```

50k tokenized run:

```powershell
python scripts/check_article_stats.py --input outputs/article_stats_50k.jsonl --window 0.45:0.70 --window 0.50:0.75 --window 0.55:0.80
```

100k tokenized run:

```powershell
python scripts/check_article_stats.py --input outputs/article_stats_100k.jsonl --window 0.45:0.70 --window 0.50:0.75 --window 0.55:0.80
```

This checks:

- JSONL file exists
- rows are written correctly
- candidate counts look reasonable
- coverage ratios are sensible by level
- readability values are being computed
- alternate windows can be compared without rerunning analysis

---

## 7. Current working project results

## Small sample run
- raw sample downloaded: 500
- cleaned articles kept: 259
- article stats rows: 777

## 50k raw run
- raw sample downloaded: 50,000
- cleaned articles kept: 28,203
- article stats rows: 84,609

## 100k raw run
- raw sample downloaded: 100,000
- cleaned articles kept: 55,256
- article stats rows: 165,768

This 100k raw run is the first run that produces **more than 50,000 cleaned articles**, which satisfies the rubric expectation for a minimum 50,000-article cleaned Wikipedia subset after preprocessing.

### 100k cleaned run metrics using coverage window 0.45-0.70

Rows by level:
- Beginner: 55,256
- Intermediate: 55,256
- Advanced: 55,256

Candidate rows by level:
- Beginner: 30,414
- Intermediate: 45,143
- Advanced: 47,481

Coverage ratio averages:
- Beginner: 0.4457
- Intermediate: 0.5056
- Advanced: 0.5244

Coverage trend is correct:
- beginner < intermediate < advanced

Simulated candidate rows for alternate windows:
- window 0.45-0.70
  - Beginner: 30,414
  - Intermediate: 45,143
  - Advanced: 47,481
- window 0.50-0.75
  - Beginner: 12,941
  - Intermediate: 34,475
  - Advanced: 39,387
- window 0.55-0.80
  - Beginner: 2,452
  - Intermediate: 16,619
  - Advanced: 23,419

---

## 8. Deliverable 1 status

The repository now supports:

- configurable beginner/intermediate/advanced vocab lists
- cumulative runtime known-word files
- reproducible raw Wikipedia download
- reproducible preprocessing
- reproducible LLM-compatible subword tokenization of cleaned articles
- article-level vocabulary analysis
- candidate filtering by configurable known-word coverage
- analysis summaries over a 50k+ cleaned article corpus

This covers the core of **Deliverable 1** before embeddings/vector search are added.

---

## 9. Next steps after Step 6 / Deliverable 1 analysis

The next work should be:

1. Save corpus summary statistics for the report
2. Inspect extreme readability outliers
3. Chunk articles for embedding generation
4. Generate dense embeddings
5. Build a vector index
6. Add retrieval and vocabulary-aware re-ranking
7. Connect the retrieval stage to the recommendation generator

---

## 10. Common problems

### `ModuleNotFoundError: No module named 'textstat'`

```powershell
python -m pip install textstat
```

### `ModuleNotFoundError: No module named 'datasets'`

```powershell
python -m pip install datasets
```

### `ModuleNotFoundError: No module named 'transformers'`

```powershell
python -m pip install transformers
```

### `analyze_articles.py` fails with a `cmudict` or NLTK corpus error

```powershell
python -m pip install nltk
python -m nltk.downloader cmudict
```

### Some tokenizers fail because SentencePiece is not installed

```powershell
python -m pip install sentencepiece
```

### `Unable to find a usable engine` for parquet

```powershell
python -m pip install pyarrow
```

### PowerShell does not like bash-style heredoc commands

Use a Python script file such as `scripts/check_article_stats.py` instead of:

```bash
python - <<'PY'
...
PY
```

### Hugging Face warning about unauthenticated requests

The Wikipedia downloads still work without a token, but larger Hub requests may be slower or more rate-limited.

Later, when needed for larger or repeated runs:

```powershell
hf auth login
```

Do **not** hardcode tokens into source files or commit them to git.

---

## 11. Suggested requirements.txt for current pipeline

At minimum for Steps 1 through 6:

```txt
pandas
pyarrow
datasets
textstat
transformers
nltk
```

If your tokenizer needs it, also include:

```txt
sentencepiece
```

A more exact frozen version can be generated with:

```powershell
python -m pip freeze > requirements.txt
```

---

## 12. Suggested git workflow

Before committing, verify:

```powershell
python scripts/build_vocab.py
python scripts/check_article_stats.py --input outputs/article_stats_100k.jsonl --window 0.45:0.70 --window 0.50:0.75 --window 0.55:0.80
```

Recommended commit checkpoints:

- vocab source files added
- runtime vocab files generated
- wiki download working
- preprocessing working
- tokenization working
- analysis output generated
- README and requirements updated

---

## 13. Notes

- The 500-article run is only for debugging.
- The 50k raw run did **not** survive cleaning at the 50k threshold.
- The 100k raw run produced **55,256 cleaned articles**, which is the current rubric-safe corpus size.
- General Wikipedia is harder than a learner-targeted reading corpus, so a lower temporary coverage window was used for analysis.
- The final repository should keep very large processed corpora or vector indexes out of git if they are too large.
