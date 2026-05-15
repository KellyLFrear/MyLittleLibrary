# My Little Library

**My Little Library** is a vocabulary-aware reading recommender and story generator for students. It recommends Wikipedia-based reading material that is relevant to a student's topic interest while staying slightly above the student's current known-word level. The final system combines a reproducible Wikipedia data pipeline, exclusive grade-band vocabulary lists, sentence-transformer embeddings, FAISS vector search, vocabulary-aware reranking, SQLite-backed user profiles, a Flask web interface, and local LLaMA GGUF inference.

---

## Project goals

The system is designed to answer one practical educational question:

> Given a student profile and a topic, can the app recommend or generate reading material that is understandable but still introduces useful new vocabulary?

The implemented system supports:

- user login/register and session-based Flask routes
- beginner, intermediate, and advanced reading profiles
- SQLite persistence for users, vocabulary knowledge, reading profiles, recommendations, user books, and saved stories
- Wikipedia article preprocessing and vocabulary analysis
- chunk-level embeddings using `all-MiniLM-L6-v2`
- FAISS vector search over normalized embeddings
- two-stage retrieval: broad semantic search followed by vocabulary-aware reranking
- local LLaMA 3.1 8B Instruct GGUF generation through `llama-cpp-python`
- generated recommendation explanations and personalized stories
- RAG evaluation, retrieval ablation, ROUGE, optional BERTScore, and coverage-window metrics

The project does **not** train QLoRA adapters. The generator is a pretrained/instruction-tuned LLaMA GGUF model used for local inference.

---

## Current architecture

The repository is organized as a modular Flask application with an offline data/indexing pipeline and an online RAG inference pipeline.

### Offline pipeline

```text
Wikipedia download
  -> preprocessing
  -> subword tokenization
  -> vocabulary/readability analysis
  -> sentence-aware chunking
  -> sentence-transformer embedding
  -> FAISS index + chunk metadata
```

### Online recommendation flow

```text
student login/profile
  -> topic query or profile-driven recommendation request
  -> query embedding
  -> FAISS broad retrieval
  -> vocabulary-aware reranking
  -> LLaMA recommendation generation
  -> SQLite save
  -> frontend display
```

### Main runtime components

| Component | Implementation |
|---|---|
| Web server | `app.py` using Flask |
| Frontend | `front-end/index.html`, `main_page.html`, CSS, JS |
| Relational database | SQLite via `src/db/schema.sql` and repositories |
| Vocabulary analysis | `scripts/analyze_articles.py` |
| Chunking | `src/embeddings/chunker.py` |
| Embeddings | `src/embeddings/embedder.py` using `all-MiniLM-L6-v2` |
| Vector store | `src/embeddings/vector_store.py` using FAISS `IndexFlatIP` |
| Reranking | `src/rag/reranker.py` using semantic + vocabulary score |
| Retrieval | `src/rag/retriever.py` two-stage/adaptive retrieval |
| Generator | `src/rag/pipeline.py` `LlamaCppGenerator` |
| CLI RAG test | `scripts/run_rag.py` |
| CLI story test | `scripts/generate_story.py` |
| Evaluation | `scripts/evaluate_rag.py` |

---

## Repository structure

```text
MyLittleLibrary/
├── app.py                              # Flask backend + API routes
├── main.py                             # Convenience pipeline runner
├── requirements.txt                    # Python dependencies used by final code
├── README.md
├── data/
│   ├── eval_queries.json               # 15 evaluation queries, 5 per level
│   ├── raw/                            # small sample raw parquet included; full data ignored
│   ├── processed/                      # small cleaned parquet included; full data ignored
│   ├── vocab/                          # final runtime vocabulary text files
│   └── vocab_sources/                  # source CSV vocabulary files
├── front-end/
│   ├── index.html                      # login/register page
│   ├── main_page.html                  # book-style main UI
│   ├── login.js
│   ├── script.js
│   ├── style.css
│   ├── main_page_stylesheet.css
│   └── images/
├── scripts/
│   ├── build_vocab_lists.py            # builds exclusive vocab bands
│   ├── build_vocab.py                  # quick vocab size verification
│   ├── download_wiki.py                # streams Wikipedia from Hugging Face datasets
│   ├── preprocess_wiki.py              # cleans/filter articles
│   ├── tokenize_articles.py            # HF or local GGUF tokenizer support
│   ├── analyze_articles.py             # article-level coverage/readability metadata
│   ├── build_index.py                  # chunk, embed, and save FAISS index
│   ├── run_rag.py                      # command-line recommendation test
│   ├── generate_story.py               # command-line story generation test
│   ├── evaluate_rag.py                 # retrieval/generation/ablation evaluation
│   ├── generate_eval_queries.py
│   └── seed_test_users.py              # creates beginner/intermediate/advanced demo users
└── src/
    ├── db/
    │   ├── connection.py
    │   ├── repositories.py
    │   └── schema.sql
    ├── embeddings/
    │   ├── chunker.py
    │   ├── embedder.py
    │   └── vector_store.py
    └── rag/
        ├── pipeline.py
        ├── retriever.py
        ├── reranker.py
        └── student_profile.py
```

Large generated files are intentionally not committed by default. This includes the full Wikipedia parquet files, `outputs/`, the full FAISS index directories, SQLite database files, and local GGUF model files.

---

## Vocabulary design

The final vocabulary files are **exclusive bands**:

| File | Meaning |
|---|---|
| `data/vocab/beginner_1000.txt` | beginner-only K-5 band |
| `data/vocab/intermediate_3000.txt` | intermediate-only grades 6-8 band |
| `data/vocab/advanced_6000.txt` | advanced-only grades 9-12 band |

The files themselves are not nested. At runtime, the analysis/indexing/profile code builds cumulative known-word sets when modeling a student:

| Student level | Known-word set used for coverage |
|---|---|
| Beginner | beginner band |
| Intermediate | beginner + intermediate bands |
| Advanced | beginner + intermediate + advanced bands |

This keeps the source vocabulary bands clean while still modeling the realistic assumption that an advanced reader also knows beginner and intermediate words.

---

## Coverage windows

Known-word coverage is the main difficulty signal.

| Mode | Target known-word range | Purpose |
|---|---:|---|
| RAG article recommendation | 85-97% | General recommendation target |
| Story challenge: low | 95-98% | Mostly familiar text |
| Story challenge: medium | 90-95% | Moderate vocabulary challenge |
| Story challenge: high | 85-92% | Higher challenge |

The weighted reranker uses the deployed scoring balance:

```text
final_score = 0.35 * semantic_score + 0.65 * vocabulary_fit_score
```

---

## Setup

### 1. Create and activate a virtual environment

Ubuntu/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Notes:

- `requirements.txt` includes the core Flask/data/RAG/evaluation packages used by the code.
- The listed `faiss-cpu` package lets the app run on CPU FAISS. To use GPU FAISS, install a CUDA-compatible FAISS build for the workstation environment.
- `llama-cpp-python` must be built/installed with CUDA support if you want GPU offload for the GGUF model.
- Do not commit Hugging Face tokens, `.env` files, local model files, local DB files, or generated FAISS indexes.

### 3. Optional NLTK setup for readability scoring

```bash
python -m nltk.downloader cmudict
```

---

## Quick demo without rebuilding the full index

This starts the Flask app and allows the UI/auth/database routes to run. If the FAISS index or GGUF model is missing, the app falls back to existing DB recommendations or placeholder story text.

```bash
python scripts/seed_test_users.py
python app.py
```

Open:

```text
http://localhost:5000
```

Demo accounts:

| Username | Password | Level |
|---|---|---|
| `test_beginner` | `test123` | Beginner |
| `test_intermediate` | `test123` | Intermediate |
| `test_advanced` | `test123` | Advanced |

Important: `seed_test_users.py` can create users and vocabulary states with the committed vocab files. It only seeds article recommendations if `outputs/article_stats.jsonl` exists.

---

## Full data pipeline

Use these commands to reproduce the final-style corpus/index workflow.

### 1. Build/verify vocabulary files

```bash
python scripts/build_vocab_lists.py \
  --input data/vocab_sources \
  --output vocab_output \
  --runtime-vocab-dir data/vocab \
  --only-band all

python scripts/build_vocab.py
```

Expected runtime files:

```text
data/vocab/beginner_1000.txt
data/vocab/intermediate_3000.txt
data/vocab/advanced_6000.txt
```

### 2. Download Wikipedia

Small debug run:

```bash
python scripts/download_wiki.py \
  --sample-size 500 \
  --output data/raw/wiki_sample.parquet
```

Full final-style run:

```bash
python scripts/download_wiki.py \
  --sample-size 1000000 \
  --output data/raw/wiki_1m.parquet
```

### 3. Preprocess articles

```bash
python scripts/preprocess_wiki.py \
  --input data/raw/wiki_1m.parquet \
  --output data/processed/wiki_clean.parquet \
  --min-words 150
```

### 4. Tokenize articles

Using a Hugging Face tokenizer:

```bash
python scripts/tokenize_articles.py \
  --input data/processed/wiki_clean.parquet \
  --output data/processed/wiki_tokenized.parquet \
  --tokenizer sentence-transformers/all-MiniLM-L6-v2 \
  --max-length 2048 \
  --batch-size 32
```

Using a local GGUF tokenizer instead:

```bash
python scripts/tokenize_articles.py \
  --input data/processed/wiki_clean.parquet \
  --output data/processed/wiki_tokenized.parquet \
  --gguf-path models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --max-length 2048 \
  --batch-size 32
```

### 5. Analyze vocabulary coverage and readability

```bash
mkdir -p outputs

python scripts/analyze_articles.py \
  --input data/processed/wiki_tokenized.parquet \
  --output outputs/article_stats_1m.jsonl \
  --coverage-min 0.85 \
  --coverage-max 0.97
```

For a smaller committed/sample run, use the sample paths:

```bash
python scripts/analyze_articles.py \
  --input data/processed/wiki_clean.parquet \
  --output outputs/article_stats.jsonl \
  --coverage-min 0.85 \
  --coverage-max 0.97
```

### 6. Check article statistics

```bash
python scripts/check_article_stats.py \
  --input outputs/article_stats_1m.jsonl \
  --window 0.85:0.97 \
  --window 0.90:0.97 \
  --window 0.45:0.70
```

### 7. Build the FAISS index

For the Flask app's default path:

```bash
python scripts/build_index.py \
  --articles outputs/article_stats_1m.jsonl \
  --index-dir data/faiss_index_1m_chunklevel \
  --model all-MiniLM-L6-v2 \
  --chunk-size 400 \
  --overlap 50 \
  --batch-size 256 \
  --device cuda
```

For a smaller local/debug path:

```bash
python scripts/build_index.py \
  --articles outputs/article_stats.jsonl \
  --index-dir data/faiss_index \
  --device cuda
```

The app currently looks for:

```text
data/faiss_index_1m_chunklevel/
```

If you build a different index path, either rename/copy that folder or update `faiss_index_dir` in `app.py`.

---

## Local LLaMA model setup

The Flask app expects the local model at:

```text
models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

The expected model family is:

```text
Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
Bartowski GGUF release
loaded through llama-cpp-python
context window: 4096
n_gpu_layers: -1
```

Create the model directory and place the GGUF file there:

```bash
mkdir -p models
# copy or download the GGUF file into models/
```

The app uses a single visible GPU by default through `CUDA_VISIBLE_DEVICES=0`. The CLI/evaluation scripts support explicit GPU visibility and tensor splitting:

```bash
python scripts/run_rag.py \
  --index-dir data/faiss_index_1m_chunklevel \
  --query "space and planets" \
  --level intermediate \
  --filename models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --cuda-visible-devices 0,1 \
  --tensor-split 0.5 0.5 \
  --n-gpu-layers -1
```

---

## Run the web app

```bash
python scripts/seed_test_users.py
python app.py
```

The server listens on:

```text
http://localhost:5000
```

The app routes include:

| Route | Purpose |
|---|---|
| `POST /api/auth/register` | create an account |
| `POST /api/auth/login` | login |
| `POST /api/auth/logout` | logout |
| `GET /api/profile` | current profile and reading history |
| `GET /api/recommendations` | current saved recommendations |
| `POST /api/recommendations/generate` | run RAG recommendation generation |
| `GET /api/library` | list user library books |
| `POST /api/library` | add a user book |
| `GET /api/books/search?q=...` | search books by title |
| `POST /api/story/generate` | generate a story |
| `POST /api/story/save` | save most recent generated story |
| `GET /api/story/list` | list saved stories |

---

## Command-line RAG tests

Topic-query mode:

```bash
python scripts/run_rag.py \
  --index-dir data/faiss_index_1m_chunklevel \
  --query "space and planets" \
  --level intermediate \
  --top-k 3 \
  --top-broad 100 \
  --filename models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

Profile-driven mode using a seeded user:

```bash
python scripts/run_rag.py \
  --index-dir data/faiss_index_1m_chunklevel \
  --user-id 1 \
  --top-k 3 \
  --filename models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

Simulate vocabulary growth after reading a result:

```bash
python scripts/run_rag.py \
  --index-dir data/faiss_index_1m_chunklevel \
  --query "animals and habitats" \
  --level beginner \
  --simulate-growth \
  --filename models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

---

## Command-line story generation

```bash
python scripts/generate_story.py \
  --level intermediate \
  --topic "space" \
  --genre sci-fi \
  --challenge medium \
  --target-words 400 \
  --max-new-vocab 10 \
  --filename models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

Using a database-backed profile:

```bash
python scripts/generate_story.py \
  --user-id 2 \
  --topic "space" \
  --genre sci-fi \
  --challenge medium \
  --filename models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

---

## Evaluation and ablation

The committed `data/eval_queries.json` contains 15 evaluation queries: 5 beginner, 5 intermediate, and 5 advanced.

### Retrieval-only evaluation

```bash
python scripts/evaluate_rag.py \
  --index-dir data/faiss_index_1m_chunklevel \
  --queries data/eval_queries.json \
  --output outputs/eval_weighted_rerank.json \
  --mode weighted_rerank \
  --skip-generation \
  --top-broad 100
```

### Ablation modes

```bash
python scripts/evaluate_rag.py \
  --index-dir data/faiss_index_1m_chunklevel \
  --queries data/eval_queries.json \
  --output outputs/eval_semantic_only.json \
  --mode semantic_only \
  --skip-generation

python scripts/evaluate_rag.py \
  --index-dir data/faiss_index_1m_chunklevel \
  --queries data/eval_queries.json \
  --output outputs/eval_coverage_filter.json \
  --mode coverage_filter \
  --skip-generation

python scripts/evaluate_rag.py \
  --index-dir data/faiss_index_1m_chunklevel \
  --queries data/eval_queries.json \
  --output outputs/eval_weighted_rerank.json \
  --mode weighted_rerank \
  --skip-generation
```

### Generation metrics

```bash
python scripts/evaluate_rag.py \
  --index-dir data/faiss_index_1m_chunklevel \
  --queries data/eval_queries.json \
  --output outputs/eval_generation.json \
  --mode weighted_rerank \
  --with-bertscore \
  --filename models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --generation-top-k 3
```

The evaluation script reports:

- Precision@5, Precision@10, Precision@20
- Recall@5, Recall@10, Recall@20
- MRR
- percentage of top results inside the target coverage window
- ROUGE-1, ROUGE-2, ROUGE-L when generation is enabled
- optional BERTScore precision/recall/F1
- structured output validity rate

---

## Final report alignment

The final report describes the full intended run with approximately:

| Pipeline stage | Count |
|---|---:|
| Downloaded Wikipedia articles | 1,000,000 |
| Cleaned usable articles | 628,652 |
| Word-tokenized articles | 628,652 |
| Subword-tokenized articles | 628,652 |
| Generated article chunks | 1,682,652 |
| FAISS vectors | 1,682,652 |

The submitted zip contains a small sample parquet and final source code, not the full generated data/index/model artifacts. Rebuild or copy the external artifacts before running the full RAG demo.

The final report's retrieval table lists these retrieval metrics for the completed evaluation run:

| Metric | @5 | @10 | @20 |
|---|---:|---:|---:|
| Precision@k | 0.2533 | 0.2067 | 0.1600 |
| Recall@k | 0.2533 | 0.4133 | 0.6400 |
| MRR | 0.5534 | - | - |

If additional evaluation runs are completed, update the report placeholders and README metrics together so the report, code, and repository stay consistent.

---

## Important implementation notes

### Pretrained LLaMA, not QLoRA training

The code uses `LlamaCppGenerator` for inference with a pretrained/instruction-tuned GGUF model. There are no project-specific QLoRA training scripts, adapter checkpoints, training epochs, optimizer states, or LoRA merge steps in this repository.

### FAISS CPU/GPU behavior

`FAISSVectorStore` uses `IndexFlatIP` over normalized vectors. It attempts GPU FAISS if available and falls back to CPU FAISS if GPU functions are unavailable.

### App first-request latency

The app lazily loads the RAG pipeline and LLaMA generator. The first recommendation/story request can be slow because it may initialize:

- FAISS index and `chunks.pkl`
- sentence-transformer embedder
- cross-encoder reranker
- local LLaMA GGUF model

For demos, start the app early or run one warm-up request before presenting.

### Large index bottlenecks

For the full million-article index, the main bottlenecks are expected to be:

- loading large chunk metadata from pickle
- GPU transfer time for FAISS index
- cross-encoder reranking over broad candidate pools
- first LLaMA model load
- Flask configured with `threaded=False`

Practical demo optimizations:

- keep `top_broad` near 100 or lower
- use `--no-cross-encoder` for faster CLI tests
- warm the model/index before the demo
- consider approximate FAISS search for larger deployments
- move chunk metadata into SQLite or memory-mapped storage later

---

## Troubleshooting

### `FAISS index not found`

The web app expects:

```text
data/faiss_index_1m_chunklevel/
```

Build it with `scripts/build_index.py`, copy it into that path, or edit `app.py` to point to your local index directory.

### Story generation returns placeholder text

The GGUF model is missing or `llama-cpp-python` is not configured correctly. Confirm this file exists:

```text
models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

Then verify llama-cpp-python can load it.

### CUDA/GPU is not being used

Check GPU visibility:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

For CLI tests, explicitly pass:

```bash
--cuda-visible-devices 0
```

or for two visible GPUs:

```bash
--cuda-visible-devices 0,1 --tensor-split 0.5 0.5
```

### `ModuleNotFoundError`

Install requirements inside the active virtual environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### `cmudict` or readability errors

```bash
python -m nltk.downloader cmudict
```

### Hugging Face rate-limit or gated model issues

Use Hugging Face authentication only on your machine/environment:

```bash
hf auth login
```

Never commit tokens into the repository.

---

## Suggested final submission checklist

Before submitting or demoing:

```bash
python -m py_compile $(find . -name '*.py' -not -path './.venv/*')
python scripts/build_vocab.py
python scripts/seed_test_users.py
python app.py
```

Also verify that these external artifacts exist for the full RAG demo:

```text
models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
data/faiss_index_1m_chunklevel/index.faiss
data/faiss_index_1m_chunklevel/chunks.pkl
data/faiss_index_1m_chunklevel/meta.json
outputs/article_stats_1m.jsonl
```

---

## Team contribution summary

- Frontend and UI: login page, book-style main page, profile display, library controls, and styling.
- Data pipeline: Wikipedia download, preprocessing, tokenization, vocabulary analysis, and corpus preparation.
- Backend and RAG: Flask routes, SQLite schema/repositories, FAISS vector store, retrieval/reranking, LLaMA generation, story generation, evaluation scripts, and GPU inference testing.

---

## License / academic use

This repository is an academic CS 322 project. Model files, Wikipedia data, and Hugging Face resources may have their own licenses or access restrictions. Keep large external artifacts and private tokens out of git.
