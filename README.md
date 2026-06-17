# TREC-COVID Information Retrieval System

Information Retrieval system built on the TREC-COVID dataset using three retrieval approaches:

- BM25 (Sparse Retrieval)
- Semantic Search (Dense Retrieval)
- Hybrid Retrieval

## Features

- BM25 keyword-based retrieval
- Semantic retrieval with Sentence Transformers
- FAISS vector indexing
- Hybrid ranking (BM25 + Semantic)
- FastAPI backend
- Evaluation on TREC-COVID benchmark
- Modular and extensible architecture

---

## Dataset

This project uses the TREC-COVID dataset provided through the BEIR framework.

| Item | Count |
|--------|--------|
| Documents | 171,332 |
| Queries | 50 |
| Relevance Judgments (Qrels) | 66,336 |

### Dataset Files

| File | Description |
|------|-------------|
| `corpus.jsonl` | Scientific articles |
| `queries.jsonl` | Search queries |
| `qrels/test.tsv` | Relevance labels |

### Relevance Labels

| Score | Meaning |
|--------|----------|
| 0 | Not Relevant |
| 1 | Relevant |
| 2 | Highly Relevant |

---

## Tech Stack

- Python
- FastAPI
- FAISS
- Sentence Transformers
- Rank-BM25
- BEIR

---

## Project Structure

```text
trec-covid-ir/
├── api/
├── core/
├── data/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd trec-covid-ir
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Phase 1 — Data Collection & EDA

The dataset was downloaded using the BEIR framework and inspected to understand:

- Corpus structure
- Query format
- Relevance judgments
- Data statistics

---

# Phase 2 — BM25 Retriever

Classical sparse retrieval using the BM25Okapi algorithm.

### Pipeline

1. Text preprocessing
2. Tokenization
3. BM25 index construction
4. Query scoring
5. Top-k ranking

### Components

| File | Responsibility |
|--------|----------------|
| `preprocessor.py` | Cleaning & tokenization |
| `bm25_retriever.py` | Indexing and retrieval |

### Output

Each result contains:

```json
{
  "_id": "...",
  "title": "...",
  "abstract": "...",
  "score": 17.53
}
```

---

# Phase 3 — Semantic Search

Dense retrieval using Sentence Transformers and FAISS.

Unlike BM25, semantic search retrieves documents based on meaning rather than exact keyword matches.

## Selected Model

After experimentation, `all-MiniLM-L6-v2` achieved better retrieval quality than `allenai/specter`.

| Model | Size | Notes |
|---------|---------|---------|
| allenai/specter | 440 MB | Citation-oriented |
| all-MiniLM-L6-v2 | 80 MB | Better semantic retrieval |

### Pipeline

1. Encode documents
2. Normalize embeddings
3. Store in FAISS
4. Encode query
5. Retrieve nearest neighbors

### Components

| File | Responsibility |
|--------|----------------|
| `semantic_retriever.py` | Embedding, indexing, retrieval |

### Output

```json
{
  "_id": "...",
  "title": "...",
  "abstract": "...",
  "score": 0.879
}
```

### Example Query

**Query**

```text
what is the origin of COVID-19
```

**Top Results**

| Title | Score |
|---------|---------|
| Strategies to trace back the origin of COVID-19 | 0.879 |
| What is COVID-19? | 0.818 |
| COVID-19 | 0.784 |

---

# Phase 4 — Hybrid Retrieval

Combines sparse and dense retrieval to benefit from both:

- BM25 → keyword matching
- Semantic Search → meaning matching

## Scoring Formula

```python
final_score = (
    bm25_normalized * 0.3
    + semantic_score * 0.7
)
```

### Weight Selection

| Method | Weight |
|----------|----------|
| BM25 | 0.3 |
| Semantic | 0.7 |

Semantic retrieval was given higher importance because it performed better on scientific search tasks.

### Components

| File | Responsibility |
|--------|----------------|
| `hybrid_retriever.py` | Hybrid ranking |

### Example Comparison

| Method | Top Result |
|---------|------------|
| BM25 | The impact of COVID-19... |
| Semantic | Strategies to trace back the origin of COVID-19 |
| Hybrid | Strategies to trace back the origin of COVID-19 |

---

## Future Work

- Query expansion
- Cross-Encoder reranking
- Evaluation metrics (NDCG, MAP, Recall@K)
- Frontend UI
- Docker deployment

---

## License

MIT License