# TREC-COVID Information Retrieval System
b## Overview
- **Dataset**: TREC-COVID (171,332 articles, 50 queries)
- **Methods**: BM25, Semantic Search, Hybrid
- **Stack**: FastAPI, FAISS, Sentence Transformers, Python

## Project Structure
trec-covid-ir/
├── data/
├── core/
├── api/
├── notebooks/
└── requirements.txt

## Installation
pip install -r requirements.txt

## Phases
- Phase 1: Data Collection & EDA
- Phase 2: BM25 Retriever
- Phase 3: Semantic Search
- Phase 4: Hybrid Retriever
- Phase 5: FastAPI
- Phase 6: Evaluation & UI

## Phase 1: Data Collection & EDA
Downloaded TREC-COVID dataset using BEIR framework.
Dataset contains 171,332 scientific articles, 50 queries, 
and 66,336 relevance judgments (qrels).

| File | Description |
|------|-------------|
| corpus.jsonl | 171,332 scientific articles |
| queries.jsonl | 50 medical queries |
| qrels/test.tsv | 66,336 relevance judgments |


- Corpus fields: _id, title, text, metadata
- Relevance scores: 0 (not relevant), 1 (relevant), 2 (highly relevant)


Relevance scores in qrels:

| Score | Meaning |
|-------|---------|
| 0 | Not relevant |
| 1 | Relevant |
| 2 | Highly relevant |

---

## Phase 2: BM25 Retriever

Classical keyword-based retrieval using the BM25Okapi algorithm.
Builds an inverted index over the corpus and ranks documents
based on term frequency and inverse document frequency.

### How it works

1. Text preprocessing: lowercase, remove punctuation, tokenize
2. Build BM25 index over tokenized corpus
3. For each query, compute BM25 scores across all documents
4. Return top-k documents sorted by score

### Components

| File | Description |
|------|-------------|
| preprocessor.py | Text cleaning and tokenization |
| bm25_retriever.py | BM25 index build, search, save, load |

### Search Output

Each result contains:
- `_id`: Document ID
- `title`: Article title  
- `abstract`: First 300 characters of text
- `score`: BM25 relevance score