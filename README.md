# TREC-COVID Information Retrieval System
## Overview
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

## Phase 3: Semantic Search

Dense retrieval using Sentence Transformers and FAISS vector database.
Unlike BM25 which matches exact keywords, semantic search understands 
the meaning of the query and finds conceptually related documents.

### Model

After testing multiple models, `all-MiniLM-L6-v2` was selected over 
`allenai/specter` due to better retrieval performance on content-based 
queries. Despite specter being specialized for scientific papers, 
MiniLM showed superior results for meaning-based search tasks.

| Model | Type | Size | Performance |
|-------|------|------|-------------|
| allenai/specter | Citation-based | 440MB | Poor on content queries |
| all-MiniLM-L6-v2 | Semantic | 80MB | Good on content queries |

### How it works

1. Encode all 171,332 articles (title + text) into 384-dimensional vectors
2. Normalize vectors using L2 normalization for cosine similarity
3. Store vectors in FAISS IndexFlatIP for fast similarity search
4. For each query, encode and normalize the query vector
5. FAISS returns top-k most similar documents

### Components

| File | Description |
|------|-------------|
| semantic_retriever.py | Encode, index, search, save, load |

### Search Output

Each result contains:
- `_id`: Document ID
- `title`: Article title
- `abstract`: First 300 characters of text
- `score`: Cosine similarity score (0 to 1, higher is better)

### Sample Results

Query: "what is the origin of COVID-19"

| Title | Score |
|-------|-------|
| Strategies to trace back the origin of COVID-19 | 0.879 |
| What is COVID-19? | 0.818 |
| COVID-19 | 0.784 |

## Phase 4: Hybrid Retrieval

Combines BM25 and Semantic Search to leverage the strengths of both methods.
BM25 excels at exact keyword matching while Semantic Search understands meaning.
Hybrid retrieval produces better results than either method alone.

### How it works

1. Both retrievers are called with top_k*2 candidates
2. BM25 scores are normalized (divided by max score)
3. Final score is computed using weighted sum:

final_score = (bm25_normalized * 0.3) + (semantic_score * 0.7)

4. Results are sorted by final score

### Why 0.7 weight for Semantic?

| Method | Weight | Reason |
|--------|--------|--------|
| BM25 | 0.3 | Exact keywords matter but are not sufficient |
| Semantic | 0.7 | Meaning-based retrieval is more effective for scientific articles |

### Results Comparison

Query: "what is the origin of COVID-19"

| Method | Top Result | Score |
|--------|-----------|-------|
| BM25 | The impact of COVID-19 ... | 17.5 |
| Semantic | Strategies to trace back the origin of COVID-19 | 0.879 |
| Hybrid | Strategies to trace back the origin of COVID-19 | 0.615 |

### Components

| File | Description |
|------|-------------|
| hybrid_retriever.py | Combines BM25 and Semantic, search |