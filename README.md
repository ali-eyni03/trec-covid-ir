# TREC-COVID Information Retrieval System

A hybrid information retrieval system on TREC-COVID dataset 
combining BM25 and Semantic Search (FAISS + Sentence Transformers), 
served via FastAPI.

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

- Corpus fields: _id, title, text, metadata
- Relevance scores: 0 (not relevant), 1 (relevant), 2 (highly relevant)