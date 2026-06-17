import json
import sys
import os
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ranx import Qrels, Run, evaluate
from core.hybrid_retriever import HybridRetriever

# 1-Load Data
def load_queries(path: str) -> dict:
    """Load queries from queries.jsonl ({query_id: query_text})"""
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            queries[str(obj["_id"])] = obj["text"]
    return queries

def load_qrels(path: str) -> dict:
    """
    Load relevance judgments from qrels/test.tsv
    Format: query_id  corpus_id  score
    Returns: {query_id: {doc_id: relevance_score}}
    """
    qrels_dict = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # skip header line
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                query_id = str(parts[0])
                doc_id = str(parts[1])
                score = int(parts[2])
                qrels_dict[query_id][doc_id] = score
    return dict(qrels_dict)


# 2-Run Retrieval for All Queries
def run_retrieval(retriever, queries: dict, method: str, top_k: int = 10) -> dict:
    """
    Run retrieval for all queries.
    Returns: {query_id: {doc_id: score}}
    """
    run_dict = {}
    total = len(queries)

    for i, (qid, qtext) in enumerate(queries.items(), 1):
        print(f"  [{i}/{total}] Query: {qtext[:60]}...")

        if method == "bm25":
            results = retriever.bm25_retriever.search(qtext, top_k)
        elif method == "semantic":
            results = retriever.semantic_retriever.search(qtext, top_k)
        elif method == "hybrid":
            results = retriever.search(qtext, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")

        run_dict[qid] = {str(r["_id"]): float(r["score"]) for r in results}

    return run_dict


# 3-Evaluate & Print Results
def print_results_table(results: dict):
    """Print a formatted comparison table."""
    metrics = ["ndcg@10", "map@10"]
    methods = list(results.keys())

    col_w = 15
    header = f"{'Method':<12}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print("\n" + "=" * (12 + col_w * len(metrics)))
    print("  EVALUATION RESULTS — TREC-COVID")
    print("=" * (12 + col_w * len(metrics)))
    print(header)
    print("-" * (12 + col_w * len(metrics)))

    for method in methods:
        row = f"{method:<12}"
        for metric in metrics:
            val = results[method].get(metric, 0.0)
            row += f"{val:>{col_w}.4f}"
        print(row)

    print("=" * (12 + col_w * len(metrics)))


def main():
    DATA_ROOT = Path("data")
    QUERIES = DATA_ROOT / "raw/trec-covid/queries.jsonl"
    QRELS = DATA_ROOT / "raw/trec-covid/qrels/test.tsv"
    BM25_IDX = DATA_ROOT / "processed/bm25_index.pkl"
    FAISS_IDX = DATA_ROOT / "processed/semantic_index.faiss"
    TOP_K = 10

    print("Loading indexes...")
    retriever = HybridRetriever()
    retriever.bm25_retriever.load_index(str(BM25_IDX))
    retriever.semantic_retriever.load_index(str(FAISS_IDX))
    print("Indexes loaded.\n")

    print("Loading queries and qrels...")
    queries = load_queries(str(QUERIES))
    qrels_dict = load_qrels(str(QRELS))
    print(f"  Queries : {len(queries)}")
    print(f"  Qrel judgments: {sum(len(v) for v in qrels_dict.values())}\n")

    # Filter queries that have qrel judgments
    queries = {qid: qtext for qid, qtext in queries.items() if qid in qrels_dict}
    print(f"  Queries with qrels: {len(queries)}\n")

    qrels = Qrels(qrels_dict)

    # Evaluate Each Method
    all_results = {}
    metrics = ["ndcg@10", "map@10"]

    for method in ["bm25", "semantic", "hybrid"]:
        print(f"Running {method.upper()}...")
        run_dict = run_retrieval(retriever, queries, method, TOP_K)
        run = Run(run_dict, name=method)
        scores = evaluate(qrels, run, metrics)
        all_results[method] = scores
        print(
            f"  nDCG@10 = {scores['ndcg@10']:.4f}  |  MAP@10 = {scores['map@10']:.4f}\n"
        )

    print_results_table(all_results)

    # Save Results
    out_path = Path("evaluation/results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
