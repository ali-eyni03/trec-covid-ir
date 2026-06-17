import subprocess
import sys
from core.text_preprocessor import TextPreProcessor
from core.bm25_retriever import BM25Retriever
from core.semantic_retriever import SemanticRetriever


def download_data():
    """Download the TREC-COVID dataset and unzip it."""
    subprocess.run([sys.executable, "download_data.py"])
    print("Data downloaded and unzipped successfully.")


def build_indexes():
    """Build BM25 and Semantic indexes from the raw corpus and save them to disk."""
    print("\nLoading corpus...")
    preprocessor = TextPreProcessor()
    docs = preprocessor.load_corpus("data/raw/trec-covid/corpus.jsonl")
    print(f"Loaded {len(docs)} documents.\n")

    print("Building BM25 index...")
    bm25 = BM25Retriever()
    bm25.build_index(docs)
    bm25.save_index("data/processed/bm25_index.pkl")
    print("BM25 index saved to data/processed/bm25_index.pkl\n")

    print("Building Semantic index (this can take a while)...")
    semantic = SemanticRetriever()
    semantic.build_index(docs)
    semantic.save_index("data/processed/semantic_index.faiss")
    print("Semantic index saved to data/processed/semantic_index.faiss\n")

    print("Done! Both indexes are ready.")


def run_evaluation():
    """Run the evaluation script on the 50 TREC-COVID queries."""
    subprocess.run([sys.executable,"-m" "evaluation.evaluate"])


def start_api():
    """Start the FastAPI server with uvicorn."""
    subprocess.run([sys.executable, "-m", "uvicorn", "api.main:app", "--reload"])


def print_menu():
    print("\n\n   TREC-COVID Information Retrieval")
    print("=" * 50)
    print("1-Download TREC-COVID dataset")
    print("2-Build indexes (BM25 + Semantic)")
    print("3-Run evaluation (nDCG@10, MAP)")
    print("4-Start API server")
    print("5-Exit")
    print("=" * 50)


def main():
    while True:
        print_menu()
        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            download_data()
        elif choice == "2":
            build_indexes()
        elif choice == "3":
            run_evaluation()
        elif choice == "4":
            start_api()
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")


if __name__ == "__main__":
    main()