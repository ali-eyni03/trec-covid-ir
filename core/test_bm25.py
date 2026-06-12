from .preprocessor import TextPreProcessor
from .bm25_retriever import BM25Retriever
preprocessor = TextPreProcessor()
docs = preprocessor.load_corpus('data/raw/trec-covid/corpus.jsonl',limit=100)
retriever = BM25Retriever()
retriever.build_index(docs)
retriever.save_index('data/processed/bm25_index.pkl')
retriever2 = BM25Retriever()
retriever2.load_index('data/processed/bm25_index.pkl')
query = "what is the origin of COVID-19"
results = retriever.search(query, top_k=5)
for result in results:
    print(result['title'])
    print(result['score'])
    print('---')
print("\n\n status of loaded file: ")
results2 = retriever2.search(query, top_k=5)
for result in results2: 
    print(result['title'])
    print(result['score'])
    print('---')