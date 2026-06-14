from .bm25_retriever import BM25Retriever
from .semantic_retriever import SemanticRetriever


class HybridRetriever:
    def __init__(self):
        self.bm25_retriever = BM25Retriever()
        self.semantic_retriever = SemanticRetriever()

    def search(self, query, top_k=5):
        bm25_result = self.bm25_retriever.search(query, top_k*2)
        semantic_result = self.semantic_retriever.search(query, top_k*2)
        
        #normalize BM25
        max_bm25 = max([result['score'] for result in bm25_result]) if bm25_result else 1
        
        scores = {}

        for result in bm25_result:
            doc_id = result['_id']
            scores[doc_id] = {
                'bm25_score': (result['score'] / max_bm25)*0.3,
                'semantic_score': 0,
                'title': result['title'],
                'text': result['text']
            }
        
        for result in semantic_result:
            doc_id = result['_id']
            if doc_id in scores:
                scores[doc_id]['semantic_score'] = result['score']*0.7
            else:
                scores[doc_id] = {
                    'bm25_score': 0,
                    'semantic_score': result['score']*0.7,
                    'title': result['title'],
                    'text': result['text']
                }
        # final score (sorted)
        combined = []
        for doc_id, score in scores.items():
            combined.append({
                '_id': doc_id,
                'title': score['title'],
                'text': score['text'],
                'score': score['bm25_score'] + score['semantic_score']
            })
        combined.sort(key=lambda x: x['score'], reverse=True)
        return combined[:top_k]