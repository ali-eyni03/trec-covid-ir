from rank_bm25 import BM25Okapi
from core.text_preprocessor import TextPreProcessor
import pickle

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.docs = []
        self.preprocessor = TextPreProcessor()
        
    def build_index(self, docs):
        # Build the BM25 index from the corpus
        self.docs = docs
        tokens_list = []
        for doc in self.docs:
            title_and_text = f'{doc["title"]} {doc["text"]}'
            tokens_list.append(self.preprocessor.clean(title_and_text))
        self.bm25 =  BM25Okapi(tokens_list)
    
    def search(self , query , top_k):
        # Implement the BM25 search algorithm to retrieve relevant documents
        tokenized_query = self.preprocessor.clean(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_k_articles = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_k_articles:
            results.append({
                '_id': self.docs[idx]['_id'],
                'title': self.docs[idx]['title'],
                'text': self.docs[idx]['text'][:150]+'...',  # Show a snippet of the text
                'score': scores[idx]
            })
        return results
    
    def save_index(self, file_path):
        # Save the BM25 index to disk
        data = {
            'bm25': self.bm25,
            'docs': self.docs
        }
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    def load_index(self, file_path):
        # Load the BM25 index from disk
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.bm25 = data['bm25']
            self.docs = data['docs']
