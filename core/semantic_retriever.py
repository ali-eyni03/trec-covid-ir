from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2
import pickle
import faiss
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HF_TOKEN")

MODEL_PATH = "./models/all-MiniLM-L6-v2"


class SemanticRetriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_PATH, local_files_only=True)
        self.index = None
        self.docs = []

    def build_index(self, docs):
        '''Encode articles and put them in FAISS (Facebook AI Similarity Search)'''
        self.docs = docs
        texts = []
        for doc in docs:
            texts.append(f"{doc['title']} {doc['text']}")
        embedding_model = self.model.encode(texts, show_progress_bar=True)
        embedding = embedding_model.astype("float32")  # convert to float32 for FAISS
        faiss.normalize_L2(embedding)  # normalize embeddings for cosine similarity
        index = faiss.IndexFlatIP(embedding.shape[1])
        index.add(embedding)
        self.index = index

    def search(self, query, top_k=5):
        '''Encode query and find the nearest neighbors in the vector store'''
        query_embedding = self.model.encode([query], show_progress_bar=True)
        query_embedding = query_embedding.astype(
            "float32"
        )  # convert to float32 for FAISS
        faiss.normalize_L2(
            query_embedding
        )  # normalize query embedding for cosine similarity
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            results.append(
                {
                    "_id": self.docs[idx]["_id"],
                    "title": self.docs[idx]["title"],
                    "text": (
                        self.docs[idx].get("text", "")
                        or self.docs[idx].get("abstract", "No abstract available")
                    )[:300],
                    "score": float(distance),
                }
            )
        return results

    def save_index(self, file_path):
        '''Save the FAISS index to disk'''
        faiss.write_index(self.index, file_path)
        with open(file_path + ".docs.pkl", "wb") as f:
            pickle.dump(self.docs, f)

    def load_index(self, file_path):
        '''Load the FAISS index from disk'''
        self.index = faiss.read_index(file_path)
        with open(file_path + ".docs.pkl", "rb") as f:
            self.docs = pickle.load(f)
