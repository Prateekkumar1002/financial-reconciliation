import faiss
import numpy as np

class FAISSIndex:

    def __init__(self, embeddings):
        self.embeddings = embeddings.astype("float32")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def search(self, query_embeddings, k=10):
        query_embeddings = query_embeddings.astype("float32")
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices