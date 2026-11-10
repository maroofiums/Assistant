import faiss
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class EmbeddingIndex:

    def __init__(self):
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": False},
        )

        self.index = None
        self.documents = []

    def build(self, chunks: list[Document]):

        if not chunks:
            raise ValueError("No document chunks provided.")

        # 1. Extract text
        texts = [chunk.page_content for chunk in chunks]

        # 2. Convert text into embeddings
        embeddings = self.embedder.embed_documents(texts)

        # 3. Convert embeddings into NumPy float32 matrix
        vectors = np.array(embeddings, dtype="float32")

        # 4. Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)

        # 5. Create FAISS index
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        # 6. Add vectors to index
        self.index.add(vectors)

        # Keep original documents for retrieving text
        self.documents = chunks

        print("Index created successfully.")
        print("Vector dimension:", dimension)
        print("Indexed vectors:", self.index.ntotal)

    def search(self, query: str, k: int = 3):

        if self.index is None:
            raise RuntimeError("Index has not been built.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        # Embed query
        query_vector = self.embedder.embed_query(query)

        query_vector = np.array(
            [query_vector],
            dtype="float32"
        )

        # Normalize query
        faiss.normalize_L2(query_vector)

        # Search nearest vectors
        scores, indices = self.index.search(query_vector, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            results.append({
                "score": float(score),
                "document": self.documents[idx]
            })

        return results