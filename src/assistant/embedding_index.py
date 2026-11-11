
import pickle
from pathlib import Path

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

        texts = [chunk.page_content for chunk in chunks]

        embeddings = self.embedder.embed_documents(texts)

        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

        dimension = vectors.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)

        self.documents = chunks

        print("Index created successfully.")
        print("Vector dimension:", dimension)
        print("Indexed vectors:", self.index.ntotal)

    def search(self, query: str, k: int = 3):

        if self.index is None:
            raise RuntimeError("Index has not been built.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if k < 1:
            raise ValueError("k must be at least 1.")

        query_vector = self.embedder.embed_query(query)

        query_vector = np.array(
            [query_vector],
            dtype=np.float32
        )

        faiss.normalize_L2(query_vector)

        k = min(k, self.index.ntotal)

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

    def save(self, path="data/vector_store"):

        if self.index is None:
            raise RuntimeError("No index to save.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(path / "index.faiss")
        )

        with open(path / "documents.pkl", "wb") as file:
            pickle.dump(self.documents, file)

        print(f"Index saved: {path}")

    def load(self, path="data/vector_store"):

        path = Path(path)

        index_path = path / "index.faiss"
        documents_path = path / "documents.pkl"

        if not index_path.exists() or not documents_path.exists():
            raise FileNotFoundError(
                f"Index files not found in: {path}"
            )

        self.index = faiss.read_index(str(index_path))

        with open(documents_path, "rb") as file:
            self.documents = pickle.load(file)

        if self.index.ntotal != len(self.documents):
            raise ValueError(
                "Index and document count do not match."
            )

        print(f"Index loaded: {path}")
        print("Indexed vectors:", self.index.ntotal)

        return self
