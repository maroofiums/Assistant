
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class ResearchVectorStore:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_store = None

    def build(self, chunks: list[Document]):

        if not chunks:
            raise ValueError("No document chunks provided.")

        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )

        return self.vector_store

    def save(self, path="data/vector_store"):

        if self.vector_store is None:
            raise RuntimeError("No vector index to save.")

        Path(path).mkdir(parents=True, exist_ok=True)

        self.vector_store.save_local(path)

        print(f"Vector store saved: {path}")

    def load(self, path="data/vector_store"):

        if not Path(path).exists():
            raise FileNotFoundError(f"Index not found: {path}")

        self.vector_store = FAISS.load_local(
            folder_path=path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

        print(f"Vector store loaded: {path}")

        return self.vector_store

    def search(self, query: str, k: int = 3):

        if self.vector_store is None:
            raise RuntimeError("Vector store is not initialized.")

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        return self.vector_store.similarity_search(
            query=query,
            k=k,
        )
