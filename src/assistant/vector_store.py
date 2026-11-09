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
            raise ValueError("Chunks cannot be empty")

        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        return self.vector_store

    def search(self, query: str, k: int = 3):

        if self.vector_store is None:
            raise RuntimeError("Vector store has not been initialized.")

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        return self.vector_store.similarity_search(
            query=query,
            k=k,
        )
