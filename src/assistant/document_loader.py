from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    """Load a PDF and extract its page"""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not Found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    if not documents:
        raise ValueError("No content extracted from PDF.")


    return documents