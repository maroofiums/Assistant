
from assistant.embedding_index import EmbeddingIndex
from langchain_core.documents import Document


chunks = [
    Document(
        page_content="Transformers use self-attention mechanisms.",
        metadata={"page": 1}
    ),
    Document(
        page_content="Neural networks learn through optimization.",
        metadata={"page": 2}
    ),
]

# Build and save
index = EmbeddingIndex()
index.build(chunks)
index.save()

# Load into a fresh instance
loaded_index = EmbeddingIndex()
loaded_index.load()

# Search
results = loaded_index.search(
    "How does self-attention work?",
    k=2
)

for result in results:
    print("Score:", result["score"])
    print("Document:", result["document"].page_content)
    print("Metadata:", result["document"].metadata)
