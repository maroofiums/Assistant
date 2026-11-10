
from assistant.document_loader import load_pdf
from assistant.text_splitter import split_documents
from assistant.vector_store import ResearchVectorStore

# Build and save
documents = load_pdf(
    "data/research_papers/attention_is_all_you_need.pdf"
)

chunks = split_documents(documents)

store = ResearchVectorStore()
store.build(chunks)
store.save()

# Load saved index
new_store = ResearchVectorStore()
new_store.load()

# Search without re-indexing
results = new_store.search(
    "What is the main contribution of this paper?",
    k=3,
)

for result in results:
    print("\nPage:", result.metadata.get("page"))
    print(result.page_content[:300])
