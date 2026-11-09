from assistant.document_loader import load_pdf
from assistant.text_splitter import split_documents
from assistant.vector_store import ResearchVectorStore

# Load PDF
documents = load_pdf(
    "data/research_papers/attention_is_all_you_need.pdf"
)

# Split into chunks
chunks = split_documents(documents)

# Build vector index
store = ResearchVectorStore()
store.build(chunks)

print("Indexed chunks:", len(chunks))

# Semantic search
query = "What is the main contribution of this research?"

results = store.search(query, k=3)

for i, document in enumerate(results, 1):
    print(f"\nResult {i}")
    print("Page:", document.metadata.get("page"))
    print("Content:", document.page_content[:300])
