from assistant.document_loader import load_pdf
from assistant.text_splitter import split_documents
from assistant.embedding_index import EmbeddingIndex

documents = load_pdf(
    "data/research_papers/attention_is_all_you_need.pdf"
)

chunks = split_documents(documents)

index = EmbeddingIndex()
index.build(chunks)

results = index.search(
    "What is the main contribution of this paper?",
    k=3
)

for result in results:
    print("\nScore:", result["score"])

    doc = result["document"]

    print("Page:", doc.metadata.get("page"))
    print("Content:", doc.page_content[:300])