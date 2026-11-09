from assistant.document_loader import load_pdf
from assistant.text_splitter import split_documents

documents = load_pdf(
    "data/research_papers/attention_is_all_you_need.pdf"
)

chunks = split_documents(documents)

print("Original pages:", len(documents))
print("Total chunks:", len(chunks))

for chunk in chunks[:3]:
    print("\nMetadata:", chunk.metadata)
    print("Content:", chunk.page_content[:200])
