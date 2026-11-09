from assistant.document_loader import load_pdf

documents = load_pdf(
    "data/research_papers/attention_is_all_you_need.pdf"
)

print("Total pages:", len(documents))
print("\nFirst page:")
print(documents[0].page_content[:1000])

print("\nMetadata:")
print(documents[0].metadata)
