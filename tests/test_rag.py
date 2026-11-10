from assistant.rag import ResearchRAG

rag = ResearchRAG()

result = rag.ask(
    "What is the main contribution of this research paper?"
)

print("\nANSWER:\n")
print(result["answer"])

print("\nSOURCES:\n")

for doc in result["sources"]:
    print("Page:", doc.metadata.get("page"))
    print(doc.page_content[:200])
    print("-" * 50)