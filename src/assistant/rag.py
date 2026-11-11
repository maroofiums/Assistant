
from assistant.embedding_index import EmbeddingIndex
from assistant.model import get_model


class ResearchRAG:

    def __init__(self):
        self.index = EmbeddingIndex()
        self.index.load()

        self.model = get_model()

    def ask(self, question: str, k: int = 4):

        results = self.index.search(question, k)

        context = "\n\n".join(
            f"[Page {r['document'].metadata.get('page', '?')}]\n"
            f"{r['document'].page_content}"
            for r in results
        )

        prompt = f"""
        Answer using only this research context.
        If unavailable, say so.
        Cite page numbers.

        Context:
        {context}

        Question: {question}
        """

        response = self.model.invoke(prompt)

        return response.text
