from langchain_core.tools import tool
from assistant.model import get_model
from assistant.rag import ResearchRAG

class ResearchAgent:
    def __init__(self):
        self.model = get_model()
        self.rag = ResearchRAG()

    def research(self, question: str):
        result = self.rag.ask(question)
        return result