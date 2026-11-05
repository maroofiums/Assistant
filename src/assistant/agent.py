
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)

from assistant.model import get_model


class ResearchAssistant:

    def __init__(self):
        self.model = get_model()
        self.history = []

        self.system_prompt = """
        You are an AI Research Assistant.

        - Explain technical and scientific concepts.
        - Analyze research questions.
        - Provide accurate, structured answers.
        - Distinguish facts from assumptions.
        - Admit uncertainty when necessary.
        """

    def ask(self, question: str) -> str:

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        messages = [
            SystemMessage(content=self.system_prompt),
            *self.history,
            HumanMessage(content=question),
        ]

        response = self.model.invoke(messages)

        answer = response.text

        self.history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer),
        ])

        return answer
