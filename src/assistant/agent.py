from langchain_core.messages import HumanMessage, SystemMessage

from assistant.model import get_model


class ResearchAssistant:
    
    def __init__(self):
        
        self.model = get_model()
        
        self.system_prompt = """
        You are an AI Research Assistant.

        Your responsibilities:
        - Explain technical and scientific concepts.
        - Analyze research questions.
        - Provide structured and accurate answers.
        - Clearly distinguish facts from assumptions.
        - Admit when you don't know something.

        Keep responses clear and concise.
        """

    def ask(self, question: str) -> str:
        """
        Ask a question to the research assistant.

        Args:
            question (str): The question to ask.

        Returns:
            str: The assistant's response.
        """

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=question)
        ]
        
        response = self.model.invoke(messages)
        
        return response.content
    