from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from assistant.tools import calculator
from assistant.model import get_model

@tool 
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    return calculator(expression)

class ResearchAgent:
    def __init__(self) -> None:
        self.model = get_model()
        self.agent = self.model.bind_tools([calculate])


    def ask(self, question: str):

        messages = [("human", question)]

        for _ in range(5):
            response = self.agent.invoke(messages)

            if not response.tool_calls:
                return response.content

            messages.append(response)

            for call in response.tool_calls:
                result = calculate.invoke(call["args"])

                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": str(result)
                })

        return "Maximum tool iterations reached."




        