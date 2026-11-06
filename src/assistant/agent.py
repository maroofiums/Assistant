
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from assistant.model import get_model


class ResearchAssistant:

    def __init__(self):
        self.model = get_model()
        self.sessions = {}


        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are an AI Research Assistant.

                - Explain technical and scientific concepts.
                - Analyze research questions.
                - Provide accurate, structured answers.
                - Distinguish facts from assumptions.
                - Admit uncertainty when necessary.
                """
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        chain = prompt | self.model

        self.agent = RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )


    def get_session_history(self, session_id: str):

        if session_id not in self.sessions:
            self.sessions[session_id] = InMemoryChatMessageHistory()

        return self.sessions[session_id]

    def ask(self, question: str, session_id="default"):

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        response = self.agent.invoke(
            {"question": question},
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )

        return response.text