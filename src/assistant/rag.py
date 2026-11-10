from langchain_core.prompts import ChatPromptTemplate

from assistant.model import get_model
from assistant.vector_store import ResearchVectorStore


class ResearchRAG:

    def __init__(self, index_path="data/vector_store"):
        self.vector_store = ResearchVectorStore()
        self.vector_store.load(path=index_path)

        self.model = get_model()
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are a research paper assistant.

                Answer using only the provided context.
                If the context does not contain the answer,
                clearly say that the information is unavailable.

                Cite the relevant page numbers when possible.
                """
            ),
            (
                "human",
                """
                Research Context:
                {context}

                Question:
                {question}
                """
            )
        ])

        self.chain = self.prompt | self.model


    def ask(self, question: str, k: int = 4) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        documents = self.vector_store.search(query=question, k=k)   

        context = "\n\n".join(
            f"[Page {doc.metadata.get('page', "Unknown")}]\n"
            f"{doc.page_content}"   
            for doc in documents
        )

        response = self.chain.invoke({
            "context": context,
            "question": question
        })


        return {
            "answer": response.content,
            "sources": documents
        }