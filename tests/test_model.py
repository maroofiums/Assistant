from assistant.model import get_model

def test_model():
    model = get_model()

    response = model.invoke(
        "Explain RAG in three sentences."
    )

    print(response.content)

if __name__ == "__main__":
    test_model()