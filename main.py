from assistant.agent import ResearchAssistant

assistant = ResearchAssistant()

while True:
    question = input("\nResearch Question (exit to quit): ")

    if question.lower() == "exit":
        break

    try:
        answer = assistant.ask(question)
        print("\nAssistant:", answer)

    except Exception as error:
        print(f"\nError: {error}")
