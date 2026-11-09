from assistant.agent import ResearchAssistant

assistant = ResearchAssistant()

while True:
    question = input("\nResearch Question (exit to quit): ")

    if question.lower() == "exit":
        break

    try:
        print("\nAssistant: ", end="", flush=True)
        assistant.stream(question)

    except Exception as error:
        print(f"\nError: {error}")
