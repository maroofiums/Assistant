
from assistant.agent import ResearchAssistant


def main():
    assistant = ResearchAssistant()
    session_id = "default"

    print("\nAI Research Assistant")
    print("Commands: exit | reset | new")
    print("-" * 40)

    while True:
        try:
            question = input("\nYou: ").strip()

            if not question:
                continue

            if question.lower() == "exit":
                print("Goodbye!")
                break

            if question.lower() == "reset":
                assistant.reset_session(session_id)
                print("Conversation cleared.")
                continue

            if question.lower() == "new":
                session_id = f"session-{len(assistant.sessions) + 1}"
                print(f"New session: {session_id}")
                continue

            print("\nAssistant: ", end="", flush=True)
            assistant.stream(question, session_id)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()
