
from src.config import OPENAI_MODEL, get_openai_client


def main():
    client = get_openai_client()

    response = client.responses.create(
        model=OPENAI_MODEL,
        input="Reply with exactly this text: OpenAI connection successful.",
        max_output_tokens=30,
    )

    print("Model:", OPENAI_MODEL)
    print("Response:", response.output_text)


if __name__ == "__main__":
    main()
