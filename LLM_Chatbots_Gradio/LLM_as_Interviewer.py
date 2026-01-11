"""
LLM as Interviewer - A Gradio-based chat interface using Ollama
This application simulates an AI engineer interview using a local LLM.
"""

import ollama
import gradio as gr

# Model configuration
MODEL_LLAMA = 'llama3.2'

# System prompt for the interviewer
SYSTEM_MESSAGE = (
    "You are an interviewer who is conducting an interview with a candidate "
    "for an AI engineer position. Ask around 10 questions to check his knowledge "
    "and at the end give feedback."
)


def chat(message: str, history: list) -> str:
    """
    Process a chat message and generate a streaming response.

    Args:
        message: The user's input message
        history: Chat history (provided by Gradio)

    Yields:
        Accumulated response text as it streams
    """
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": message}
    ]

    stream = ollama.chat(model=MODEL_LLAMA, messages=messages, stream=True)

    response_text = ""
    for chunk in stream:
        response_text += chunk['message']['content']
        yield response_text


def main():
    """Launch the Gradio chat interface."""
    interface = gr.ChatInterface(
        fn=chat,
        type="messages",
        title="AI Engineer Interview Simulator",
        description="Practice your AI engineering interview with an LLM interviewer."
    )
    interface.launch(share=True)


if __name__ == "__main__":
    main()
