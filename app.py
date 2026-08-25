import gradio as gr
from dotenv import load_dotenv

from implement.answer import answer_question

load_dotenv(override=True)


def format_context(context):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"

    for doc in context:
        result += (
            f"<span style='color: #ff7800;'>"
            f"Source: {doc.metadata['source']}"
            f"</span>\n\n"
        )
        result += doc.page_content + "\n\n"

    return result


def extract_text(content):
    """
    Convert Gradio's message content into a plain string.

    Gradio may give us:
        "hello"

    or:
        {"text": "hello", "type": "text"}

    or:
        [{"text": "hello", "type": "text"}]
    """

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        return content.get("text", "")

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                text_parts.append(item.get("text", ""))

        return "".join(text_parts)

    return str(content)


def clean_history(history):
    """
    Convert all message contents into plain strings
    before sending them to the RAG pipeline.
    """

    cleaned = []

    for message in history:
        cleaned.append(
            {
                "role": message["role"],
                "content": extract_text(message["content"]),
            }
        )

    return cleaned


def chat(history):

    # Get the latest user message
    last_message = extract_text(history[-1]["content"])

    # Previous conversation, excluding the latest user message
    prior = clean_history(history[:-1])

    print("QUESTION:", last_message)
    print("TYPE:", type(last_message))

    # Add an empty assistant message
    history = history.copy()

    history.append(
        {
            "role": "assistant",
            "content": "",
        }
    )

    # answer_question() is a generator because it streams
    for answer, context in answer_question(
        last_message,
        prior,
    ):

        # Update assistant response progressively
        history[-1]["content"] = answer

        yield history, format_context(context)


def main():

    def put_message_in_chatbot(message, history):

        # Convert whatever Gradio gives us into plain text
        message = extract_text(message)

        # Add the user's message
        history = history + [
            {
                "role": "user",
                "content": message,
            }
        ]

        # Clear textbox and update chatbot
        return "", history

    theme = gr.themes.Soft(
        font=[
            "Inter",
            "system-ui",
            "sans-serif",
        ]
    )

    with gr.Blocks(
        title="Nvidia Expert Assistant"
    ) as ui:

        gr.Markdown(
            "# 🏢 Nvidia Expert Assistant\n"
            "Ask me anything about Nvidia!"
        )

        with gr.Row():

            # -----------------------------
            # CHAT COLUMN
            # -----------------------------

            with gr.Column(scale=1):

                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=600,
                )

                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Nvidia...",
                    show_label=False,
                )

            # -----------------------------
            # CONTEXT COLUMN
            # -----------------------------

            with gr.Column(scale=1):

                context_markdown = gr.Markdown(
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        # -----------------------------
        # MESSAGE SUBMISSION
        # -----------------------------

        message.submit(
            put_message_in_chatbot,
            inputs=[
                message,
                chatbot,
            ],
            outputs=[
                message,
                chatbot,
            ],
        ).then(
            chat,
            inputs=chatbot,
            outputs=[
                chatbot,
                context_markdown,
            ],
        )

    # Gradio 6:
    # theme belongs in launch(), NOT Blocks()
    ui.launch(
        inbrowser=True,
        theme=theme,
    )


if __name__ == "__main__":
    main()