import gradio as gr
from inference import run

history = []


def chat(message, chat_history):
    global history
    response = run(message, history)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    chat_history.append((message, response))
    return "", chat_history


def clear():
    global history
    history = []
    return [], ""


with gr.Blocks(title="Tool-Calling Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Tool-Calling Assistant")
    gr.Markdown("Fine-tuned **Qwen2-0.5B** | Tools: weather, calendar, convert, currency, SQL")

    chatbot = gr.Chatbot(height=450, label="Chat")
    msg = gr.Textbox(
        placeholder="Type a message... e.g., What's the weather in London?",
        label="Your Message",
        scale=4,
    )

    with gr.Row():
        send_btn = gr.Button("Send 🚀", variant="primary")
        clear_btn = gr.Button("Clear 🗑️")

    gr.Markdown("### 💡 Try these examples:")
    gr.Examples(
        examples=[
            "What's the weather in London?",
            "Convert 100 meters to feet",
            "Show my calendar for 2026-05-01",
            "Convert 50 USD to EUR",
            "SELECT * FROM users WHERE age > 25",
            "Hello, how are you?",
            "Tell me a joke",
        ],
        inputs=msg,
    )

    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    send_btn.click(chat, [msg, chatbot], [msg, chatbot])
    clear_btn.click(clear, outputs=[chatbot, msg])

demo.launch()
