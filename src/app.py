from utils.ui_settings import UISettings
from utils.app_utils import get_available_collections, get_available_databases
from utils.env_loader import load_environment
from chatbot.chatbot_backend import ChatBot
import gradio as gr

# Load environment variables at startup
load_environment()


def format_database_info() -> str:
    """Format available database information for display."""
    databases = get_available_databases()
    info = "Available Databases:\n\n"
    
    for db_type, db_names in databases.items():
        info += f"{db_type.upper()}:\n"
        for name in db_names:
            info += f"- {name}\n"
        info += "\n"
    
    return info


def format_collection_info() -> str:
    """Format available collection information for display."""
    collections = get_available_collections()
    info = "Available Document Collections:\n\n"
    
    for collection in collections:
        info += f"- {collection}\n"
    
    return info


with gr.Blocks() as demo:
    # Information about available resources
    with gr.Row():
        with gr.Column():
            gr.Markdown(format_database_info())
        with gr.Column():
            gr.Markdown(format_collection_info())
    
    # Chat interface
    with gr.Row():
        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            bubble_full_width=False,
            height=500,
            avatar_images=(
                "images/AI_RT.png", "images/openai.png"
            ),
        )
        chatbot.like(UISettings.feedback, None, None)
    
    # Input area
    with gr.Row():
        input_txt = gr.Textbox(
            lines=3,
            scale=8,
            placeholder="Ask me anything about your documents or databases...",
            container=False,
        )
    
    # Control buttons
    with gr.Row():
        text_submit_btn = gr.Button(value="Submit")
        clear_button = gr.ClearButton([input_txt, chatbot])
    
    # Event handlers
    txt_msg = input_txt.submit(
        fn=ChatBot.respond,
        inputs=[chatbot, input_txt],
        outputs=[input_txt, chatbot],
        queue=False
    ).then(
        lambda: gr.Textbox(interactive=True),
        None,
        [input_txt],
        queue=False
    )

    txt_msg = text_submit_btn.click(
        fn=ChatBot.respond,
        inputs=[chatbot, input_txt],
        outputs=[input_txt, chatbot],
        queue=False
    ).then(
        lambda: gr.Textbox(interactive=True),
        None,
        [input_txt],
        queue=False
    )


if __name__ == "__main__":
    demo.launch()
