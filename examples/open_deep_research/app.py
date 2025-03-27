# === app.py ===

import os
import gradio as gr
import logging
from run import create_agent, agent_logs, log_lock
from smolagents.gradio_ui import GradioUI  # Ensure this is the correct import based on smolagents' version

# Simple dark theme styling.
CSS = """
body {
    background-color: #2c2c2c;
    color: #ffffff;
}
.gradio-container {
    background-color: #3a3a3a;
    border-radius: 10px;
    padding: 20px;
}
h1, h2, h3 {
    color: #79c0ff;
}
"""

def set_keys(openai_api_key, serper_api_key, hf_token):
    """
    Update environment variables with the user-provided API keys.
    """
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["SERPER_API_KEY"] = serper_api_key
    os.environ["HF_TOKEN"] = hf_token
    return "API keys have been updated successfully! Please restart the agent for changes to take effect."

def build_app():
    with gr.Blocks(theme="CultriX/gradio-theme", css=CSS) as demo:
        # Title
        title = gr.HTML(
            """<h1>SmolAgents Open_Deep_Search 🥳</h1>""",
            elem_id="title",
        )
        gr.Markdown("## Enhanced Agent UI")
        
        # Configuration Accordion
        with gr.Accordion("Configuration (Click to Expand)", open=False):
            openai_field = gr.Textbox(label="OPENAI_API_KEY", type="password", placeholder="Enter your OpenAI API key")
            serper_field = gr.Textbox(label="SERPER_API_KEY", type="password", placeholder="Enter your Serper API key")
            hf_field = gr.Textbox(label="HF_TOKEN", type="password", placeholder="Enter your Hugging Face Token")
            update_btn = gr.Button("Update Keys")
            status_box = gr.Markdown("*(No keys set yet)*")
            
            update_btn.click(
                fn=set_keys,
                inputs=[openai_field, serper_field, hf_field],
                outputs=status_box
            )
        
        # Initialize the agent
        agent = create_agent()
        
        # Initialize GradioUI with the agent (optional, based on smolagents' capabilities)
        ui = GradioUI(agent)
        
        # Placeholder for logs
        log_textbox = gr.Textbox(label="Agent Logs", lines=20, interactive=False)
        
        # Function to get the answer and logs
        def get_answer(question):
            agent_logs.clear()  # Clear previous logs
            answer = agent.run(question)
            
            with log_lock:
                logs = "\n".join(agent_logs)
            
            return answer, logs
        
        # Ask Question Section
        gr.Markdown("### Ask your question below:")
        question_input = gr.Textbox(label="Your Question", placeholder="Enter your question here...")
        submit_btn = gr.Button("Get Answer")
        answer_output = gr.Textbox(label="Answer", interactive=False)
        
        submit_btn.click(
            fn=get_answer,
            inputs=question_input,
            outputs=[answer_output, log_textbox]
        )
        
    return demo

if __name__ == "__main__":
    demo = build_app()
    # Listen on all interfaces so HF Spaces can route traffic appropriately
    demo.launch(server_name="0.0.0.0")
