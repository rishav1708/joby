"""Gradio demo (runs locally or on Hugging Face Spaces). Set GOOGLE_API_KEY first.

    python app.py
"""
import uuid

import gradio as gr
from langchain_core.messages import HumanMessage

from jobpilot.graph import chat_graph, match_graph


def match(file):
    out = match_graph.invoke({"resume_path": file}, {"configurable": {"thread_id": str(uuid.uuid4())}})
    return out["profile"], f"**Search query:** {out['query']} (rewrites: {out['attempts']})\n\n{out['report']}"


def chat(message, history, user_id):
    out = chat_graph.invoke({"messages": [HumanMessage(message)]}, {"configurable": {"thread_id": user_id or "anon"}})
    return out["messages"][-1].text


with gr.Blocks(title="JobPilot") as demo:
    gr.Markdown("# JobPilot\nUpload a resume (PDF, scan or screenshot) to get matched to live jobs.")
    with gr.Tab("Match my resume"):
        f = gr.File(type="filepath", file_types=[".pdf", ".png", ".jpg", ".jpeg"])
        btn = gr.Button("Find jobs")
        profile, report = gr.JSON(label="Parsed profile (VLM)"), gr.Markdown()
        btn.click(match, f, [profile, report])
    with gr.Tab("Chat"):
        uid = gr.Textbox(label="Your name (the agent remembers you by it)", value="anon")
        gr.ChatInterface(chat, additional_inputs=[uid])

if __name__ == "__main__":
    demo.launch()
