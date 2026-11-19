"""Gradio app: home page (upload, search, how it works, eval results) + chat agent. Set GOOGLE_API_KEY first.

    python app.py      # then open http://127.0.0.1:7860
"""
import json
import time
import uuid

import gradio as gr
from langchain_core.messages import HumanMessage

from evals import RESULTS
from jobpilot.graph import chat_graph, match_graph
from jobpilot.ingest import JOBS_PATH
from jobpilot.retrieve import get_index, seniority

CSS = """
.hero h1 {font-size: 2.4rem; margin-bottom: .2rem}
.pipeline {display: flex; flex-wrap: wrap; gap: .5rem; align-items: stretch}
.step {flex: 1 1 180px; border: 1px solid var(--border-color-primary); border-radius: 10px; padding: .8rem;
       background: var(--background-fill-secondary)}
.step b {display: block; margin-bottom: .3rem}
.arrow {align-self: center; font-size: 1.4rem; opacity: .6}
"""

PIPELINE = """
<div class="pipeline">
  <div class="step"><b>1 · Read (VLM)</b>Gemini vision reads your resume PDF, scan or screenshot into a validated
    Pydantic profile: skills, years, target roles.</div>
  <div class="arrow">→</div>
  <div class="step"><b>2 · Retrieve (hybrid RAG)</b>BM25 + bge-small embeddings in FAISS, fused with reciprocal
    rank fusion, reranked by a cross-encoder, with seniority-aware ranking for students.</div>
  <div class="arrow">→</div>
  <div class="step"><b>3 · Grade &amp; retry (LangGraph)</b>An LLM grades each job. Fewer than 3 good fits? The agent
    rewrites the query and searches again (corrective RAG).</div>
  <div class="arrow">→</div>
  <div class="step"><b>4 · Analyse (RAG)</b>Fit score, gaps and a tailored resume bullet per job, grounded in and
    cited to the job descriptions.</div>
</div>
"""


def index_stats() -> str:
    if not JOBS_PATH.exists():
        return "**No jobs indexed yet.** Run `python -m jobpilot.ingest`."
    jobs = [json.loads(line) for line in JOBS_PATH.open()]
    updated = time.strftime("%d %b %Y, %H:%M", time.localtime(JOBS_PATH.stat().st_mtime))
    return (f"**{len(jobs):,} live jobs** from **{len({j['company'] for j in jobs})} companies** "
            f"(Greenhouse, Lever, Remotive) · refreshed {updated}")


def eval_tables() -> str:
    if not RESULTS.exists():
        return ("_No eval results yet. Run `python -m evals.retrieval`, `python -m evals.vlm` and "
                "`python -m evals.rag`; the numbers appear here automatically._")
    out = []
    for name, sec in json.loads(RESULTS.read_text()).items():
        metrics = list(next(iter(sec["rows"].values())))
        out += [f"**{name}** ({sec['note']})", "", "| | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
        for row, vals in sec["rows"].items():
            out.append(f"| {row} | " + " | ".join(f"{vals[m]:.3f}" if isinstance(vals[m], float) else str(vals[m])
                                                for m in metrics) + " |")
        out.append("")
    return "\n".join(out)


def search(query: str) -> str:
    if not query.strip():
        return ""
    lines = [f"{n}. **[{j['title']}]({j['url']})** · {j['company']} · {j['location'] or 'n/a'}"
             + (" · _entry level_" if seniority(j["title"]) == "junior" else "")
             for n, j in enumerate(get_index().search(query, 10), 1)]
    return "\n".join(lines) or "No matches."


def match(file):
    if not file:
        raise gr.Error("Upload a resume first.")
    out = match_graph.invoke({"resume_path": file}, {"configurable": {"thread_id": str(uuid.uuid4())}})
    return out["profile"], f"**Search query:** {out['query']} (rewrites: {out['attempts']})\n\n{out['report']}"


def chat(message, history, user_id):
    out = chat_graph.invoke({"messages": [HumanMessage(message)]}, {"configurable": {"thread_id": user_id or "anon"}})
    return out["messages"][-1].text


with gr.Blocks(title="JobPilot") as demo:
    with gr.Tab("Home"):
        gr.Markdown("# JobPilot\n### An AI career agent that reads your resume and finds live jobs you can "
                    "actually get, then tells you what's missing.", elem_classes="hero")
        stats = gr.Markdown()
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.Markdown("#### Match my resume")
                f = gr.File(type="filepath", file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"],
                            label="Resume (PDF, scan or screenshot)")
                go = gr.Button("Find my jobs", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("#### Or just search")
                q = gr.Textbox(placeholder="e.g. junior ML engineer pytorch", label="Search live jobs")
                gr.Examples(["junior ML engineer pytorch", "LLM engineer RAG langchain",
                             "machine learning intern", "computer vision engineer"], q)
                sbtn = gr.Button("Search")
        hits = gr.Markdown()
        with gr.Accordion("Your parsed profile and gap analysis", open=False) as acc:
            profile, report = gr.JSON(label="Parsed profile (VLM)"), gr.Markdown()
        gr.Markdown("## How it works")
        gr.HTML(PIPELINE)
        gr.Markdown("## Evaluation")
        evals = gr.Markdown()
        gr.Markdown("Code: [github.com/rishav1708/joby](https://github.com/rishav1708/joby)")

        go.click(match, f, [profile, report], show_progress="full").then(lambda: gr.Accordion(open=True), None, acc)
        sbtn.click(search, q, hits)
        q.submit(search, q, hits)
        demo.load(lambda: (index_stats(), eval_tables()), None, [stats, evals])

    with gr.Tab("Chat with the agent"):
        uid = gr.Textbox(label="Your name (the agent remembers you by it)", value="anon")
        gr.ChatInterface(chat, additional_inputs=[uid])

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
