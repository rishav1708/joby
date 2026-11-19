"""Gradio app: home page (upload, search, how it works, eval results) + chat agent. Set GOOGLE_API_KEY first.

    python app.py      # then open http://127.0.0.1:7860
"""
import html
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
.rating {display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: center; padding: 1rem;
         border: 1px solid var(--border-color-primary); border-radius: 12px; background: var(--background-fill-secondary)}
.score {width: 120px; height: 120px; border-radius: 50%; display: grid; place-items: center; font-size: 2.2rem;
        font-weight: 700; border: 8px solid var(--c)}
.score small {display: block; font-size: .8rem; font-weight: 400; opacity: .7; margin-top: -1.6rem}
.bars {flex: 1 1 280px} .bar {margin: .35rem 0} .bar .lbl {display: flex; justify-content: space-between}
.track {height: 8px; border-radius: 4px; background: var(--border-color-primary)}
.fill {height: 8px; border-radius: 4px; background: var(--color-accent)}
.reason {font-size: .85rem; opacity: .75}
.fixes {flex: 1 1 260px} .fixes ol {margin: .3rem 0 0 1.1rem}
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
  <div class="step"><b>4 · Rate &amp; analyse (RAG)</b>One structured Gemini call reads your resume and the
    retrieved jobs: a score out of 100, the top 3 fixes, and per-job gaps with a tailored bullet.</div>
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


LEVELS = {"Entry": "junior", "Mid": "mid", "Senior": "senior"}
STEPS = {"parse": "Read resume (VLM)", "retrieve": "Retrieved + reranked jobs", "grade": "Graded matches (LLM)",
         "rewrite": "Rewrote search query (corrective RAG)", "analyze": "Rated resume + gap analysis (RAG)"}
TIMEOUT_S = 150


def filters(location, levels, company) -> dict:
    return {"location": location, "levels": [LEVELS[x] for x in levels] or None,
            "company": None if company in (None, "Any") else company}


def companies() -> list[str]:
    if not JOBS_PATH.exists():
        return ["Any"]
    return ["Any"] + sorted({json.loads(line)["company"] for line in JOBS_PATH.open()})


def jobs_table(jobs: list[dict]) -> str:
    if not jobs:
        return "No matches. Try loosening the filters."
    rows = ["| # | Role | Company | Location | Level |", "|---|---|---|---|---|"]
    for n, j in enumerate(jobs, 1):
        title = j["title"].replace("|", "/").strip()
        level = {"junior": "Entry", "mid": "Mid", "senior": "Senior"}[seniority(j["title"])]
        rows.append(f"| {n} | [{title}]({j['url']}) | {j['company']} | {j['location'] or 'n/a'} | {level} |")
    return "\n".join(rows)


def search(query, location, levels, company) -> str:
    if not query.strip():
        return ""
    return jobs_table(get_index().search(query, 10, **filters(location, levels, company)))


CATEGORIES = {"skills_match": "Skills match to these jobs", "impact": "Impact & metrics",
              "ml_depth": "ML / AI depth", "ats_format": "Format & ATS readiness"}


def rating_html(r: dict) -> str:
    color = "#16a34a" if r["overall"] >= 75 else "#d97706" if r["overall"] >= 55 else "#dc2626"
    bars = "".join(
        f'<div class="bar"><div class="lbl"><span>{label}</span><b>{r[k]["score"]}/25</b></div>'
        f'<div class="track"><div class="fill" style="width:{r[k]["score"] * 4}%"></div></div>'
        f'<div class="reason">{html.escape(r[k]["reason"])}</div></div>' for k, label in CATEGORIES.items())
    fixes = "".join(f"<li>{html.escape(x)}</li>" for x in r["top_fixes"][:3])
    return (f'<div class="rating"><div class="score" style="--c:{color}">{r["overall"]}<small>/ 100</small></div>'
            f'<div class="bars">{bars}</div><div class="fixes"><b>Top 3 fixes</b><ol>{fixes}</ol></div></div>')


def match(file, location, levels, company):
    """Streams each LangGraph step to the UI as it finishes."""
    if not file:
        raise gr.Error("Upload a resume first.")
    start, log, profile, hits, report, rating = time.time(), [], None, "", "", ""
    yield "⏳ Reading your resume with the VLM…", rating, hits, report, profile
    state = {"resume_path": file, "filters": filters(location, levels, company)}
    try:
        for update in match_graph.stream(state, {"configurable": {"thread_id": str(uuid.uuid4())}},
                                         stream_mode="updates"):
            node, out = next(iter(update.items()))
            log.append(f"✓ {STEPS.get(node, node)} · {time.time() - start:.0f}s")
            profile = out.get("profile", profile)
            hits = jobs_table(out["hits"]) if "hits" in out else hits
            if "rating" in out:
                rating, report = rating_html(out["rating"]), "### Gap analysis\n" + out["report"]
            busy = "" if report else "\n\n⏳ Working…"
            yield "\n\n".join(log) + busy, rating, hits, report, profile
            if not report and time.time() - start > TIMEOUT_S:
                raise gr.Error(f"Stopped after {TIMEOUT_S}s: Gemini is responding slowly (free tier). "
                               "Your parsed resume is cached, so the next try is faster.")
    except gr.Error:
        raise
    except Exception as e:
        if "503" in str(e) or "UNAVAILABLE" in str(e):
            raise gr.Error("Gemini is overloaded right now (both the main and backup models). "
                           "This is usually temporary, so please try again in a minute or two.")
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            raise gr.Error("Gemini free-tier rate limit reached. Wait about a minute and try again "
                           "(your parsed resume is cached).")
        raise gr.Error(f"Something failed: {str(e)[:300]}")


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
        with gr.Row():
            loc = gr.Radio(["Any", "India", "Remote"], value="Any", label="Location")
            lvl = gr.CheckboxGroup(list(LEVELS), label="Seniority (none = all)")
            comp = gr.Dropdown(companies(), value="Any", label="Company")
        status = gr.Markdown()
        rating = gr.HTML()
        hits = gr.Markdown()
        report = gr.Markdown()
        with gr.Accordion("Details: parsed profile (VLM output)", open=False):
            profile = gr.JSON(show_label=False)
        gr.Markdown("## How it works")
        gr.HTML(PIPELINE)
        gr.Markdown("## Evaluation")
        evals = gr.Markdown()
        gr.Markdown("Code: [github.com/rishav1708/joby](https://github.com/rishav1708/joby)")

        fs = [loc, lvl, comp]
        go.click(match, [f] + fs, [status, rating, hits, report, profile])
        sbtn.click(search, [q] + fs, hits)
        q.submit(search, [q] + fs, hits)
        demo.load(lambda: (index_stats(), eval_tables()), None, [stats, evals])

    with gr.Tab("Chat with the agent"):
        uid = gr.Textbox(label="Your name (the agent remembers you by it)", value="anon")
        gr.ChatInterface(chat, additional_inputs=[uid])

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
