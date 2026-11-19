"""Gradio app: home page (upload, search, resume rating, eval results) + chat agent. Set GOOGLE_API_KEY first.

    python app.py      # then open http://127.0.0.1:7860
"""
import html
import json
import queue
import threading
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
.wait {padding: 1rem; border: 1px solid var(--border-color-primary); border-radius: 12px;
       background: var(--background-fill-secondary)}
.wait .head {font-size: 1.1rem; margin-bottom: .5rem} .clock {font-size: 1.8rem; font-weight: 700;
       font-variant-numeric: tabular-nums; color: var(--color-accent); margin-right: .3rem}
.pbar {height: 10px; border-radius: 5px; background: var(--border-color-primary); overflow: hidden}
.pbar div {height: 100%; border-radius: 5px; transition: width 1s linear;
           background-color: #6366f1;
           background-image: linear-gradient(90deg, #6366f1, #a855f7, #6366f1); background-size: 200% 100%; animation: flow 2s linear infinite}
@keyframes flow {to {background-position: -200% 0}}
.steps {display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .8rem}
.st {flex: 1 1 140px; display: flex; align-items: center; gap: .4rem; padding: .5rem .6rem; border-radius: 8px;
     opacity: .5; border: 1px solid var(--border-color-primary)}
.st small {margin-left: auto; opacity: .7} .st.done {opacity: 1} .st.done .ic {color: #16a34a; font-weight: 700}
.st.on {opacity: 1; border-color: var(--color-accent); animation: pulse 1.2s ease-in-out infinite}
@keyframes pulse {50% {box-shadow: 0 0 0 4px color-mix(in srgb, var(--color-accent) 25%, transparent)}}
.tip {margin-top: .7rem; font-size: .95rem; opacity: .85}
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


def describe(location, levels, company) -> str:
    parts = [p for p in [location if location != "Any" else "", ", ".join(levels),
                         company if company not in (None, "Any") else ""] if p]
    return "Filters: " + " · ".join(parts) if parts else "Filters: none"


def refilter(last, query, location, levels, company) -> str:
    """Filters are live: re-run retrieval (no LLM calls) for the last resume match, else the search box."""
    if last:
        jobs = get_index().search(last["query"], 10, junior=last["junior"], **filters(location, levels, company))
        return f"_{describe(location, levels, company)} (re-ranked without re-running the agent)_\n\n" + jobs_table(jobs)
    return search(query, location, levels, company)


def search(query, location, levels, company) -> str:
    if not query.strip():
        return ""
    return (f"_{describe(location, levels, company)}_\n\n"
            + jobs_table(get_index().search(query, 10, **filters(location, levels, company))))


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


STAGES = [("parse", "Reading your resume", "📄"), ("retrieve", "Searching live jobs", "🔎"),
          ("grade", "Grading matches", "⚖️"), ("analyze", "Rating your resume", "⭐")]
TIPS = [
    "Start every bullet with a strong verb: built, shipped, cut, scaled.",
    "Numbers get read first. '2x faster' beats 'improved performance'.",
    "Recruiters skim a resume in about 7 seconds. Put your best project on top.",
    "Mirror the job's keywords. ATS filters match exact terms like 'PyTorch' or 'RAG'.",
    "Interview tip: tell project stories as problem, what you did, result.",
    "One page is plenty for a student. Cut anything that doesn't sell you.",
    "Link a live demo. A working URL beats three paragraphs of description.",
    "Interview tip: know the eval numbers on your resume and how you measured them.",
    "Fun fact: the agent may rewrite its own search query if the first results are weak.",
    "Tailor the top 3 bullets for each job. That alone lifts response rates.",
]


def wait_html(done: dict[str, float], elapsed: float, rewrites: int, finished: bool = False) -> str:
    """Live stepper: elapsed counter, progress bar, step states, rotating tip."""
    current = next((k for k, *_ in STAGES if k not in done), None)
    pct = 100 if finished else min(97, (len(done) + min(1.0, elapsed / 60) * 0.6) / len(STAGES) * 100)
    steps = "".join(
        f'<div class="st {"done" if k in done else "on" if k == current else ""}">'
        f'<span class="ic">{"✓" if k in done else icon}</span>{label}'
        f'<small>{f"{done[k]:.0f}s" if k in done else "…" if k == current else ""}</small></div>'
        for k, label, icon in STAGES)
    note = f'<div class="tip">🔁 The agent rewrote its search {rewrites}× to find better fits.</div>' if rewrites else ""
    tip = "" if finished else f'<div class="tip">💡 {TIPS[int(elapsed // 5) % len(TIPS)]}</div>'
    head = f"✅ Done in {elapsed:.0f}s" if finished else f'<span class="clock">{elapsed:.0f}s</span> working on it'
    return (f'<div class="wait"><div class="head">{head}</div><div class="pbar"><div style="width:{pct:.0f}%">'
            f'</div></div><div class="steps">{steps}</div>{note}{tip}</div>')


def match(file, location, levels, company):
    """Runs the LangGraph pipeline in a thread and ticks the wait screen every second."""
    if not file:
        raise gr.Error("Upload a resume first.")
    start, done, rewrites = time.time(), {}, 0
    profile, hits, report, rating, last = None, "", "", "", None
    state = {"resume_path": file, "filters": filters(location, levels, company)}
    events: queue.Queue = queue.Queue()

    def run():
        try:
            for update in match_graph.stream(state, {"configurable": {"thread_id": str(uuid.uuid4())}},
                                             stream_mode="updates"):
                events.put(("update", update))
            events.put(("done", None))
        except Exception as e:  # surfaced to the UI below
            events.put(("error", e))

    threading.Thread(target=run, daemon=True).start()  # ponytail: can't cancel a running thread on timeout
    while True:
        elapsed = time.time() - start
        try:
            kind, val = events.get(timeout=1)
        except queue.Empty:
            if elapsed > TIMEOUT_S:
                raise gr.Error(f"Stopped after {TIMEOUT_S}s: Gemini is responding slowly (free tier). "
                               "Your parsed resume is cached, so the next try is faster.")
            yield wait_html(done, elapsed, rewrites), rating, hits, report, profile, last
            continue
        if kind == "error":
            e = str(val)
            if "503" in e or "UNAVAILABLE" in e:
                raise gr.Error("Gemini is overloaded right now (both the main and backup models). "
                               "This is usually temporary, so please try again in a minute or two.")
            if "429" in e or "RESOURCE_EXHAUSTED" in e:
                raise gr.Error("Gemini free-tier rate limit reached. Wait about a minute and try again "
                               "(your parsed resume is cached).")
            raise gr.Error(f"Something failed: {e[:300]}")
        if kind == "done":
            yield (wait_html(done, elapsed, rewrites, finished=True) + f"<small>{describe(location, levels, company)}"
                   "</small>", rating, hits, report, profile, last)
            return
        node, out = next(iter(val.items()))
        if node == "rewrite":
            rewrites += 1
            done.pop("retrieve", None), done.pop("grade", None)  # the stepper goes back to searching
        else:
            done[node] = elapsed
        profile = out.get("profile", profile)
        if "query" in out:
            last = {"query": out["query"], "junior": profile["years_experience"] < 2 or None}
        hits = jobs_table(out["hits"]) if "hits" in out else hits
        if "rating" in out:
            rating, report = rating_html(out["rating"]), "### Gap analysis\n" + out["report"]
        yield wait_html(done, elapsed, rewrites), rating, hits, report, profile, last


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
        status = gr.HTML()
        last = gr.State(None)  # last resume run's query, so filter changes re-rank it live
        rating = gr.HTML()
        hits = gr.Markdown()
        report = gr.Markdown()
        with gr.Accordion("Details: parsed profile (VLM output)", open=False):
            profile = gr.JSON(show_label=False)
        gr.Markdown("## Evaluation")
        evals = gr.Markdown()
        gr.Markdown("Code: [github.com/rishav1708/joby](https://github.com/rishav1708/joby)")

        fs = [loc, lvl, comp]
        go.click(match, [f] + fs, [status, rating, hits, report, profile, last])
        sbtn.click(lambda *a: (None, search(*a)), [q] + fs, [last, hits])
        q.submit(lambda *a: (None, search(*a)), [q] + fs, [last, hits])
        for c in fs:
            c.change(refilter, [last, q] + fs, hits)
        demo.load(lambda: (index_stats(), eval_tables()), None, [stats, evals])

    with gr.Tab("Chat with the agent"):
        uid = gr.Textbox(label="Your name (the agent remembers you by it)", value="anon")
        gr.ChatInterface(chat, additional_inputs=[uid])

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
