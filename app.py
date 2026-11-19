"""Gradio app: resume matching, job search, resume review, eval results, and a chat agent.
Set GOOGLE_API_KEY first.

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

LEVELS = {"Entry": "junior", "Mid": "mid", "Senior": "senior"}
LEVEL_NAME = {"junior": "Entry", "mid": "Mid", "senior": "Senior"}
TIMEOUT_S = 150
STAGES = [("parse", "Read resume"), ("retrieve", "Search roles"), ("grade", "Check fit"), ("analyze", "Score resume")]
TIPS = [
    "Start each bullet with a verb that names what you did: built, shipped, reduced, scaled.",
    "Put numbers in bullets. \"2x faster training\" is read before \"improved performance\".",
    "Recruiters skim a resume in seconds, so lead with your strongest project.",
    "Use the job's own terms. Many screening systems match exact keywords such as PyTorch or RAG.",
    "In interviews, tell each project as the problem, what you did, and the measured result.",
    "Be ready to explain how you measured every number on your resume.",
    "A link to a working demo is often more persuasive than another paragraph.",
    "Rewrite your top three bullets for each role you apply to.",
]
CATEGORIES = {"skills_match": "Skills match", "impact": "Impact and metrics",
              "ml_depth": "ML and AI depth", "ats_format": "Format and ATS"}
COMPANY_NAMES = {"openai": "OpenAI", "scaleai": "Scale AI", "deepmind": "Google DeepMind", "phonepe": "PhonePe",
                 "gitlab": "GitLab", "mongodb": "MongoDB", "thoughtspot": "ThoughtSpot", "hubspot": "HubSpot",
                 "browserstack": "BrowserStack", "elevenlabs": "ElevenLabs", "togetherai": "Together AI",
                 "cred": "CRED", "inmobi": "InMobi", "sharechat": "ShareChat", "moengage": "MoEngage",
                 "huggingface": "Hugging Face", "weightsandbiases": "Weights & Biases", "wandb": "Weights & Biases",
                 "highradius": "HighRadius", "runwayml": "Runway", "stabilityai": "Stability AI",
                 "assemblyai": "AssemblyAI", "snorkelai": "Snorkel AI", "characterai": "Character.AI",
                 "dream11": "Dream11", "fireworks-ai": "Fireworks AI"}

CSS = """
:root { --ink: #0f172a; --muted: #64748b; --line: #e2e8f0; --soft: #f8fafc; --accent: #1d4ed8; }
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; }
footer { display: none !important; }
.topbar { display: flex; align-items: center; justify-content: space-between; padding: 14px 0 18px;
          border-bottom: 1px solid var(--line); margin-bottom: 28px; flex-wrap: wrap; gap: 8px; }
.brand { font-weight: 700; font-size: 18px; letter-spacing: -0.01em; color: var(--ink); }
.brand span { color: var(--accent); }
.topbar nav { display: flex; gap: 20px; font-size: 14px; color: var(--muted); }
.topbar nav a { color: var(--muted); text-decoration: none; } .topbar nav a:hover { color: var(--ink); }
.intro h1 { font-size: 34px; line-height: 1.15; letter-spacing: -0.02em; margin: 0 0 10px; color: var(--ink); }
.intro p { font-size: 16px; color: var(--muted); max-width: 640px; margin: 0 0 8px; }
.panel { border: 1px solid var(--line) !important; border-radius: 10px !important; background: #fff !important;
         padding: 18px !important; box-shadow: none !important; }
.eyebrow { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em;
           color: var(--muted); margin-bottom: 8px; }
.filters { margin-top: 12px; }
.section-title { font-size: 20px; font-weight: 600; color: var(--ink); margin: 32px 0 4px; }
.section-sub { font-size: 14px; color: var(--muted); margin-bottom: 14px; }
.results-head { display: flex; justify-content: space-between; font-size: 13px; color: var(--muted);
                margin: 14px 0 10px; }
.jobs { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }
.job { border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; background: #fff; text-align: left; }
.job:hover { border-color: #cbd5e1; }
.job a.t { display: block; font-weight: 600; font-size: 15px; color: var(--ink); text-decoration: none;
           line-height: 1.35; } .job a.t:hover { color: var(--accent); }
.job .co { font-size: 14px; color: var(--muted); margin: 2px 0 10px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { font-size: 12px; padding: 2px 8px; border-radius: 999px; background: var(--soft);
        border: 1px solid var(--line); color: #334155; }
.chip.entry { background: #ecfdf5; border-color: #a7f3d0; color: #047857; }
.chip.senior { color: #64748b; }
.empty { padding: 24px; text-align: center; color: var(--muted); border: 1px dashed var(--line); border-radius: 10px;
         margin-top: 14px; }
.progress { border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; background: #fff; margin-top: 16px; }
.progress .row { display: flex; justify-content: space-between; align-items: baseline; }
.progress .label { font-weight: 600; color: var(--ink); }
.progress .time { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 14px; }
.track { height: 6px; border-radius: 3px; background: var(--line); overflow: hidden; margin: 12px 0 14px; }
.track div { height: 100%; background: var(--accent); transition: width 1s linear; }
.stages { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
.stage { font-size: 13px; color: #94a3b8; display: flex; gap: 8px; align-items: center; }
.stage i { font-style: normal; width: 20px; height: 20px; border-radius: 50%; display: grid; place-items: center;
           font-size: 11px; border: 1px solid var(--line); color: #94a3b8; flex: none; }
.stage.on { color: var(--ink); } .stage.on i { border-color: var(--accent); color: var(--accent); }
.stage.done { color: var(--ink); } .stage.done i { background: var(--accent); border-color: var(--accent); color: #fff; }
.tip { font-size: 13px; color: var(--muted); margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--line); }
.tip b { color: var(--ink); font-weight: 600; }
.review { display: grid; grid-template-columns: 180px 1fr 1fr; gap: 28px; border: 1px solid var(--line);
          border-radius: 10px; padding: 20px; background: #fff; }
.score .n { font-size: 52px; font-weight: 700; letter-spacing: -0.03em; color: var(--ink); line-height: 1; }
.score .n small { font-size: 18px; font-weight: 500; color: var(--muted); }
.score .verdict { margin-top: 8px; font-size: 13px; color: var(--muted); }
.metric { margin-bottom: 12px; } .metric .top { display: flex; justify-content: space-between; font-size: 14px; }
.metric .top b { font-variant-numeric: tabular-nums; font-weight: 600; }
.metric .bar { height: 4px; background: var(--line); border-radius: 2px; margin: 5px 0 4px; }
.metric .bar div { height: 100%; background: var(--accent); border-radius: 2px; }
.metric p { font-size: 12.5px; color: var(--muted); margin: 0; }
.fixes ol { margin: 0; padding-left: 18px; } .fixes li { font-size: 14px; margin-bottom: 8px; color: #334155; }
.gaps { display: grid; gap: 12px; }
.gap { border: 1px solid var(--line); border-radius: 10px; padding: 16px; background: #fff; }
.gap .head { display: flex; justify-content: space-between; gap: 12px; text-align: left; }
.gap .head a { font-weight: 600; color: var(--ink); text-decoration: none; }
.gap .fit { font-size: 13px; font-variant-numeric: tabular-nums; color: var(--muted); white-space: nowrap; }
.gap .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 10px 0; font-size: 13.5px; }
.gap .cols b { display: block; font-size: 12px; color: var(--muted); font-weight: 600; margin-bottom: 4px; }
.gap blockquote { margin: 0; padding: 10px 12px; background: var(--soft); border-left: 3px solid var(--accent);
                  font-size: 13.5px; color: #334155; border-radius: 0 6px 6px 0; }
.learn { font-size: 14px; color: #334155; margin-top: 12px; }
.evals table { border-collapse: collapse; width: 100%; font-size: 14px; margin-bottom: 18px; }
.evals th, .evals td { border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; }
.evals th { font-weight: 600; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.evals td.num { font-variant-numeric: tabular-nums; }
.evals h4 { margin: 8px 0 6px; font-size: 15px; } .evals h4 small { color: var(--muted); font-weight: 400; }
.muted { color: var(--muted); font-size: 14px; }
.job a.t, .gap .head a { text-align: left !important; padding: 0 !important; }
@media (max-width: 760px) { .review, .gap .cols, .stages { grid-template-columns: 1fr; } }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate, radius_size="md",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#f8fafc", block_shadow="none",
    button_primary_background_fill="#1d4ed8", button_primary_background_fill_hover="#1e40af",
    button_primary_text_color="#ffffff", button_secondary_background_fill="#ffffff",
    button_secondary_border_color="#cbd5e1", input_background_fill="#ffffff",
)
FORCE_LIGHT = """() => { const u = new URL(location.href);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); location.replace(u); } }"""


def esc(x) -> str:
    return html.escape(str(x or ""))


def company_name(slug: str) -> str:
    return COMPANY_NAMES.get(slug, slug if any(c.isupper() for c in slug) else slug.replace("-", " ").title())


def load_jobs() -> list[dict]:
    return [json.loads(line) for line in JOBS_PATH.open() if line.strip()] if JOBS_PATH.exists() else []


def header() -> str:
    jobs = load_jobs()
    meta = "No roles indexed yet"
    if jobs:
        updated = time.strftime("%d %b %Y", time.localtime(JOBS_PATH.stat().st_mtime))
        meta = f"{len(jobs):,} open roles · {len({j.get('company') for j in jobs})} companies · updated {updated}"
    return (f'<div class="topbar"><div class="brand">Job<span>Pilot</span></div><nav><span>{meta}</span>'
            f'<a href="https://github.com/rishav1708/joby" target="_blank">Source</a></nav></div>'
            '<div class="intro"><h1>Find roles that fit your resume</h1><p>Upload your resume to get ranked matches '
            'from live postings and a scored review against those roles. Or search the postings directly.</p></div>')


def eval_tables() -> str:
    if not RESULTS.exists():
        return ('<div class="muted">No results yet. Run <code>python -m evals.retrieval</code>, '
                '<code>python -m evals.vlm</code> and <code>python -m evals.rag</code>; '
                'the numbers appear here automatically.</div>')
    out = ['<div class="evals">']
    for name, sec in json.loads(RESULTS.read_text()).items():
        metrics = list(next(iter(sec["rows"].values())))
        out.append(f"<h4>{esc(name)} <small>· {esc(sec['note'])}</small></h4><table><tr><th></th>"
                   + "".join(f"<th>{esc(m)}</th>" for m in metrics) + "</tr>")
        for row, vals in sec["rows"].items():
            cells = "".join(f'<td class="num">{v:.3f}</td>' if isinstance(v, float) else f'<td class="num">{esc(v)}</td>'
                            for v in (vals[m] for m in metrics))
            out.append(f"<tr><td>{esc(row)}</td>{cells}</tr>")
        out.append("</table>")
    return "".join(out) + "</div>"


def filters(location, levels, company) -> dict:
    return {"location": location, "levels": [LEVELS[x] for x in levels] or None,
            "company": None if company in (None, "Any") else company}


def companies() -> list[tuple[str, str]]:
    slugs = sorted({j.get("company") or "" for j in load_jobs()} - {""}, key=lambda s: company_name(s).lower())
    return [("All companies", "Any")] + [(company_name(s), s) for s in slugs]


def describe(location, levels, company) -> str:
    parts = [location if location != "Any" else "", ", ".join(levels),
             company_name(company) if company not in (None, "Any") else ""]
    parts = [p for p in parts if p]
    return "Filtered by " + " · ".join(parts) if parts else "No filters"


def job_cards(jobs: list[dict], note: str = "") -> str:
    if not jobs:
        return '<div class="empty">No roles match. Try removing a filter.</div>'
    cards = []
    for j in jobs:
        lvl = seniority(j["title"])
        cls = {"junior": "entry", "senior": "senior"}.get(lvl, "")
        chips = f'<span class="chip {cls}">{LEVEL_NAME[lvl]}</span>'
        if j.get("location"):
            chips += f'<span class="chip">{esc(j["location"][:60])}</span>'
        cards.append(f'<div class="job"><a class="t" href="{esc(j.get("url"))}" target="_blank">{esc(j["title"])}</a>'
                     f'<div class="co">{esc(company_name(j["company"]))}</div><div class="chips">{chips}</div></div>')
    return (f'<div class="results-head"><span>{len(jobs)} roles</span><span>{esc(note)}</span></div>'
            f'<div class="jobs">{"".join(cards)}</div>')


def refilter(last, query, location, levels, company) -> str:
    """Filters are live: re-run retrieval (no LLM calls) for the last resume match, else the search box."""
    if last:
        jobs = get_index().search(last["query"], 10, junior=last["junior"], **filters(location, levels, company))
        return '<div class="section-title">Matched roles</div>' + job_cards(jobs, describe(location, levels, company))
    return search(query, location, levels, company)


def search(query, location, levels, company) -> str:
    if not query.strip():
        return ""
    jobs = get_index().search(query, 10, **filters(location, levels, company))
    return job_cards(jobs, describe(location, levels, company))


def review_html(r: dict) -> str:
    verdict = ("Strong for these roles" if r["overall"] >= 75 else "Competitive, with some gaps"
               if r["overall"] >= 55 else "Needs work for these roles")
    metrics = "".join(
        f'<div class="metric"><div class="top"><span>{label}</span><b>{r[k]["score"]}/25</b></div>'
        f'<div class="bar"><div style="width:{r[k]["score"] * 4}%"></div></div><p>{esc(r[k]["reason"])}</p></div>'
        for k, label in CATEGORIES.items())
    fixes = "".join(f"<li>{esc(x)}</li>" for x in r["top_fixes"][:3])
    return ('<div class="section-title">Resume review</div><div class="section-sub">Scored against the roles '
            f'matched below.</div><div class="review"><div class="score"><div class="eyebrow">Overall</div>'
            f'<div class="n">{r["overall"]}<small> / 100</small></div><div class="verdict">{verdict}</div></div>'
            f'<div>{metrics}</div><div class="fixes"><div class="eyebrow">Fix first</div><ol>{fixes}</ol></div></div>')


def gaps_html(r: dict, seen: dict[str, dict]) -> str:
    cards = []
    for f in r["jobs"]:
        j = seen.get(f["job_id"]) or get_index().jobs.get(f["job_id"]) or {"title": f["job_id"], "company": "", "url": ""}
        cards.append(
            f'<div class="gap"><div class="head"><a href="{esc(j.get("url"))}" target="_blank">{esc(j["title"])}'
            f'<span class="muted"> · {esc(company_name(j["company"]))}</span></a>'
            f'<span class="fit">Fit {f["fit"]}/10</span></div>'
            f'<div class="cols"><div><b>You already show</b>{esc(", ".join(f["meets"]) or "None listed")}</div>'
            f'<div><b>Missing</b>{esc(", ".join(f["gaps"]) or "None listed")}</div></div>'
            f'<blockquote>{esc(f["tailored_bullet"])}</blockquote></div>')
    learn = f'<div class="learn"><b>Worth learning next:</b> {esc(", ".join(r["skills_to_learn"]))}</div>'
    return ('<div class="section-title">Gap analysis</div><div class="section-sub">What each top role asks for, '
            f'and one of your bullets rewritten for it.</div><div class="gaps">{"".join(cards)}</div>{learn}')


def progress_html(done: dict[str, float], elapsed: float, rewrites: int, finished: bool = False) -> str:
    current = next((k for k, _ in STAGES if k not in done), None)
    pct = 100 if finished else min(96, (len(done) + min(1.0, elapsed / 60) * 0.6) / len(STAGES) * 100)
    stages = "".join(f'<div class="stage {"done" if k in done else "on" if k == current else ""}">'
                     f'<i>{"✓" if k in done else n}</i>{label}</div>' for n, (k, label) in enumerate(STAGES, 1))
    label = "Finished" if finished else dict(STAGES).get(current, "Working") + "…"
    extra = f" · search refined {rewrites}×" if rewrites else ""
    tip = "" if finished else f'<div class="tip"><b>Tip.</b> {esc(TIPS[int(elapsed // 6) % len(TIPS)])}</div>'
    return (f'<div class="progress"><div class="row"><span class="label">{label}</span>'
            f'<span class="time">{elapsed:.0f}s{extra}</span></div><div class="track"><div style="width:{pct:.0f}%">'
            f'</div></div><div class="stages">{stages}</div>{tip}</div>')


def match(file, location, levels, company):
    """Runs the LangGraph pipeline in a thread and updates the progress panel every second."""
    if not file:
        raise gr.Error("Upload a resume first.")
    start, done, rewrites, seen = time.time(), {}, 0, {}
    profile, hits, report, review, last = None, "", "", "", None
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
                raise gr.Error(f"Stopped after {TIMEOUT_S}s because Gemini is responding slowly. "
                               "Your parsed resume is cached, so the next attempt is faster.")
            yield progress_html(done, elapsed, rewrites), review, hits, report, profile, last
            continue
        if kind == "error":
            e = str(val)
            if "503" in e or "UNAVAILABLE" in e:
                raise gr.Error("Gemini is overloaded at the moment, including the backup model. "
                               "Please try again in a minute or two.")
            if "429" in e or "RESOURCE_EXHAUSTED" in e:
                raise gr.Error("Gemini's free-tier rate limit was reached. Wait about a minute and try again.")
            raise gr.Error(f"Something went wrong: {e[:300]}")
        if kind == "done":
            yield progress_html(done, elapsed, rewrites, finished=True), review, hits, report, profile, last
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
        if "hits" in out:
            seen.update((j["id"], j) for j in out["hits"])
            hits = ('<div class="section-title">Matched roles</div>'
                    + job_cards(out["hits"], describe(location, levels, company)))
        if "rating" in out:
            review, report = review_html(out["rating"]), gaps_html(out["rating"], seen)
        yield progress_html(done, elapsed, rewrites), review, hits, report, profile, last


def chat(message, history, user_id):
    out = chat_graph.invoke({"messages": [HumanMessage(message)]}, {"configurable": {"thread_id": user_id or "anon"}})
    return out["messages"][-1].text


with gr.Blocks(title="JobPilot") as demo:
    top = gr.HTML()
    with gr.Tab("Match"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes="panel"):
                gr.HTML('<div class="eyebrow">Your resume</div>')
                f = gr.File(type="filepath", file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"], show_label=False,
                            height=150)
                go = gr.Button("Match my resume", variant="primary")
            with gr.Column(scale=1, elem_classes="panel"):
                gr.HTML('<div class="eyebrow">Search roles</div>')
                q = gr.Textbox(placeholder="e.g. machine learning intern, LLM engineer", show_label=False)
                gr.Examples(["machine learning intern", "LLM engineer RAG", "computer vision engineer",
                             "data scientist India"], q, label="Try")
                sbtn = gr.Button("Search", variant="secondary")
        with gr.Row(elem_classes=["panel", "filters"]):
            loc = gr.Radio(["Any", "India", "Remote"], value="Any", label="Location")
            lvl = gr.CheckboxGroup(list(LEVELS), label="Level")
            comp = gr.Dropdown(companies(), value="Any", label="Company", filterable=True)
        status = gr.HTML()
        last = gr.State(None)  # last resume run's query, so filter changes re-rank it live
        review = gr.HTML()
        hits = gr.HTML()
        report = gr.HTML()
        with gr.Accordion("Extracted profile", open=False):
            profile = gr.JSON(show_label=False)

        fs = [loc, lvl, comp]
        go.click(match, [f] + fs, [status, review, hits, report, profile, last])
        sbtn.click(lambda *a: (None, search(*a)), [q] + fs, [last, hits])
        q.submit(lambda *a: (None, search(*a)), [q] + fs, [last, hits])
        for c in fs:
            c.change(refilter, [last, q] + fs, hits)

    with gr.Tab("Assistant"):
        gr.HTML('<div class="section-sub">Ask about roles, companies or what to learn next. The assistant searches '
                'the same live postings and remembers your preferences by name.</div>')
        uid = gr.Textbox(label="Your name", value="guest")
        gr.ChatInterface(chat, additional_inputs=[uid])

    with gr.Tab("Evaluation"):
        gr.HTML('<div class="section-sub">Offline benchmarks for retrieval, resume extraction and grounding.</div>')
        evals = gr.HTML()

    demo.load(lambda: (header(), eval_tables()), None, [top, evals])

if __name__ == "__main__":
    if not JOBS_PATH.exists():  # fresh machine or Space without a prebuilt index
        from jobpilot import ingest
        ingest.main()
    print("Loading the job index (new jobs get embedded now, so searches are instant later)...")
    print(f"ready: {len(get_index().jobs):,} jobs")
    demo.launch(css=CSS, theme=THEME, js=FORCE_LIGHT)
