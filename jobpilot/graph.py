"""LangGraph agents.

match_graph: parse (VLM) -> retrieve (hybrid + rerank) -> grade (LLM) -> [rewrite -> retrieve]* -> analyze (RAG)
chat_graph:  tool-calling agent (search_jobs, get_job) with SQLite-checkpointed memory per thread_id.

    python -m jobpilot.graph resume.pdf
"""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from . import vlm
from .llm import llm
from .retrieve import get_index

MAX_REWRITES = 1  # each rewrite costs 2 Gemini calls; one retry catches most bad queries
ANALYZE_JOBS = 3
PROFILE_CACHE = Path("data/profiles")
MIN_RELEVANT = 3
Path("data").mkdir(exist_ok=True)
memory = SqliteSaver(sqlite3.connect("data/memory.db", check_same_thread=False))


# ---------- match graph ----------
class MatchState(TypedDict, total=False):
    resume_path: str
    filters: dict  # passed to Index.search: location, levels, company
    profile: dict
    query: str
    attempts: int
    hits: list[dict]
    relevant: list[dict]
    contexts: list[str]
    rating: dict
    report: str


class Grades(BaseModel):
    relevant_ids: list[str] = Field(description="ids of jobs that genuinely fit the candidate")


def parse(s: MatchState) -> MatchState:
    """VLM call, cached by file hash: re-running the same resume costs nothing."""
    PROFILE_CACHE.mkdir(parents=True, exist_ok=True)
    cached = PROFILE_CACHE / (hashlib.sha256(Path(s["resume_path"]).read_bytes()).hexdigest()[:16] + ".json")
    if cached.exists():
        p = json.loads(cached.read_text())
    else:
        p = vlm.read(s["resume_path"], vlm.Profile).model_dump()
        cached.write_text(json.dumps(p))
    return {"profile": p, "query": f"{' / '.join(p['target_roles'])}. Skills: {', '.join(p['skills'][:15])}",
            "attempts": 0}


def retrieve(s: MatchState) -> MatchState:
    junior = s["profile"]["years_experience"] < 2 or None  # students/new grads get entry-level ranking
    return {"hits": get_index().search(s["query"], k=10, junior=junior, **s.get("filters", {}))}


def grade(s: MatchState) -> MatchState:
    listing = "\n".join(f"[{j['id']}] {j['title']} at {j['company']}: {j['text'][:400]}" for j in s["hits"])
    g = llm().with_structured_output(Grades).invoke(
        f"Candidate: {s['profile']}\n\nJobs:\n{listing}\n\n"
        "Return the ids of jobs this candidate could realistically get interviewed for "
        "(seniority, skills and location must fit).")
    keep = set(g.relevant_ids)
    return {"relevant": [j for j in s["hits"] if j["id"] in keep]}


def route(s: MatchState) -> str:
    if len(s["relevant"]) >= MIN_RELEVANT or s["attempts"] >= MAX_REWRITES:
        return "analyze"
    return "rewrite"


def rewrite(s: MatchState) -> MatchState:
    rejected = ", ".join(j["title"] for j in s["hits"] if j not in s["relevant"])
    q = llm(0.3).invoke(
        f"Candidate: {s['profile']}\nSearch query '{s['query']}' returned mostly poor fits: {rejected}.\n"
        "Write one better job-search query (titles + key skills). Return only the query.").text
    return {"query": q.strip(), "attempts": s["attempts"] + 1}


class Score(BaseModel):
    score: int = Field(ge=0, le=25)
    reason: str = Field(description="one short sentence")


class JobFit(BaseModel):
    job_id: str
    fit: int = Field(ge=0, le=10)
    meets: list[str] = Field(description="requirements from this job the resume already shows")
    gaps: list[str] = Field(description="requirements from this job the resume lacks")
    tailored_bullet: str = Field(description="one resume bullet rewritten for this job, using only real experience")


class Rating(BaseModel):
    skills_match: Score = Field(description="how well the resume's skills match the JOBS below")
    impact: Score = Field(description="quantified results and metrics in bullets")
    ml_depth: Score = Field(description="depth of ML/AI work: models, RAG, agents, evals, deployment")
    ats_format: Score = Field(description="layout, section order, keywords, length, ATS parseability")
    top_fixes: list[str] = Field(description="the 3 highest-impact concrete edits, most important first")
    jobs: list[JobFit]
    skills_to_learn: list[str] = Field(description="top 3 skills to learn, based on the gaps across the jobs")

    @property
    def overall(self) -> int:
        return self.skills_match.score + self.impact.score + self.ml_depth.score + self.ats_format.score


def render(r: Rating, jobs: list[dict]) -> str:
    """Markdown gap analysis from the structured rating (also what RAGAS scores)."""
    by_id = {j["id"]: j for j in jobs}
    out = []
    for f in r.jobs:
        j = by_id.get(f.job_id, {"title": f.job_id, "company": "", "url": ""})
        out += [f"#### [{j['title']}]({j.get('url', '')}) · {j['company']} · fit {f.fit}/10 [{f.job_id}]",
                f"**Meets:** {', '.join(f.meets) or '-'}  ", f"**Gaps:** {', '.join(f.gaps) or '-'}  ",
                f"**Tailored bullet:** {f.tailored_bullet}", ""]
    out.append(f"**Top skills to learn:** {', '.join(r.skills_to_learn)}")
    return "\n".join(out)


def analyze(s: MatchState) -> MatchState:
    """One structured multimodal call: the model sees the resume itself (for format/impact) plus the
    retrieved job descriptions (RAG), and returns the rating and per-job gap analysis together."""
    jobs = (s["relevant"] or s["hits"])[:ANALYZE_JOBS]
    contexts = [f"[{j['id']}] {j['title']} at {j['company']} ({j['location']})\n{j['text'][:1500]}" for j in jobs]
    prompt = ("You are a strict technical recruiter for AI/ML roles. Rate the attached resume against ONLY the "
              "job descriptions below. Each category is out of 25; be calibrated (a typical student resume scores "
              "40-65 overall). Ground every gap and requirement in the jobs and use their ids.\n\n"
              "JOBS:\n" + "\n\n".join(contexts))
    msg = HumanMessage(content=[{"type": "text", "text": prompt}, vlm.file_block(Path(s["resume_path"]))])
    r = llm().with_structured_output(Rating).invoke([msg])
    rating = {**r.model_dump(), "overall": r.overall}
    return {"rating": rating, "report": render(r, jobs), "contexts": contexts}


g = StateGraph(MatchState)
for name, fn in [("parse", parse), ("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite),
                 ("analyze", analyze)]:
    g.add_node(name, fn)
g.add_edge(START, "parse")
g.add_edge("parse", "retrieve")
g.add_edge("retrieve", "grade")
g.add_conditional_edges("grade", route, ["rewrite", "analyze"])
g.add_edge("rewrite", "retrieve")
g.add_edge("analyze", END)
match_graph = g.compile(checkpointer=memory)


# ---------- chat graph ----------
@tool
def search_jobs(query: str, k: int = 5) -> str:
    """Search live job postings. Returns id, title, company, location and url."""
    return "\n".join(f"[{j['id']}] {j['title']} | {j['company']} | {j['location']} | {j['url']}"
                     for j in get_index().search(query, k))


@tool
def get_job(job_id: str) -> str:
    """Get the full description of one job by id."""
    j = get_index().jobs.get(job_id)
    return f"{j['title']} at {j['company']}\n{j['text']}" if j else "no such job id"


TOOLS = [search_jobs, get_job]
SYSTEM = ("You are JobPilot, a career agent. Use the tools to find and read real postings; never invent jobs. "
          "Remember the user's stated preferences (location, role, stack) from earlier in the conversation.")


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def agent(s: ChatState) -> ChatState:
    return {"messages": [llm().bind_tools(TOOLS).invoke([SystemMessage(SYSTEM)] + s["messages"])]}


c = StateGraph(ChatState)
c.add_node("agent", agent)
c.add_node("tools", ToolNode(TOOLS))
c.add_edge(START, "agent")
c.add_conditional_edges("agent", tools_condition)
c.add_edge("tools", "agent")
chat_graph = c.compile(checkpointer=memory)


if __name__ == "__main__":
    out = match_graph.invoke({"resume_path": sys.argv[1]}, {"configurable": {"thread_id": "cli"}})
    print(f"query: {out['query']}  (rewrites: {out['attempts']})\n\n{out['report']}")
