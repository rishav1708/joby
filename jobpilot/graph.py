"""LangGraph agents.

match_graph: parse (VLM) -> retrieve (hybrid + rerank) -> grade (LLM) -> [rewrite -> retrieve]* -> analyze (RAG)
chat_graph:  tool-calling agent (search_jobs, get_job) with SQLite-checkpointed memory per thread_id.

    python -m jobpilot.graph resume.pdf
"""
import sqlite3
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from . import vlm
from .llm import llm
from .retrieve import get_index

MAX_REWRITES = 2
MIN_RELEVANT = 3
Path("data").mkdir(exist_ok=True)
memory = SqliteSaver(sqlite3.connect("data/memory.db", check_same_thread=False))


# ---------- match graph ----------
class MatchState(TypedDict, total=False):
    resume_path: str
    profile: dict
    query: str
    attempts: int
    hits: list[dict]
    relevant: list[dict]
    contexts: list[str]
    report: str


class Grades(BaseModel):
    relevant_ids: list[str] = Field(description="ids of jobs that genuinely fit the candidate")


def parse(s: MatchState) -> MatchState:
    p = vlm.read(s["resume_path"], vlm.Profile).model_dump()
    return {"profile": p, "query": f"{' / '.join(p['target_roles'])}. Skills: {', '.join(p['skills'][:15])}",
            "attempts": 0}


def retrieve(s: MatchState) -> MatchState:
    return {"hits": get_index().search(s["query"], k=10)}


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


def analyze(s: MatchState) -> MatchState:
    jobs = (s["relevant"] or s["hits"])[:5]
    contexts = [f"[{j['id']}] {j['title']} at {j['company']} ({j['location']})\n{j['text'][:2500]}" for j in jobs]
    context = "\n\n".join(contexts)
    report = llm().invoke(
        "You are a career coach. Using ONLY the job descriptions below, for each job give: a fit score 0-10, "
        "the requirements the candidate already meets, the gaps, and one resume bullet they could tailor. "
        "Cite the job id in brackets for every claim. Then list the top 3 skills to learn across all jobs.\n\n"
        f"CANDIDATE:\n{s['profile']}\n\nJOBS:\n{context}").text
    return {"report": report, "contexts": contexts}


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
