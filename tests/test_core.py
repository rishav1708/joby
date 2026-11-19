"""Offline checks (no API key needed): fusion, metrics, hybrid index, and the LangGraph corrective-retrieval loop.

    GOOGLE_API_KEY=x python -m pytest -q
"""
import uuid

import numpy as np
from langchain_core.embeddings import DeterministicFakeEmbedding

from evals.retrieval import mrr, ndcg, recall
from jobpilot import graph
from jobpilot.ingest import clean
from jobpilot.retrieve import Index, rrf

JOBS = [
    {"id": "a", "title": "ML Engineer", "company": "x", "location": "Remote", "text": "pytorch computer vision"},
    {"id": "b", "title": "Frontend Engineer", "company": "y", "location": "NYC", "text": "react typescript css"},
    {"id": "c", "title": "Data Engineer", "company": "z", "location": "Remote", "text": "spark airflow sql"},
]


def fake_index():
    emb = DeterministicFakeEmbedding(size=16)
    vecs = np.array(emb.embed_documents([j["text"] for j in JOBS]))
    ix = Index(JOBS, vecs, emb)
    ix.rerank = lambda q, ids, k=10: ids[:k]  # skip the cross-encoder download offline
    return ix


def test_clean_html():
    assert clean("&lt;p&gt;Hello&amp;nbsp;&lt;b&gt;world&lt;/b&gt;&lt;/p&gt;") == "Hello world"


def test_rrf_rewards_agreement():
    assert rrf([["a", "b", "c"], ["c", "b", "a"], ["b"]])[0] == "b"


def test_metrics():
    rel = {"a": 2, "b": 1}
    assert ndcg(["a", "b"], rel) == 1.0
    assert ndcg(["b", "a"], rel) < 1.0
    assert recall(["a", "x"], rel) == 0.5
    assert mrr(["x", "b"], rel) == 0.5


def test_index_search():
    ix = fake_index()
    assert ix.bm25_search("pytorch vision", 1) == ["a"]
    assert set(ix.dense_search("anything", 3)) == {"a", "b", "c"}
    assert ix.hybrid_search("spark sql", 3)[0] == "c"


class FakeLLM:
    """Grades nothing relevant on the first search, everything on the second, to force one rewrite."""
    calls = 0

    def __init__(self, *_):
        pass

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        if isinstance(prompt, str) and prompt.startswith("Candidate:") and "Jobs:" in prompt:
            FakeLLM.calls += 1
            return graph.Grades(relevant_ids=[] if FakeLLM.calls == 1 else ["a", "b", "c"])
        return type("Msg", (), {"text": "ml engineer pytorch"})()


def test_match_graph_rewrites_then_analyzes(monkeypatch):
    ix = fake_index()
    monkeypatch.setattr(graph, "get_index", lambda: ix)
    monkeypatch.setattr(graph, "llm", FakeLLM)
    monkeypatch.setattr(graph.vlm, "read", lambda path, schema: graph.vlm.Profile(
        headline="student", skills=["pytorch"], years_experience=0, target_roles=["ML Engineer"]))
    out = graph.match_graph.invoke({"resume_path": "r.pdf"}, {"configurable": {"thread_id": str(uuid.uuid4())}})
    assert out["attempts"] == 1 and out["query"] == "ml engineer pytorch"
    assert len(out["relevant"]) == 3 and len(out["contexts"]) == 3
