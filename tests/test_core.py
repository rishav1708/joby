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
        if isinstance(prompt, list):  # analyze: multimodal structured rating
            sc = graph.Score(score=10, reason="ok")
            return graph.Rating(skills_match=sc, impact=sc, ml_depth=sc, ats_format=sc, top_fixes=["a", "b", "c"],
                                jobs=[graph.JobFit(job_id="a", fit=7, meets=["pytorch"], gaps=["k8s"],
                                                   tailored_bullet="Built X")], skills_to_learn=["k8s"])
        return type("Msg", (), {"text": "ml engineer pytorch"})()


def test_match_graph_rewrites_then_analyzes(monkeypatch, tmp_path):
    ix = fake_index()
    resume = tmp_path / "r.pdf"
    resume.write_bytes(b"%PDF" + uuid.uuid4().bytes)  # unique bytes, so the profile cache misses
    monkeypatch.setattr(graph, "PROFILE_CACHE", tmp_path / "profiles")
    monkeypatch.setattr(graph, "get_index", lambda: ix)
    monkeypatch.setattr(graph, "llm", FakeLLM)
    monkeypatch.setattr(graph.vlm, "read", lambda path, schema: graph.vlm.Profile(
        headline="student", skills=["pytorch"], years_experience=0, target_roles=["ML Engineer"]))
    out = graph.match_graph.invoke({"resume_path": str(resume)}, {"configurable": {"thread_id": str(uuid.uuid4())}})
    assert out["attempts"] == 1 and out["query"] == "ml engineer pytorch"
    assert len(out["relevant"]) == 3 and len(out["contexts"]) == 3
    assert len(list((tmp_path / "profiles").iterdir())) == 1  # profile cached for next time
    assert out["rating"]["overall"] == 40 and "fit 7/10 [a]" in out["report"]


def test_build_resumes_and_skips_cached(tmp_path, monkeypatch):
    import json
    from jobpilot import retrieve
    calls = []
    fake = DeterministicFakeEmbedding(size=16)

    class Counting:
        def embed_documents(self, texts):
            calls.append(len(texts))
            return fake.embed_documents(texts)

        def embed_query(self, text):
            return fake.embed_query(text)

    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text("".join(json.dumps(j) + "\n" for j in JOBS))
    monkeypatch.setattr(retrieve, "JOBS_PATH", jobs_path)
    monkeypatch.setattr(retrieve, "VEC_PATH", tmp_path / "vectors.npz")
    monkeypatch.setattr(retrieve, "embedder", Counting)
    retrieve.Index.build(chunk=2)
    assert calls == [2, 1]  # saved in chunks
    retrieve.Index.build(chunk=2)
    assert calls == [2, 1]  # second build embeds nothing new


def test_seniority():
    from jobpilot.retrieve import seniority
    assert seniority("Machine Learning Engineer, New Grad") == "junior"
    assert seniority("ML Intern (Summer 2027)") == "junior"
    assert seniority("Staff ML Engineer") == "senior"
    assert seniority("Tech Lead, Inference") == "senior"
    assert seniority("Machine Learning Engineer") == "mid"
    assert seniority("Software Engineer III") == "senior"


def test_junior_query_ranks_entry_level_first():
    jobs = [
        {"id": "s1", "title": "Senior ML Engineer", "company": "x", "location": "", "text": "pytorch ml engineer"},
        {"id": "s2", "title": "Staff ML Engineer", "company": "x", "location": "", "text": "pytorch ml engineer"},
        {"id": "m1", "title": "ML Engineer", "company": "x", "location": "", "text": "pytorch ml engineer"},
        {"id": "j1", "title": "ML Engineer, New Grad", "company": "x", "location": "", "text": "pytorch"},
    ]
    emb = DeterministicFakeEmbedding(size=16)
    ix = Index(jobs, np.array(emb.embed_documents([j["text"] for j in jobs])), emb)
    order = ["s1", "s2", "m1", "j1"]  # reranker prefers senior roles, as on the real index
    ix.rerank = lambda q, ids, k=10: [d for d in order if d in ids][:k]
    junior = [j["id"] for j in ix.search("junior ML engineer pytorch", 4)]
    assert junior[0] == "j1" and junior[-2:] == ["s1", "s2"]
    assert [j["id"] for j in ix.search("ML engineer pytorch", 4)] == order  # no boost without junior intent


def test_file_block_detects_type_from_bytes(tmp_path):
    import pytest
    from jobpilot.vlm import file_block
    pdf = tmp_path / "resume"  # no extension
    pdf.write_bytes(b"%PDF-1.7 ...")
    assert file_block(pdf)["mime_type"] == "application/pdf"
    png = tmp_path / "shot.pdf"  # wrong extension
    png.write_bytes(b"\x89PNG\r\n\x1a\n...")
    assert file_block(png)["mime_type"] == "image/png"
    docx = tmp_path / "cv.docx"
    docx.write_bytes(b"PK\x03\x04...")
    with pytest.raises(ValueError, match="export it as PDF"):
        file_block(docx)
    with pytest.raises(FileNotFoundError):
        file_block(tmp_path / "missing.pdf")


def test_filters_and_company_cap():
    from jobpilot.retrieve import diversify, matches
    j = {"id": "gh-x-1", "title": "Senior ML Engineer", "company": "databricks", "location": "Bengaluru, India"}
    assert matches(j, "India") and not matches(j, "Remote")
    assert matches({**j, "id": "rm-9", "location": "Worldwide"}, "Remote")
    assert not matches(j, levels=["junior", "mid"]) and matches(j, levels=["senior"])
    assert not matches(j, company="stripe")
    ids = ["d1", "d2", "d3", "d4", "s1", "d5"]
    comp = {d: "databricks" if d[0] == "d" else "stripe" for d in ids}
    assert diversify(ids, comp, 4, per_company=2) == ["d1", "d2", "s1", "d3"]  # cap, then backfill in order


def test_llm_falls_back_on_overload(monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from jobpilot import llm as llm_mod

    class Overloaded(FakeListChatModel):
        def _call(self, *a, **k):
            raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")

    def fake(model, temperature):
        return Overloaded(responses=["x"]) if model == llm_mod.LLM_MODEL else FakeListChatModel(responses=["backup"])

    monkeypatch.setattr(llm_mod, "_gemini", fake)
    assert llm_mod.llm().invoke("hi").content == "backup"


def test_match_graph_applies_filters_on_every_retrieve(monkeypatch, tmp_path):
    from jobpilot.retrieve import matches
    jobs = [{"id": f"gh-{i}", "title": t, "company": c, "location": loc, "url": "", "text": "pytorch ml engineer"}
            for i, (t, c, loc) in enumerate([
                ("ML Engineer Intern", "a", "Bengaluru, India"), ("Senior ML Engineer", "b", "Bengaluru, India"),
                ("ML Engineer, New Grad", "c", "London, UK"), ("ML Engineer Intern", "d", "Pune, India"),
                ("Staff ML Engineer", "e", "Remote - USA")])]
    emb = DeterministicFakeEmbedding(size=16)
    ix = Index(jobs, np.array(emb.embed_documents([j["text"] for j in jobs])), emb)
    ix.rerank = lambda q, ids, k=10: ids[:k]
    seen = []
    real_search = ix.search
    ix.search = lambda *a, **k: seen.append(real_search(*a, **k)) or seen[-1]
    FakeLLM.calls = 0  # first grade rejects everything -> rewrite -> second retrieve
    monkeypatch.setattr(graph, "get_index", lambda: ix)
    monkeypatch.setattr(graph, "llm", FakeLLM)
    monkeypatch.setattr(graph, "PROFILE_CACHE", tmp_path / "profiles")
    monkeypatch.setattr(graph.vlm, "read", lambda path, schema: graph.vlm.Profile(
        headline="student", skills=["pytorch"], years_experience=0, target_roles=["ML Engineer"]))
    resume = tmp_path / "r.pdf"
    resume.write_bytes(b"%PDF" + uuid.uuid4().bytes)
    filters = {"location": "India", "levels": ["junior"], "company": None}
    out = graph.match_graph.invoke({"resume_path": str(resume), "filters": filters},
                                   {"configurable": {"thread_id": str(uuid.uuid4())}})
    assert out["attempts"] == 1 and len(seen) == 2  # retrieved before and after the rewrite
    for hits in seen:
        assert {j["id"] for j in hits} == {"gh-0", "gh-3"}
        assert all(matches(j, **filters) for j in hits)


def test_ingest_skips_dead_boards_and_reads_ashby(tmp_path, monkeypatch, capsys):
    from jobpilot import ingest
    ashby = {"jobs": [{"id": "7", "title": "ML Engineer", "location": "Bengaluru", "isRemote": False,
                       "jobUrl": "u", "descriptionPlain": "train models"}]}

    def fake_get(url):
        if "ashbyhq" in url and "openai" in url:
            return ashby
        raise OSError("HTTP Error 404: Not Found")

    monkeypatch.setattr(ingest, "get_json", fake_get)
    monkeypatch.setattr(ingest, "JOBS_PATH", tmp_path / "jobs.jsonl")
    monkeypatch.setattr(ingest, "GREENHOUSE", ["gone"])
    monkeypatch.setattr(ingest, "LEVER", [])
    monkeypatch.setattr(ingest, "ASHBY", ["openai"])
    ingest.main()
    out = capsys.readouterr().out
    assert "wrote 1 jobs from 1 companies" in out and "greenhouse:gone" in out and "remotive" in out
    assert '"company": "openai"' in (tmp_path / "jobs.jsonl").read_text()
