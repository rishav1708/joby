"""Hybrid retrieval over the jobs index: BM25 + dense (FAISS), fused with RRF, then cross-encoder rerank.

    python -m jobpilot.retrieve "junior ML engineer computer vision pytorch"
"""
import json
import re
import sys
from functools import cache
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .ingest import JOBS_PATH
from .llm import EMBED_MODEL, embedder

VEC_PATH = Path("data/vectors.npz")
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
JUNIOR = re.compile(r"\b(intern(ship)?|junior|jr|new[- ]grad(uate)?|graduate|entry[- ]level|early[- ]career|"
                    r"apprentice|trainee|fresher|student)\b", re.I)
SENIOR = re.compile(r"\b(senior|sr|staff|principal|lead|head|director|manager|architect|vp|distinguished|"
                    r"iii|iv|[3-9])\b", re.I)  # "Engineer III" / "Engineer 3" count as senior
# rank positions moved after rerank. Senior's shift is effectively a filter: senior roles only appear
# after every junior/mid candidate, as a fallback when there aren't k of those.
SHIFT = {"junior": -5, "mid": 0, "senior": 1000}


INDIA = re.compile(r"india|bengaluru|bangalore|hyderabad|pune|mumbai|delhi|gurgaon|gurugram|noida|chennai|kolkata",
                   re.I)


def seniority(title: str) -> str:
    return "junior" if JUNIOR.search(title) else "senior" if SENIOR.search(title) else "mid"


def seniority_boost(ids: list[str], titles: dict[str, str]) -> list[str]:
    """Entry-level searches: junior roles move up; senior/staff/lead roles go last (filter-first, with fallback)."""
    return [d for _, d in sorted((i + SHIFT[seniority(titles[d])], d) for i, d in enumerate(ids))]


def matches(j: dict, location: str = "Any", levels: list[str] | None = None, company: str | None = None) -> bool:
    """Hard filters. location: Any | India | Remote; levels: subset of junior/mid/senior."""
    if location == "India" and not INDIA.search(j["location"]):
        return False
    if location == "Remote" and not (j["id"].startswith("rm-") or "remote" in j["location"].lower()):
        return False
    if levels and seniority(j["title"]) not in levels:
        return False
    return not company or j["company"] == company


def diversify(ids: list[str], company: dict[str, str], k: int, per_company: int = 3) -> list[str]:
    """Keep ranking order but allow at most per_company jobs per company; backfill if that leaves < k."""
    seen: dict[str, int] = {}
    picked = []
    for d in ids:
        if seen.get(company[d], 0) < per_company:
            picked.append(d)
            seen[company[d]] = seen.get(company[d], 0) + 1
    return (picked + [d for d in ids if d not in picked])[:k]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+#.]+", text.lower())


def doc_text(j: dict) -> str:
    return f"{j['title']} at {j['company']} ({j['location']}). {j['text'][:2000]}"


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion: score(d) = sum 1/(k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class Index:
    def __init__(self, jobs: list[dict], vecs: np.ndarray, emb=None):
        self.jobs = {j["id"]: j for j in jobs}
        self.ids = [j["id"] for j in jobs]
        self.bm25 = BM25Okapi([tokenize(doc_text(j)) for j in jobs])
        vecs = np.ascontiguousarray(vecs, dtype="float32")
        faiss.normalize_L2(vecs)
        self.faiss = faiss.IndexFlatIP(vecs.shape[1])  # ponytail: exact search, swap to IVF/HNSW past ~1M jobs
        self.faiss.add(vecs)
        self.emb = emb or embedder()

    @classmethod
    def build(cls, chunk: int = 256) -> "Index":
        """Load jobs and embed only the ones not seen before (the index changes daily).
        Saves after every chunk, so an interrupted run resumes where it stopped."""
        jobs = [json.loads(line) for line in JOBS_PATH.open()]
        old = dict(np.load(VEC_PATH)) if VEC_PATH.exists() else {}
        same_model = str(old.get("model", "")) == EMBED_MODEL
        cached = dict(zip(old["ids"], old["vecs"])) if same_model else {}
        new = [j for j in jobs if j["id"] not in cached]
        emb = embedder()
        for i in range(0, len(new), chunk):
            batch = new[i:i + chunk]
            cached.update(zip([j["id"] for j in batch], emb.embed_documents([doc_text(j) for j in batch])))
            np.savez(VEC_PATH, ids=np.array(list(cached)), vecs=np.array(list(cached.values()), dtype="float32"),
                     model=EMBED_MODEL)
            print(f"embedded {min(i + chunk, len(new))}/{len(new)} new jobs")
        vecs = np.array([cached[j["id"]] for j in jobs], dtype="float32")
        np.savez(VEC_PATH, ids=np.array([j["id"] for j in jobs]), vecs=vecs, model=EMBED_MODEL)  # drop expired jobs
        return cls(jobs, vecs, emb)

    def bm25_search(self, query: str, k: int = 50) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        return [self.ids[i] for i in np.argsort(-scores)[:k]]

    def dense_search(self, query: str, k: int = 50) -> list[str]:
        q = np.array([self.emb.embed_query(query)], dtype="float32")
        faiss.normalize_L2(q)
        _, idx = self.faiss.search(q, k)
        return [self.ids[i] for i in idx[0] if i >= 0]

    def hybrid_search(self, query: str, k: int = 50) -> list[str]:
        return rrf([self.bm25_search(query, k), self.dense_search(query, k)])[:k]

    def rerank(self, query: str, ids: list[str], k: int = 10) -> list[str]:
        scores = reranker().predict([(query, doc_text(self.jobs[i])) for i in ids])
        return [ids[i] for i in np.argsort(-scores)[:k]]

    def search(self, query: str, k: int = 10, junior: bool | None = None, location: str = "Any",
               levels: list[str] | None = None, company: str | None = None, per_company: int = 3) -> list[dict]:
        """Hybrid retrieve -> filter -> cross-encoder rerank -> seniority boost -> per-company cap.
        junior=None infers entry-level intent from the query ("intern", "new grad", "junior", ...)."""
        if junior is None:
            junior = bool(JUNIOR.search(query))
        filtered = location != "Any" or bool(levels) or bool(company)
        pool = self.hybrid_search(query, 500 if filtered else 100)  # filters need a deep pool to survive
        pool = [d for d in pool if matches(self.jobs[d], location, levels, company)]
        pool = pool[:100] if junior or filtered else pool[:60]  # 60+ so the per-company cap has room
        ids = self.rerank(query, pool, len(pool)) if pool else []
        if junior:
            ids = seniority_boost(ids, {d: self.jobs[d]["title"] for d in ids})  # senior roles go last
        ids = ids[:k] if company else diversify(ids, {d: self.jobs[d]["company"] for d in ids}, k, per_company)
        return [self.jobs[d] for d in ids]


@cache
def reranker():
    from sentence_transformers import CrossEncoder  # lazy: pulls in torch
    return CrossEncoder(RERANKER)


@cache
def get_index() -> Index:
    return Index.build()


if __name__ == "__main__":
    for j in get_index().search(" ".join(sys.argv[1:])):
        print(f"{j['id']:40} {j['title']} | {j['company']} | {j['location']}")
