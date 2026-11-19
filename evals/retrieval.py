"""Retrieval eval: BM25 vs dense vs hybrid (RRF) vs hybrid + cross-encoder rerank.

1) python -m evals.retrieval --label   # pool top-10 from every method, Gemini grades 0/1/2 -> evals/qrels.jsonl
   (spot-check ~20% of the labels by hand and fix them; say so in the README)
2) python -m evals.retrieval           # prints Recall@10, nDCG@10, MRR per method
"""
import json
import math
import sys
from pathlib import Path

from pydantic import BaseModel

from jobpilot.llm import llm
from jobpilot.retrieve import doc_text, get_index

QUERIES = Path("evals/queries.txt")
QRELS = Path("evals/qrels.jsonl")
K = 10


def methods(ix):
    return {
        "bm25": lambda q: ix.bm25_search(q, K),
        "dense": lambda q: ix.dense_search(q, K),
        "hybrid_rrf": lambda q: ix.hybrid_search(q, K),
        "hybrid_rrf+rerank": lambda q: ix.rerank(q, ix.hybrid_search(q, 30), K),
        "+seniority_boost": lambda q: [j["id"] for j in ix.search(q, K)],
    }


def ndcg(ranked: list[str], rel: dict[str, int], k: int = K) -> float:
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall(ranked: list[str], rel: dict[str, int], k: int = K) -> float:
    good = {d for d, r in rel.items() if r > 0}
    return len(good & set(ranked[:k])) / len(good) if good else 0.0


def mrr(ranked: list[str], rel: dict[str, int]) -> float:
    return next((1 / (i + 1) for i, d in enumerate(ranked) if rel.get(d, 0) > 0), 0.0)


class Grade(BaseModel):
    grades: dict[str, int]  # job id -> 0 not relevant, 1 partly, 2 strong match


def label():
    ix = get_index()
    grader = llm().with_structured_output(Grade)
    with QRELS.open("w") as f:
        for q in QUERIES.read_text().splitlines():
            pool = {d for m in methods(ix).values() for d in m(q)}
            docs = "\n".join(f"[{d}] {doc_text(ix.jobs[d])[:500]}" for d in pool)
            g = grader.invoke(f"Query: {q}\nGrade each job 0 (irrelevant), 1 (partial), 2 (strong match).\n{docs}")
            f.write(json.dumps({"q": q, "rel": {d: g.grades.get(d, 0) for d in pool}}) + "\n")
            print(q, sum(v > 0 for v in g.grades.values()), "relevant")


def evaluate():
    ix = get_index()
    rows = [json.loads(line) for line in QRELS.open()]
    print(f"{'method':20} Recall@{K}  nDCG@{K}   MRR   (n={len(rows)} queries)")
    for name, fn in methods(ix).items():
        r = [(recall(d, x["rel"]), ndcg(d, x["rel"]), mrr(d, x["rel"])) for x in rows for d in [fn(x["q"])]]
        avg = [sum(col) / len(r) for col in zip(*r)]
        print(f"{name:20} {avg[0]:.3f}      {avg[1]:.3f}    {avg[2]:.3f}")


if __name__ == "__main__":
    label() if "--label" in sys.argv else evaluate()
