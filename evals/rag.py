"""RAG eval with RAGAS: is the gap-analysis report grounded in the retrieved job descriptions?

Runs the full LangGraph match pipeline on every resume in evals/resumes/ and scores
Faithfulness (claims supported by the retrieved jobs) and ResponseRelevancy.

    python -m evals.rag
"""
import uuid
from pathlib import Path

from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy

from evals import save
from jobpilot.graph import match_graph
from jobpilot.llm import embedder, llm

RESUMES = Path("evals/resumes")
QUESTION = "Which of these jobs fit this candidate, what are the gaps, and how should they tailor their resume?"


def main():
    samples = []
    for f in sorted(RESUMES.iterdir()):
        out = match_graph.invoke({"resume_path": str(f)}, {"configurable": {"thread_id": str(uuid.uuid4())}})
        samples.append({"user_input": f"{QUESTION}\nCandidate: {out['profile']}",
                        "retrieved_contexts": out["contexts"], "response": out["report"]})
        print(f"{f.name}: rewrites={out['attempts']} relevant={len(out['relevant'])}")
    result = evaluate(EvaluationDataset.from_list(samples), metrics=[Faithfulness(), ResponseRelevancy()],
                      llm=llm(), embeddings=embedder())
    print(result)
    scores = result.to_pandas().select_dtypes("number").mean().to_dict()
    save("RAG grounding (RAGAS)", {"JobPilot": scores}, f"{len(samples)} resumes")


if __name__ == "__main__":
    main()
