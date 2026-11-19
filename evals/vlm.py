"""VLM extraction eval: Gemini vision vs text-layer/OCR + LLM baseline, on hand-labelled resumes and job posts.

evals/vlm_labels.jsonl, one line per file (put the files in evals/docs/):
  {"file": "evals/docs/resume_01.png", "kind": "resume", "skills": ["python", "pytorch", ...]}
  {"file": "evals/docs/post_07.png",   "kind": "job",    "title": "ML Engineer", "skills": [...]}

    python -m evals.vlm
"""
import json
from pathlib import Path

from jobpilot import vlm

LABELS = Path("evals/vlm_labels.jsonl")


def norm(xs) -> set[str]:
    return {x.strip().lower() for x in xs}


def f1(pred: set, gold: set) -> float:
    if not pred or not gold:
        return float(pred == gold)
    tp = len(pred & gold)
    return 2 * tp / (len(pred) + len(gold))


def skills_of(out) -> set[str]:
    return norm(out.skills if isinstance(out, vlm.Profile) else out.required_skills + out.nice_to_have)


def main():
    rows = [json.loads(line) for line in LABELS.open()]
    for name, reader in [("vlm (gemini vision)", vlm.read), ("baseline (pypdf/ocr + llm)", vlm.read_baseline)]:
        scores, failures = [], 0
        for r in rows:
            schema = vlm.Profile if r["kind"] == "resume" else vlm.JobPost
            try:
                scores.append(f1(skills_of(reader(r["file"], schema)), norm(r["skills"])))
            except Exception as e:
                failures += 1
                scores.append(0.0)
                print(f"  {name} failed on {r['file']}: {e}")
        print(f"{name:28} skill F1 = {sum(scores) / len(scores):.3f}  failures = {failures}/{len(rows)}")


if __name__ == "__main__":
    main()
