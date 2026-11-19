# JobPilot

A multimodal job-matching agent. It reads a resume (PDF, scan or screenshot) with a vision-language model, retrieves matching **live** job postings with hybrid RAG, and a **LangGraph** agent grades, re-queries and writes a grounded gap analysis. Everything runs on the Gemini free tier.

```
resume.pdf / .png
      │
  [parse]  Gemini VLM ──► Profile (Pydantic, structured output)
      │
  [retrieve]  BM25 ─┐
              FAISS ┴─► RRF fusion ─► cross-encoder rerank (MiniLM)
      │
  [grade]  LLM judges each hit ──► < 3 relevant? ──► [rewrite] query ──┐
      │                                                               │
      │◄──────────────────────────── (max 2 rewrites) ───────────────┘
  [analyze]  RAG over the top job descriptions ─► fit score, gaps, tailored bullets (cited by job id)

chat agent: Gemini + tools (search_jobs, get_job), memory via LangGraph SQLite checkpointer
```

## What's in here

| Path | What it does |
|---|---|
| `jobpilot/ingest.py` | Pulls live postings from ~90 company boards on Greenhouse, Lever and Ashby (AI labs, product companies, India-based teams) plus Remotive, in parallel. Runs daily in CI. |
| `jobpilot/retrieve.py` | BM25 + local bge-small embeddings in FAISS, fused with reciprocal rank fusion, reranked by a cross-encoder, then a seniority-aware boost for entry-level searches. Only new postings get embedded. |
| `jobpilot/vlm.py` | Gemini vision reads resumes and job-post screenshots into Pydantic schemas. Also a text-layer/OCR baseline for comparison. |
| `jobpilot/graph.py` | Two LangGraph graphs: the corrective-RAG match pipeline and a tool-calling chat agent with persistent memory. |
| `evals/retrieval.py` | Recall@10, nDCG@10, MRR for BM25 vs dense vs hybrid vs hybrid+rerank. Labels by LLM pooling, then hand-checked. |
| `evals/vlm.py` | Skill-extraction F1: VLM vs pypdf/OCR + LLM baseline on hand-labelled docs. |
| `evals/rag.py` | RAGAS Faithfulness and Response Relevancy for the gap-analysis reports. |
| `app.py` | Gradio demo (deployable to Hugging Face Spaces). |

## Run it

On macOS (and most Linux), the command is `python3`, not `python`. Once the venv is active, `python` works too.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY=...            # free key from aistudio.google.com
python -m jobpilot.ingest            # fetch jobs -> data/jobs.jsonl
python -m jobpilot.retrieve "junior ML engineer pytorch"   # first run embeds the jobs locally (resumes if interrupted)
python -m jobpilot.graph my_resume.pdf
python app.py                        # home page at http://127.0.0.1:7860
```

Offline tests (no key needed, run inside the venv): `GOOGLE_API_KEY=x python -m pytest -q`

Models default to `gemini-3.5-flash` (chat + vision) and local `BAAI/bge-small-en-v1.5` embeddings (no API quota); override with `JOBPILOT_LLM` / `JOBPILOT_EMBED`. If Gemini is overloaded (503) or out of quota (429), calls fall back to `gemini-3.5-flash-lite` (`JOBPILOT_LLM_FALLBACK`, empty to disable).

## Results

_Fill in after running the evals._

| Retrieval (30 queries) | Recall@10 | nDCG@10 | MRR |
|---|---|---|---|
| BM25 | | | |
| Dense (bge-small + FAISS) | | | |
| Hybrid (RRF) | | | |
| Hybrid + rerank | | | |
| + seniority boost | | | |

| Extraction (n docs) | Skill F1 |
|---|---|
| pypdf / OCR + LLM | |
| Gemini VLM | |

| RAG (n resumes) | Faithfulness | Response relevancy |
|---|---|---|
| JobPilot | | |
