"""Deploy the app + a fresh job index to a Hugging Face Space (free CPU). Run by the GitHub Action.

    HF_TOKEN=... HF_SPACE=<user>/jobpilot python scripts/deploy_space.py pull   # reuse the Space's embeddings
    python -m jobpilot.ingest && python -c "from jobpilot.retrieve import Index; Index.build()"
    HF_TOKEN=... HF_SPACE=<user>/jobpilot python scripts/deploy_space.py push   # one commit -> one restart
"""
import os
import sys
from pathlib import Path

import gradio
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

SPACE = os.environ["HF_SPACE"]
api = HfApi(token=os.environ["HF_TOKEN"])
FILES = ["app.py", "requirements.txt", *map(str, Path("jobpilot").glob("*.py")), *map(str, Path("evals").glob("*.py")),
         "evals/results.json", "data/jobs.jsonl", "data/vectors.npz"]
FRONTMATTER = f"""---
title: JobPilot
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: {gradio.__version__}
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Resume-to-job matching with RAG, LangGraph and a VLM
---

"""


def pull():
    """Download the Space's previous embeddings so only new postings get embedded."""
    try:
        hf_hub_download(SPACE, "data/vectors.npz", repo_type="space", local_dir=".", token=api.token)
        print("reused data/vectors.npz from the Space")
    except Exception as e:  # first deploy: nothing to reuse
        print(f"no previous vectors ({type(e).__name__})")


def push():
    api.create_repo(SPACE, repo_type="space", space_sdk="gradio", exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=f) for f in FILES if Path(f).exists()]
    ops.append(CommitOperationAdd("README.md", (FRONTMATTER + Path("README.md").read_text()).encode()))
    api.create_commit(SPACE, ops, repo_type="space", commit_message="Deploy app and refreshed job index")
    print(f"deployed {len(ops)} files to https://huggingface.co/spaces/{SPACE}")


if __name__ == "__main__":
    {"pull": pull, "push": push}[sys.argv[1]]()
