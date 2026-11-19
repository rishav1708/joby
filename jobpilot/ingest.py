"""Pull live postings from public job-board APIs into data/jobs.jsonl.

Run daily (cron / GitHub Actions); retrieve.Index.build() only embeds new postings.
    python -m jobpilot.ingest
"""
import html
import json
import re
import urllib.request
from pathlib import Path

JOBS_PATH = Path("data/jobs.jsonl")
GREENHOUSE = ["anthropic", "databricks", "scaleai", "stripe", "figma", "discord",
              "robinhood", "coinbase", "gitlab", "airbnb", "dropbox", "pinterest"]
LEVER = ["palantir", "plaid", "spotify"]


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "jobpilot/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def clean(text: str) -> str:
    """Greenhouse sends HTML-escaped HTML, so unescape, strip tags, unescape again."""
    text = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()[:8000]


def greenhouse(board: str):
    for j in get_json(f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true")["jobs"]:
        yield {"id": f"gh-{board}-{j['id']}", "title": j["title"], "company": board,
               "location": (j.get("location") or {}).get("name", ""), "url": j["absolute_url"],
               "text": clean(j.get("content", "")), "posted_at": j.get("updated_at", "")}


def lever(company: str):
    for j in get_json(f"https://api.lever.co/v0/postings/{company}?mode=json"):
        yield {"id": f"lv-{company}-{j['id']}", "title": j["text"], "company": company,
               "location": j.get("categories", {}).get("location", ""), "url": j["hostedUrl"],
               "text": clean(j.get("descriptionPlain") or j.get("description", "")),
               "posted_at": str(j.get("createdAt", ""))}


def remotive():
    for j in get_json("https://remotive.com/api/remote-jobs?category=software-dev")["jobs"]:
        yield {"id": f"rm-{j['id']}", "title": j["title"], "company": j["company_name"],
               "location": j.get("candidate_required_location", ""), "url": j["url"],
               "text": clean(j.get("description", "")), "posted_at": j.get("publication_date", "")}


def main():
    sources = [(greenhouse, b) for b in GREENHOUSE] + [(lever, c) for c in LEVER] + [(remotive, None)]
    jobs = {}
    for fn, arg in sources:
        try:
            for j in (fn(arg) if arg else fn()):
                if j["text"]:
                    jobs[j["id"]] = j
        except Exception as e:  # one dead board must not kill the daily refresh
            print(f"skip {fn.__name__}:{arg} ({e})")
    JOBS_PATH.parent.mkdir(exist_ok=True)
    JOBS_PATH.write_text("".join(json.dumps(j) + "\n" for j in jobs.values()))
    print(f"wrote {len(jobs)} jobs to {JOBS_PATH}")


if __name__ == "__main__":
    main()
