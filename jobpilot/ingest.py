"""Pull live postings from public job-board APIs into data/jobs.jsonl.

Run daily (cron / GitHub Actions); retrieve.Index.build() only embeds new postings.
    python -m jobpilot.ingest
"""
import html
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

JOBS_PATH = Path("data/jobs.jsonl")
# Board slugs as they appear in each ATS's public URL. Dead or renamed boards are skipped and
# listed at the end of a run, so this list can be generous.
GREENHOUSE = [
    # AI / ML labs and infra
    "anthropic", "deepmind", "scaleai", "databricks", "togetherai", "anyscale", "runwayml", "stabilityai",
    "huggingface", "pinecone", "weightsandbiases", "labelbox", "snorkelai", "assemblyai", "deepgram",
    # big tech-ish product companies hiring ML (many with India offices)
    "stripe", "figma", "discord", "robinhood", "coinbase", "gitlab", "airbnb", "dropbox", "pinterest",
    "datadog", "cloudflare", "mongodb", "elastic", "twilio", "reddit", "lyft", "instacart", "samsara",
    "brex", "affirm", "roblox", "asana", "hubspot", "intercom", "rubrik", "thoughtspot", "okta",
    # India-based or big India engineering hubs
    "postman", "phonepe", "browserstack", "groww", "inmobi", "sharechat", "uniphore", "druva",
    "chargebee", "hasura", "gojek", "zeta", "moengage", "atlan",
]
LEVER = ["palantir", "spotify", "zoox", "cred", "meesho", "razorpay", "zeta", "upstox", "jupiter",
         "swiggy", "dream11", "paytm", "highradius", "sprinklr", "mindtickle", "whatfix"]
ASHBY = ["openai", "cohere", "perplexity", "elevenlabs", "notion", "linear", "ramp", "replit", "supabase",
         "mistral", "characterai", "harvey", "glean", "sarvam", "krutrim", "wandb", "modal", "baseten",
         "fireworks-ai", "cursor", "lovable", "writer", "sierra"]


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "jobpilot/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def clean(text: str) -> str:
    """Greenhouse sends HTML-escaped HTML, so unescape, strip tags, unescape again."""
    text = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()[:8000]


FIELDS = ("id", "title", "company", "location", "url", "text", "posted_at")


def normalize(j: dict) -> dict:
    """Every field a plain string: boards send null/missing locations, dict locations, numeric ids."""
    out = {}
    for f in FIELDS:
        v = j.get(f)
        if isinstance(v, dict):
            v = v.get("name") or v.get("location") or ""
        out[f] = str(v).strip() if v is not None else ""
    return out


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


def ashby(org: str):
    data = get_json(f"https://api.ashbyhq.com/posting-api/job-board/{org}")
    for j in data.get("jobs", []):
        loc = j.get("location") or ""
        if j.get("isRemote") and "remote" not in loc.lower():
            loc = f"{loc} (Remote)".strip()
        yield {"id": f"ab-{org}-{j['id']}", "title": j["title"], "company": org, "location": loc,
               "url": j.get("jobUrl", ""), "text": clean(j.get("descriptionPlain") or j.get("descriptionHtml", "")),
               "posted_at": j.get("publishedAt", "")}


def remotive():
    for j in get_json("https://remotive.com/api/remote-jobs?category=software-dev")["jobs"]:
        yield {"id": f"rm-{j['id']}", "title": j["title"], "company": j["company_name"],
               "location": j.get("candidate_required_location", ""), "url": j["url"],
               "text": clean(j.get("description", "")), "posted_at": j.get("publication_date", "")}


def fetch(source):
    fn, arg = source
    try:
        return source, [n for j in (fn(arg) if arg else fn()) if (n := normalize(j))["text"] and n["title"]], None
    except Exception as e:  # one dead board must not kill the daily refresh
        return source, [], e


def main():
    sources = ([(greenhouse, b) for b in GREENHOUSE] + [(lever, c) for c in LEVER]
               + [(ashby, o) for o in ASHBY] + [(remotive, None)])
    jobs, dead = {}, []
    with ThreadPoolExecutor(16) as pool:  # ~90 boards: parallel keeps the refresh under a minute
        for (fn, arg), found, err in pool.map(fetch, sources):
            if err:
                dead.append(f"{fn.__name__}:{arg}")
            jobs.update((j["id"], j) for j in found)
    JOBS_PATH.parent.mkdir(exist_ok=True)
    JOBS_PATH.write_text("".join(json.dumps(j) + "\n" for j in jobs.values()))
    companies = {j["company"] for j in jobs.values()}
    print(f"wrote {len(jobs)} jobs from {len(companies)} companies to {JOBS_PATH}")
    if dead:
        print(f"skipped {len(dead)} boards that returned errors (renamed or not on that ATS): {', '.join(dead)}")


if __name__ == "__main__":
    main()
