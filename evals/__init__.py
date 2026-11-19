import json
from pathlib import Path

RESULTS = Path("evals/results.json")


def save(name: str, table: dict, note: str = "") -> None:
    """Store an eval table ({row: {metric: value}}) so the app's home page can show it."""
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[name] = {"note": note, "rows": table}
    RESULTS.write_text(json.dumps(data, indent=2))
