import json
import os
from datetime import datetime

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")


def _cache_key(company: str) -> str:
    return company.lower().replace(" ", "_")


def _cache_path(company: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{_cache_key(company)}.json")


def save_report(company: str, brief: str, sections: dict) -> None:
    """Persist the brief, timestamp, and per-section content to a JSON cache file."""
    payload = {
        "company": company,
        "brief": brief,
        "sections": sections,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(_cache_path(company), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_report(company: str) -> dict | None:
    """Load a cached report. Returns None if no cache exists."""
    path = _cache_path(company)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def report_exists(company: str) -> bool:
    return os.path.exists(_cache_path(company))
