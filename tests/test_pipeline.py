"""
End-to-end pipeline test.

Runs the full 10-node LangGraph graph for a single company and asserts that
every stage produced meaningful output. Makes real Tavily and Gemini API calls.

Run with:
    pytest tests/test_pipeline.py -v -s
"""

import pytest
from dotenv import load_dotenv

load_dotenv()

from agent.graph import build_graph

COMPANY = "Notion"

# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline_output():
    """Run the full graph once and share the result across all assertions."""
    graph = build_graph()
    result = graph.invoke({"company_name": COMPANY})
    return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_results(output: dict, key: str) -> list[dict]:
    results = output.get(key, [])
    assert isinstance(results, list), f"{key} should be a list"
    for i, r in enumerate(results):
        for field in ("title", "url", "content", "score"):
            assert field in r, f"{key}[{i}] missing '{field}'"
    return results


# ── Search node outputs ────────────────────────────────────────────────────────

def test_news_results(pipeline_output):
    results = _check_results(pipeline_output, "news_results")
    print(f"\n[news] {len(results)} results | first: {results[0]['title'] if results else 'none'}")
    assert len(results) > 0, "news_node returned no results"


def test_funding_results(pipeline_output):
    results = _check_results(pipeline_output, "funding_results")
    print(f"\n[funding] {len(results)} results | first: {results[0]['title'] if results else 'none'}")
    assert len(results) > 0, "funding_node returned no results"


def test_techstack_results(pipeline_output):
    results = _check_results(pipeline_output, "techstack_results")
    print(f"\n[techstack] {len(results)} results | first: {results[0]['title'] if results else 'none'}")
    assert len(results) > 0, "techstack_node returned no results"


def test_competitor_results(pipeline_output):
    results = _check_results(pipeline_output, "competitor_results")
    print(f"\n[competitor] {len(results)} results | first: {results[0]['title'] if results else 'none'}")
    assert len(results) > 0, "competitor_node returned no results"


def test_people_results(pipeline_output):
    results = _check_results(pipeline_output, "people_results")
    print(f"\n[people] {len(results)} results | first: {results[0]['title'] if results else 'none'}")
    assert len(results) > 0, "people_node returned no results"


def test_product_results(pipeline_output):
    results = _check_results(pipeline_output, "product_results")
    print(f"\n[product] {len(results)} results | first: {results[0]['title'] if results else 'none'}")
    assert len(results) > 0, "product_node returned no results"


# ── Synthesis ──────────────────────────────────────────────────────────────────

def test_brief_generated(pipeline_output):
    brief = pipeline_output.get("brief", "")
    assert isinstance(brief, str), "brief should be a string"
    assert len(brief) > 500, f"brief is suspiciously short ({len(brief)} chars)"
    print(f"\n[brief] {len(brief)} chars | first 200: {brief[:200]}")


def test_brief_has_required_sections(pipeline_output):
    # Reuse the validator's result rather than re-checking the brief text
    vr = pipeline_output.get("validation_result", {})
    missing = vr.get("incomplete_sections", [])
    assert not missing, f"Brief missing required sections: {missing}"


def test_all_sources_populated(pipeline_output):
    sources = pipeline_output.get("all_sources", [])
    assert isinstance(sources, list), "all_sources should be a list"
    assert len(sources) > 0, "all_sources is empty — synthesize_node may not have extracted sources"
    print(f"\n[sources] {len(sources)} total sources")


# ── Validation ─────────────────────────────────────────────────────────────────

def test_validation_result_present(pipeline_output):
    vr = pipeline_output.get("validation_result")
    assert vr is not None, "validation_result missing from state"
    assert "is_valid" in vr, "validation_result missing 'is_valid'"
    assert "overall_score" in vr, "validation_result missing 'overall_score'"
    score = vr["overall_score"]
    print(f"\n[validation] score={score:.2f} | is_valid={vr['is_valid']}")
    print(f"             ungrounded={vr.get('ungrounded_claims', [])} | incomplete={vr.get('incomplete_sections', [])}")


def test_validation_score_acceptable(pipeline_output):
    vr = pipeline_output.get("validation_result", {})
    score = vr.get("overall_score", 0)
    assert score >= 0.5, f"Validation score too low: {score:.2f} (expected >= 0.50)"


# ── Change detection ───────────────────────────────────────────────────────────

def test_changes_detected_field_present(pipeline_output):
    changes = pipeline_output.get("changes_detected")
    assert changes is not None, "changes_detected missing from state"
    assert isinstance(changes, list), "changes_detected should be a list"
    print(f"\n[changes] {len(changes)} change(s) detected")
    for c in changes[:3]:
        print(f"  - {c}")


# ── Metadata ───────────────────────────────────────────────────────────────────

def test_last_searched_timestamp(pipeline_output):
    ts = pipeline_output.get("last_searched", "")
    assert ts, "last_searched timestamp missing"
    print(f"\n[meta] last_searched={ts}")
