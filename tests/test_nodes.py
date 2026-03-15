"""
Integration tests for the 6 LangGraph search nodes.

These tests make real Tavily API calls. Run with:

    pytest tests/test_nodes.py -v -s

Each test:
  - Calls one search node with company_name='Notion'
  - Prints the number of results returned and the first result title
  - Asserts the result key exists and is a list
  - Asserts each result contains the required fields (title, url, content, score)
"""

import pytest
from dotenv import load_dotenv

load_dotenv()  # load TAVILY_API_KEY and GROQ_API_KEY from .env before importing nodes

from agent.nodes import (
    news_node,
    funding_node,
    techstack_node,
    competitor_node,
    people_node,
    product_node,
)

COMPANY = "Notion"

# ── Shared fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def state():
    """Minimal AgentState for a single-company search."""
    return {"company_name": COMPANY}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_summary(node_name: str, results: list[dict]) -> None:
    count = len(results)
    print(f"\n[{node_name}] results returned: {count}")
    if count > 0:
        print(f"[{node_name}] first result title: {results[0]['title']}")
    else:
        print(f"[{node_name}] no results found")


def _assert_result_shape(results: list[dict], node_name: str) -> None:
    """Every result must have the four fields _run_searches produces."""
    for i, r in enumerate(results):
        assert "title" in r,   f"{node_name}[{i}] missing 'title'"
        assert "url" in r,     f"{node_name}[{i}] missing 'url'"
        assert "content" in r, f"{node_name}[{i}] missing 'content'"
        assert "score" in r,   f"{node_name}[{i}] missing 'score'"


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_news_node(state):
    output = news_node(state)

    assert "news_results" in output
    results = output["news_results"]
    assert isinstance(results, list)

    _print_summary("news_node", results)
    _assert_result_shape(results, "news_node")


def test_funding_node(state):
    output = funding_node(state)

    assert "funding_results" in output
    results = output["funding_results"]
    assert isinstance(results, list)

    _print_summary("funding_node", results)
    _assert_result_shape(results, "funding_node")


def test_techstack_node(state):
    output = techstack_node(state)

    assert "techstack_results" in output
    results = output["techstack_results"]
    assert isinstance(results, list)

    _print_summary("techstack_node", results)
    _assert_result_shape(results, "techstack_node")


def test_competitor_node(state):
    output = competitor_node(state)

    assert "competitor_results" in output
    results = output["competitor_results"]
    assert isinstance(results, list)

    _print_summary("competitor_node", results)
    _assert_result_shape(results, "competitor_node")


def test_people_node(state):
    output = people_node(state)

    assert "people_results" in output
    results = output["people_results"]
    assert isinstance(results, list)

    _print_summary("people_node", results)
    _assert_result_shape(results, "people_node")


def test_product_node(state):
    output = product_node(state)

    assert "product_results" in output
    results = output["product_results"]
    assert isinstance(results, list)

    _print_summary("product_node", results)
    _assert_result_shape(results, "product_node")
