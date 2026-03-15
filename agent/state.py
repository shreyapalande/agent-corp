from typing import TypedDict


class SearchResult(TypedDict):
    title: str
    url: str
    content: str
    score: float


class AgentState(TypedDict):
    company_name: str
    news_results: list[SearchResult]
    funding_results: list[SearchResult]
    techstack_results: list[SearchResult]
    competitor_results: list[SearchResult]
    people_results: list[SearchResult]
    product_results: list[SearchResult]
    brief: str
    all_sources: list[dict]
    is_first_run: bool
    cached_report: str
    changes_detected: list[str]
    last_searched: str
