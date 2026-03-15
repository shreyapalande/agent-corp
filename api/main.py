import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from api.config import settings
from api.schemas import CachedReport, ResearchRequest, ResearchResponse, SourceItem, ValidationResult as ValidationSchema
from utils.cache import load_report, report_exists
from utils.export import parse_confidence_scores
from utils.tracing import configure_tracing

configure_tracing()

app = FastAPI(
    title="AgentCorp Sales Intelligence API",
    version=settings.api_version,
    description="Agentic B2B sales intelligence — powered by LangGraph, Tavily, and Groq.",
)

_executor = ThreadPoolExecutor(max_workers=4)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_graph(company: str) -> dict:
    """Execute the LangGraph pipeline synchronously (called in thread pool)."""
    from agent.graph import build_graph

    graph = build_graph()
    initial_state = {
        "company_name": company,
        "news_results": [],
        "funding_results": [],
        "techstack_results": [],
        "competitor_results": [],
        "people_results": [],
        "product_results": [],
        "brief": "",
        "all_sources": [],
        "is_first_run": False,
        "cached_report": "",
        "changes_detected": [],
        "last_searched": "",
        "validation_result": {},
    }
    return graph.invoke(initial_state)


def _build_response(company: str, state: dict) -> ResearchResponse:
    """Map LangGraph final state to a ResearchResponse."""
    brief = state.get("brief", "")

    # Extract per-section confidence scores
    sections = parse_confidence_scores(brief)
    confidence_scores = {
        heading: data["score"] for heading, data in sections.items()
    }

    # Raw results per dimension
    raw_results = {
        "news":        state.get("news_results", []),
        "funding":     state.get("funding_results", []),
        "techstack":   state.get("techstack_results", []),
        "competitors": state.get("competitor_results", []),
        "people":      state.get("people_results", []),
        "product":     state.get("product_results", []),
    }

    vr = state.get("validation_result", {})
    validation = ValidationSchema(
        is_valid=vr.get("is_valid", True),
        ungrounded_claims=vr.get("ungrounded_claims", []),
        incomplete_sections=vr.get("incomplete_sections", []),
        no_data_sections=vr.get("no_data_sections", []),
        overall_score=vr.get("overall_score", 1.0),
    )

    return ResearchResponse(
        company=company,
        brief=brief,
        sources=[SourceItem(**s) for s in state.get("all_sources", [])],
        changes=state.get("changes_detected", []),
        confidence_scores=confidence_scores,
        is_first_run=state.get("is_first_run", False),
        timestamp=datetime.now(timezone.utc).isoformat(),
        raw_results=raw_results,
        validation=validation,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": settings.api_version}


@app.post("/research", response_model=ResearchResponse)
async def run_research(req: ResearchRequest):
    """
    Run the full 6-node intelligence pipeline for a company.
    Blocks until synthesis and change detection are complete (~30s).
    """
    loop = asyncio.get_event_loop()
    try:
        final_state = await loop.run_in_executor(
            _executor, _run_graph, req.company
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _build_response(req.company, final_state)


@app.get("/research/{company}", response_model=CachedReport)
def get_cached_report(company: str):
    """
    Return the most recent cached report for a company.
    Returns 404 if no cache exists.
    """
    if not report_exists(company):
        raise HTTPException(
            status_code=404,
            detail=f"No cached report found for '{company}'. Run POST /research first.",
        )
    data = load_report(company)
    return CachedReport(
        company=data["company"],
        brief=data["brief"],
        sections=data.get("sections", {}),
        timestamp=data["timestamp"],
    )
