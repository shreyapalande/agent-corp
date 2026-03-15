import time
from tavily import TavilyClient
from langsmith import traceable
from api.config import settings
from .state import AgentState
from .prompts import SYNTHESIS_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


def _get_tavily() -> TavilyClient:
    if not settings.tavily_api_key:
        raise ValueError("TAVILY_API_KEY not set in environment")
    return TavilyClient(api_key=settings.tavily_api_key)


@traceable(run_type="retriever", name="tavily_search")
def _run_searches(
    client: TavilyClient,
    queries: list[str],
    include_domains: list[str],
    days: int = 30,
    max_results: int = 5,
) -> list[dict]:
    """Run multiple Tavily queries with domain filtering and return deduplicated results."""
    seen_urls: set[str] = set()
    results: list[dict] = []

    for query in queries:
        t0 = time.perf_counter()
        try:
            response = client.search(
                query,
                max_results=max_results,
                search_depth="advanced",
                include_raw_content=False,
                include_domains=include_domains,
                days=days,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            raw_hits = response.get("results", [])
            new_count = 0

            for r in raw_hits:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    new_count += 1
                    results.append(
                        {
                            "title": r.get("title", "Untitled"),
                            "url": url,
                            "content": r.get("content", ""),
                            "score": r.get("score", 0.0),
                        }
                    )

            logger.info(
                "Tavily query completed | query=%r | domains=%s | days=%d "
                "| hits=%d | new=%d | elapsed_ms=%.0f",
                query,
                ",".join(include_domains[:3]) + ("…" if len(include_domains) > 3 else ""),
                days,
                len(raw_hits),
                new_count,
                elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                "Tavily query failed | query=%r | elapsed_ms=%.0f | error=%s",
                query,
                elapsed_ms,
                e,
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    logger.debug("Tavily batch done | total_unique_results=%d", len(results))
    return results


# ── Cache Loader Node ─────────────────────────────────────────────────────────

def load_cache_node(state: AgentState) -> dict:
    """Runs at START (in parallel with search nodes) to snapshot the old cache
    into state BEFORE synthesize_node overwrites it."""
    from utils.cache import load_report
    from datetime import datetime, timezone

    company = state["company_name"]
    cached = load_report(company)
    if not cached:
        logger.debug("load_cache_node | company=%r | no existing cache", company)
        return {
            "cached_report": "",
            "last_searched": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }
    logger.debug(
        "load_cache_node | company=%r | found cache from %s",
        company,
        cached.get("timestamp", "unknown"),
    )
    return {
        "cached_report": cached.get("brief", ""),
        "last_searched": cached.get("timestamp", ""),
    }


# ── Search Nodes ──────────────────────────────────────────────────────────────

def news_node(state: AgentState) -> dict:
    company = state["company_name"]
    logger.info("news_node started | company=%r", company)
    client = _get_tavily()
    queries = [
        f'"{company}" latest news announcement 2024 2025',
        f'"{company}" press release product launch partnership expansion',
    ]
    domains = [
        "techcrunch.com", "reuters.com", "bloomberg.com", "forbes.com",
        "businessinsider.com", "theverge.com", "wired.com",
        "venturebeat.com", "inc.com", "fastcompany.com",
    ]
    results = _run_searches(client, queries, domains, days=7)
    logger.info("news_node done | company=%r | results=%d", company, len(results))
    return {"news_results": results}


def funding_node(state: AgentState) -> dict:
    company = state["company_name"]
    logger.info("funding_node started | company=%r", company)
    client = _get_tavily()
    queries = [
        f'"{company}" funding round Series investment valuation',
        f'"{company}" revenue growth investors financial health IPO acquisition',
    ]
    domains = [
        "crunchbase.com", "pitchbook.com", "tracxn.com", "dealroom.co",
        "techcrunch.com", "bloomberg.com", "sec.gov", "axios.com",
    ]
    results = _run_searches(client, queries, domains, days=90)
    logger.info("funding_node done | company=%r | results=%d", company, len(results))
    return {"funding_results": results}


def techstack_node(state: AgentState) -> dict:
    company = state["company_name"]
    logger.info("techstack_node started | company=%r", company)
    client = _get_tavily()
    queries = [
        f'"{company}" tech stack technology tools infrastructure cloud',
        f'"{company}" engineering blog job posting software developer hiring',
    ]
    domains = [
        "stackshare.io", "github.com", "dev.to", "medium.com",
        "builtwith.com", "npmjs.com", "pypi.org",
    ]
    results = _run_searches(client, queries, domains, days=180)
    logger.info("techstack_node done | company=%r | results=%d", company, len(results))
    return {"techstack_results": results}


def competitor_node(state: AgentState) -> dict:
    company = state["company_name"]
    logger.info("competitor_node started | company=%r", company)
    client = _get_tavily()
    queries = [
        f'"{company}" competitors alternatives market landscape comparison',
        f'"{company}" vs competitor differentiation market share positioning',
    ]
    domains = [
        "g2.com", "capterra.com", "getapp.com", "trustradius.com",
        "similarweb.com", "producthunt.com", "alternativeto.net",
        "gartner.com", "forrester.com",
    ]
    results = _run_searches(client, queries, domains, days=30)
    logger.info("competitor_node done | company=%r | results=%d", company, len(results))
    return {"competitor_results": results}


def people_node(state: AgentState) -> dict:
    company = state["company_name"]
    logger.info("people_node started | company=%r", company)
    client = _get_tavily()
    queries = [
        f'"{company}" CEO founder CTO executive leadership team',
        f'"{company}" VP director hire executive LinkedIn profile',
    ]
    domains = [
        "linkedin.com", "twitter.com", "x.com", "crunchbase.com",
        "bloomberg.com", "forbes.com", "medium.com", "substack.com",
    ]
    results = _run_searches(client, queries, domains, days=30)
    logger.info("people_node done | company=%r | results=%d", company, len(results))
    return {"people_results": results}


def product_node(state: AgentState) -> dict:
    company = state["company_name"]
    logger.info("product_node started | company=%r", company)
    client = _get_tavily()
    queries = [
        f'"{company}" product review user sentiment rating experience',
        f'"{company}" product listing features pricing plans',
    ]
    domains = [
        "producthunt.com", "g2.com", "capterra.com", "getapp.com",
        "trustradius.com", "trustpilot.com", "reddit.com", "appsumo.com",
    ]
    results = _run_searches(client, queries, domains, days=30)
    logger.info("product_node done | company=%r | results=%d", company, len(results))
    return {"product_results": results}


# ── LLM Helpers (traceable) ───────────────────────────────────────────────────

@traceable(run_type="llm", name="groq_synthesis")
def _call_groq_synthesis(prompt: str, api_key: str) -> str:
    """Calls Groq/Llama to synthesize the full sales brief. Traced as an LLM span."""
    from groq import Groq
    client = Groq(api_key=api_key)
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=4000,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    usage = response.usage
    logger.info(
        "LLM call | fn=groq_synthesis | model=llama-3.3-70b-versatile"
        " | prompt_tokens=%d | completion_tokens=%d | total_tokens=%d | elapsed_ms=%.0f",
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
        elapsed_ms,
    )
    return response.choices[0].message.content


@traceable(run_type="llm", name="groq_change_detection")
def _call_groq_change_detection(prompt: str, api_key: str) -> str:
    """Calls Groq/Llama to diff old vs new news. Traced as an LLM span."""
    from groq import Groq
    client = Groq(api_key=api_key)
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    usage = response.usage
    logger.info(
        "LLM call | fn=groq_change_detection | model=llama-3.3-70b-versatile"
        " | prompt_tokens=%d | completion_tokens=%d | total_tokens=%d | elapsed_ms=%.0f",
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
        elapsed_ms,
    )
    return response.choices[0].message.content


# ── Synthesis Node ────────────────────────────────────────────────────────────

def synthesize_node(state: AgentState) -> dict:
    api_key = settings.groq_api_key
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment")

    company = state["company_name"]
    logger.info("synthesize_node started | company=%r", company)

    def _format_section(results: list[dict], max_items: int = 5) -> str:
        if not results:
            return "No data retrieved for this dimension."
        parts = []
        for r in results[:max_items]:
            snippet = r["content"][:500].strip()
            parts.append(f"**{r['title']}**\n{snippet}\nSource: {r['url']}")
        return "\n\n".join(parts)

    # Collect all unique sources across every dimension
    all_sources: list[dict] = []
    seen_source_urls: set[str] = set()
    dimension_map = {
        "news":        state.get("news_results", []),
        "funding":     state.get("funding_results", []),
        "techstack":   state.get("techstack_results", []),
        "competitors": state.get("competitor_results", []),
        "people":      state.get("people_results", []),
        "product":     state.get("product_results", []),
    }
    for dimension, results in dimension_map.items():
        for r in results:
            if r.get("url") and r["url"] not in seen_source_urls:
                seen_source_urls.add(r["url"])
                all_sources.append(
                    {"title": r["title"], "url": r["url"], "dimension": dimension}
                )

    logger.debug(
        "synthesize_node | company=%r | total_unique_sources=%d",
        company,
        len(all_sources),
    )

    prompt = SYNTHESIS_PROMPT.format(
        company_name=company,
        news_section=_format_section(state.get("news_results", [])),
        funding_section=_format_section(state.get("funding_results", [])),
        techstack_section=_format_section(state.get("techstack_results", [])),
        competitor_section=_format_section(state.get("competitor_results", [])),
        people_section=_format_section(state.get("people_results", [])),
        product_section=_format_section(state.get("product_results", [])),
    )

    brief = _call_groq_synthesis(prompt, api_key)

    # Persist to cache
    from utils.export import parse_confidence_scores
    from utils.cache import save_report
    sections = parse_confidence_scores(brief)
    save_report(company, brief, {k: v["content"] for k, v in sections.items()})

    logger.info(
        "synthesize_node done | company=%r | brief_chars=%d | sources=%d",
        company,
        len(brief),
        len(all_sources),
    )
    return {"brief": brief, "all_sources": all_sources}


# ── Validation Node ───────────────────────────────────────────────────────────

def validation_node(state: AgentState) -> dict:
    """
    Runs after synthesize_node. Checks source grounding, completeness, and staleness.
    """
    from utils.validator import validate_report

    company = state["company_name"]
    logger.info("validation_node started | company=%r", company)

    result = validate_report(
        brief=state.get("brief", ""),
        all_sources=state.get("all_sources", []),
        news_results=state.get("news_results", []),
        funding_results=state.get("funding_results", []),
        techstack_results=state.get("techstack_results", []),
        competitor_results=state.get("competitor_results", []),
        people_results=state.get("people_results", []),
        product_results=state.get("product_results", []),
        groq_api_key=settings.groq_api_key,
    )

    log_msg = (
        "validation_node done | company=%r | score=%.2f | is_valid=%s"
        " | ungrounded=%d | incomplete=%s | no_data=%s"
    )
    log_args = (
        company,
        result.overall_score,
        result.is_valid,
        len(result.ungrounded_claims),
        result.incomplete_sections,
        result.no_data_sections,
    )

    if result.overall_score < 0.7:
        logger.warning(log_msg, *log_args)
    else:
        logger.info(log_msg, *log_args)

    return {"validation_result": result.to_dict()}


# ── Change Detection Node ─────────────────────────────────────────────────────

def change_detection_node(state: AgentState) -> dict:
    # Use the old brief that load_cache_node stashed in state BEFORE
    # synthesize_node overwrote the cache file — this is the correct prior snapshot.
    old_brief = state.get("cached_report", "")
    company = state["company_name"]

    # No previous cache means this is the first run
    if not old_brief:
        logger.info("change_detection_node | company=%r | first_run=True", company)
        return {
            "is_first_run": True,
            "changes_detected": [],
        }

    # Extract old news section from the cached brief text
    import re
    old_news = ""
    news_match = re.search(
        r"##\s+Recent Activity.*?\n(.*?)(?=\n##|\Z)", old_brief, re.DOTALL | re.IGNORECASE
    )
    if news_match:
        old_news = news_match.group(1).strip()

    new_news_results = state.get("news_results", [])
    new_news = "\n\n".join(
        f"{r['title']}: {r['content'][:300]}" for r in new_news_results[:5]
    )

    if not old_news or not new_news:
        logger.info(
            "change_detection_node | company=%r | skipped (no comparable news sections)",
            company,
        )
        return {
            "is_first_run": False,
            "changes_detected": [],
        }

    change_prompt = (
        "You are comparing two versions of a company intelligence report.\n\n"
        f"OLD REPORT (cached):\n{old_news}\n\n"
        f"NEW REPORT (fresh):\n{new_news}\n\n"
        "Identify what has meaningfully changed. Focus on:\n"
        "- New funding, acquisitions, or partnerships\n"
        "- Leadership changes\n"
        "- New product launches or shutdowns\n"
        "- Significant press coverage that wasn't there before\n\n"
        "Return a bullet list of changes. "
        "If nothing meaningful changed, say exactly: No significant changes detected."
    )

    api_key = settings.groq_api_key
    raw = _call_groq_change_detection(change_prompt, api_key).strip()

    if "no significant changes detected" in raw.lower():
        changes: list[str] = []
    else:
        changes = [
            line.lstrip("-•* ").strip()
            for line in raw.splitlines()
            if line.strip().startswith(("-", "•", "*")) or line.strip()
        ]
        changes = [c for c in changes if c]

    logger.info(
        "change_detection_node done | company=%r | changes_found=%d",
        company,
        len(changes),
    )
    return {
        "is_first_run": False,
        "changes_detected": changes,
    }
