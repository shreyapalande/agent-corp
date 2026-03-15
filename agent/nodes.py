import os
from tavily import TavilyClient
from .state import AgentState
from .prompts import SYNTHESIS_PROMPT


def _get_tavily() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not set in environment")
    return TavilyClient(api_key=api_key)


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
        try:
            response = client.search(
                query,
                max_results=max_results,
                search_depth="advanced",
                include_raw_content=False,
                include_domains=include_domains,
                days=days,
            )
            for r in response.get("results", []):
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append(
                        {
                            "title": r.get("title", "Untitled"),
                            "url": url,
                            "content": r.get("content", ""),
                            "score": r.get("score", 0.0),
                        }
                    )
        except Exception as e:
            print(f"[Tavily] Error on query '{query}': {e}")

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ── Search Nodes ──────────────────────────────────────────────────────────────

def news_node(state: AgentState) -> dict:
    company = state["company_name"]
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
    return {"news_results": _run_searches(client, queries, domains, days=7)}


def funding_node(state: AgentState) -> dict:
    company = state["company_name"]
    client = _get_tavily()
    queries = [
        f'"{company}" funding round Series investment valuation',
        f'"{company}" revenue growth investors financial health IPO acquisition',
    ]
    domains = [
        "crunchbase.com", "pitchbook.com", "tracxn.com", "dealroom.co",
        "techcrunch.com", "bloomberg.com", "sec.gov", "axios.com",
    ]
    return {"funding_results": _run_searches(client, queries, domains, days=90)}


def techstack_node(state: AgentState) -> dict:
    company = state["company_name"]
    client = _get_tavily()
    queries = [
        f'"{company}" tech stack technology tools infrastructure cloud',
        f'"{company}" engineering blog job posting software developer hiring',
    ]
    domains = [
        "stackshare.io", "github.com", "dev.to", "medium.com",
        "builtwith.com", "npmjs.com", "pypi.org",
    ]
    return {"techstack_results": _run_searches(client, queries, domains, days=180)}


def competitor_node(state: AgentState) -> dict:
    company = state["company_name"]
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
    return {"competitor_results": _run_searches(client, queries, domains, days=30)}


def people_node(state: AgentState) -> dict:
    company = state["company_name"]
    client = _get_tavily()
    queries = [
        f'"{company}" CEO founder CTO executive leadership team',
        f'"{company}" VP director hire executive LinkedIn profile',
    ]
    domains = [
        "linkedin.com", "twitter.com", "x.com", "crunchbase.com",
        "bloomberg.com", "forbes.com", "medium.com", "substack.com",
    ]
    return {"people_results": _run_searches(client, queries, domains, days=30)}


def product_node(state: AgentState) -> dict:
    company = state["company_name"]
    client = _get_tavily()
    queries = [
        f'"{company}" product review user sentiment rating experience',
        f'"{company}" product listing features pricing plans',
    ]
    domains = [
        "producthunt.com", "g2.com", "capterra.com", "getapp.com",
        "trustradius.com", "trustpilot.com", "reddit.com", "appsumo.com",
    ]
    return {"product_results": _run_searches(client, queries, domains, days=30)}


# ── Synthesis Node ────────────────────────────────────────────────────────────

def synthesize_node(state: AgentState) -> dict:
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment")

    def _format_section(results: list[dict], max_items: int = 5) -> str:
        if not results:
            return "No data retrieved for this dimension."
        parts = []
        for r in results[:max_items]:
            snippet = r["content"][:500].strip()
            parts.append(f"**{r['title']}**\n{snippet}\nSource: {r['url']}")
        return "\n\n".join(parts)

    company = state["company_name"]

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

    prompt = SYNTHESIS_PROMPT.format(
        company_name=company,
        news_section=_format_section(state.get("news_results", [])),
        funding_section=_format_section(state.get("funding_results", [])),
        techstack_section=_format_section(state.get("techstack_results", [])),
        competitor_section=_format_section(state.get("competitor_results", [])),
        people_section=_format_section(state.get("people_results", [])),
        product_section=_format_section(state.get("product_results", [])),
    )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=4000,
    )

    brief = response.choices[0].message.content

    # Persist to cache
    from utils.export import parse_confidence_scores
    from utils.cache import save_report
    sections = parse_confidence_scores(brief)
    save_report(company, brief, {k: v["content"] for k, v in sections.items()})

    return {"brief": brief, "all_sources": all_sources}


# ── Change Detection Node ─────────────────────────────────────────────────────

def change_detection_node(state: AgentState) -> dict:
    from groq import Groq
    from utils.cache import load_report, report_exists
    from datetime import datetime, timezone

    company = state["company_name"]
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Load the *previous* cache (saved before this run's synthesize_node overwrote it)
    # synthesize_node already saved the new report, so we compare new news vs cached news section
    cached = load_report(company)

    # On first run there is no prior cache to compare against
    if not cached or not cached.get("sections"):
        return {
            "is_first_run": True,
            "cached_report": "",
            "changes_detected": [],
            "last_searched": timestamp,
        }

    old_news = cached["sections"].get("Recent Activity & Sales Triggers", "")
    if not old_news:
        # Try alternate heading names the LLM may have used
        for key in cached["sections"]:
            if "news" in key.lower() or "activity" in key.lower() or "trigger" in key.lower():
                old_news = cached["sections"][key]
                break

    new_news_results = state.get("news_results", [])
    new_news = "\n\n".join(
        f"{r['title']}: {r['content'][:300]}" for r in new_news_results[:5]
    )

    if not old_news or not new_news:
        return {
            "is_first_run": False,
            "cached_report": cached.get("brief", ""),
            "changes_detected": [],
            "last_searched": cached.get("timestamp", "unknown"),
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

    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": change_prompt}],
        temperature=0.2,
        max_tokens=800,
    )

    raw = response.choices[0].message.content.strip()

    if "no significant changes detected" in raw.lower():
        changes: list[str] = []
    else:
        # Parse bullet lines into a clean list
        changes = [
            line.lstrip("-•* ").strip()
            for line in raw.splitlines()
            if line.strip().startswith(("-", "•", "*")) or line.strip()
        ]
        changes = [c for c in changes if c]

    return {
        "is_first_run": False,
        "cached_report": cached.get("brief", ""),
        "changes_detected": changes,
        "last_searched": cached.get("timestamp", "unknown"),
    }
