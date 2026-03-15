"""
Validation module for AgentCorp research briefs.

Three checks:
1. Source Grounding  — LLM verifies each claim has supporting evidence in sources
2. Completeness      — Ensures all 6 required sections are present in the brief
3. Staleness         — Flags dimensions where Tavily returned zero results
"""

from __future__ import annotations

import time

from utils.logger import get_logger
from utils.validation_result import ValidationResult

logger = get_logger(__name__)

REQUIRED_SECTIONS = [
    "Company Snapshot",
    "Recent Signals",
    "Tech Stack",
    "Funding & Growth",
    "Key People",
    "Product Sentiment",
]

_GROUNDING_PROMPT = """\
You are a fact-checking assistant.

Below is a sales intelligence brief about a company, followed by the raw source snippets \
that were used to write it.

Your job: identify claims in the brief that are NOT supported by any of the sources.
A claim is ungrounded if it asserts a specific fact (number, event, name, date, product) \
that does not appear in any source snippet.

Return ONLY a JSON array of short quoted claim strings that are ungrounded.
If every claim is supported, return an empty array: []

Example output:
["The company raised $50M in Series C", "CEO John Smith joined in 2023"]

Brief:
{brief}

Sources:
{sources}
"""


# ── Check 1: Source Grounding ─────────────────────────────────────────────────

def check_source_grounding(
    brief: str,
    all_sources: list[dict],
    groq_api_key: str,
    max_sources: int = 20,
) -> list[str]:
    """
    Ask Groq to identify claims in the brief that cannot be traced to any source.
    Returns a list of ungrounded claim strings.
    """
    if not groq_api_key or not brief or not all_sources:
        logger.debug("check_source_grounding | skipped (missing api_key, brief, or sources)")
        return []

    source_text = "\n\n".join(
        f"[{i+1}] {s.get('title', '')}: {s.get('content', '')[:400]}"
        for i, s in enumerate(all_sources[:max_sources])
    )

    prompt = _GROUNDING_PROMPT.format(brief=brief[:6000], sources=source_text)
    logger.debug(
        "check_source_grounding | sources_sent=%d | brief_chars=%d",
        min(len(all_sources), max_sources),
        len(brief[:6000]),
    )

    try:
        from groq import Groq
        import json

        client = Groq(api_key=groq_api_key)
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        usage = response.usage
        logger.info(
            "LLM call | fn=groq_grounding_check | model=llama-3.3-70b-versatile"
            " | prompt_tokens=%d | completion_tokens=%d | total_tokens=%d | elapsed_ms=%.0f",
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            elapsed_ms,
        )

        raw = response.choices[0].message.content.strip()

        # Extract the JSON array even if there's surrounding text
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            logger.debug("check_source_grounding | no JSON array in response")
            return []

        claims = json.loads(raw[start:end])
        ungrounded = [c for c in claims if isinstance(c, str) and c.strip()]
        logger.debug(
            "check_source_grounding | ungrounded_claims=%d", len(ungrounded)
        )
        return ungrounded
    except Exception as e:
        logger.error("check_source_grounding | failed | error=%s", e)
        return []


# ── Check 2: Completeness ─────────────────────────────────────────────────────

def check_completeness(brief: str) -> list[str]:
    """
    Verify that each of the 6 required section headings appears in the brief.
    Returns a list of missing section names.
    """
    brief_lower = brief.lower()
    missing = [s for s in REQUIRED_SECTIONS if s.lower() not in brief_lower]
    if missing:
        logger.debug("check_completeness | missing_sections=%s", missing)
    else:
        logger.debug("check_completeness | all sections present")
    return missing


# ── Check 3: Staleness ────────────────────────────────────────────────────────

def check_staleness(
    news_results: list[dict],
    funding_results: list[dict],
    techstack_results: list[dict],
    competitor_results: list[dict],
    people_results: list[dict],
    product_results: list[dict],
) -> list[str]:
    """
    Flag any search dimension that returned zero results.
    Returns a list of dimension names with no data.
    """
    dimension_map = {
        "News": news_results,
        "Funding": funding_results,
        "Tech Stack": techstack_results,
        "Competitors": competitor_results,
        "People": people_results,
        "Product": product_results,
    }
    empty = [name for name, results in dimension_map.items() if not results]
    if empty:
        logger.debug("check_staleness | empty_dimensions=%s", empty)
    else:
        logger.debug("check_staleness | all dimensions have data")
    return empty


# ── Composite Validator ───────────────────────────────────────────────────────

def validate_report(
    brief: str,
    all_sources: list[dict],
    news_results: list[dict],
    funding_results: list[dict],
    techstack_results: list[dict],
    competitor_results: list[dict],
    people_results: list[dict],
    product_results: list[dict],
    groq_api_key: str,
) -> ValidationResult:
    """
    Run all three checks and return a ValidationResult.

    Scoring:
    - Starts at 1.0
    - Each ungrounded claim  → -0.05 (capped at -0.30)
    - Each missing section   → -0.10
    - Each no-data dimension → -0.05
    """
    logger.debug("validate_report | starting all checks")

    ungrounded = check_source_grounding(brief, all_sources, groq_api_key)
    incomplete = check_completeness(brief)
    no_data = check_staleness(
        news_results, funding_results, techstack_results,
        competitor_results, people_results, product_results,
    )

    penalty = 0.0
    penalty += min(len(ungrounded) * 0.05, 0.30)
    penalty += len(incomplete) * 0.10
    penalty += len(no_data) * 0.05

    score = max(0.0, round(1.0 - penalty, 2))
    is_valid = score >= 0.6 and not incomplete

    logger.debug(
        "validate_report | score=%.2f | penalty=%.2f | is_valid=%s",
        score,
        penalty,
        is_valid,
    )
    return ValidationResult(
        is_valid=is_valid,
        ungrounded_claims=ungrounded,
        incomplete_sections=incomplete,
        no_data_sections=no_data,
        overall_score=score,
    )
