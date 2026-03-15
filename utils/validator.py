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
You are a fact-checking agent. Your job is to check if a claim \
is reasonably supported by the provided sources.

A claim is GROUNDED if:
- The information is directly stated in the sources, OR
- The information can be reasonably inferred from the sources, OR
- It is a general summary of what multiple sources say

A claim is UNGROUNDED only if:
- It directly contradicts the sources, OR
- It introduces specific facts (numbers, dates, names) that appear \
  nowhere in the sources

Claim: {sentence}
Sources: {tavily_results}

Reply with only: GROUNDED or UNGROUNDED\
"""

# Sections whose claims are sent to the grounding check
_GROUNDING_SECTIONS = frozenset({"Funding & Growth", "Key People", "Recent Signals"})

# Sentence-opening phrases that signal a summary or connector — not checkable facts
_TRANSITION_PHRASES = (
    "overall",
    "in summary",
    "additionally",
    "furthermore",
    "based on",
)

# Minimum word count for a sentence to be worth checking
_MIN_WORDS = 8


def _is_grounding_section(heading: str) -> bool:
    """Return True if heading text matches one of the grounding target sections."""
    heading_lower = heading.lower()
    return any(s.lower() in heading_lower for s in _GROUNDING_SECTIONS)


def _should_skip(sentence: str) -> bool:
    """Return True if a sentence should be excluded from grounding checks."""
    if len(sentence.split()) < _MIN_WORDS:
        return True
    lower = sentence.lower()
    return any(lower.startswith(p) for p in _TRANSITION_PHRASES)


def _extract_claims(brief: str, max_claims: int) -> list[str]:
    """
    Pull individual checkable claims from the grounding-target sections only.

    Sections checked:   Funding & Growth, Key People, Recent Signals
    Sections skipped:   Company Snapshot, Tech Stack, Competitor Context (and any other)

    Within target sections, each claim is also skipped when:
    - It has fewer than _MIN_WORDS words
    - It starts with a transition phrase (Overall, In summary, Additionally, …)
    """
    import re
    claims: list[str] = []
    in_target_section = False

    for line in brief.splitlines():
        line = line.strip()
        if not line:
            continue

        # Detect section heading and update scope
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            in_target_section = _is_grounding_section(heading)
            logger.debug(
                "_extract_claims | heading=%r | in_target_section=%s",
                heading,
                in_target_section,
            )
            continue

        if not in_target_section:
            continue

        # Split line into candidate sentences
        if line[:1] in ("-", "*", "•", "+"):
            candidates = [line.lstrip("-*•+ ").strip()]
        else:
            candidates = re.split(r"(?<=[.!?])\s+", line)

        for sentence in candidates:
            sentence = sentence.strip()
            if _should_skip(sentence):
                logger.debug("_extract_claims | skipped=%r", sentence[:60])
                continue
            claims.append(sentence)
            if len(claims) >= max_claims:
                return claims

    return claims


# ── Check 1: Source Grounding ─────────────────────────────────────────────────

def check_source_grounding(
    brief: str,
    all_sources: list[dict],
    groq_api_key: str,
    max_sources: int = 15,
    max_claims: int = 20,
) -> tuple[list[str], int]:
    """
    Check each extracted claim individually against the Tavily sources.

    Only claims from Funding & Growth, Key People, and Recent Signals sections
    are checked. Short sentences and transition phrases are skipped before any
    API call is made.

    Returns:
        (ungrounded_claims, checked_count)
        - ungrounded_claims: list of claim strings judged UNGROUNDED
        - checked_count: number of claims that were actually sent to Groq
          (used as the denominator for the grounding score)
    """
    if not groq_api_key or not brief or not all_sources:
        logger.debug("check_source_grounding | skipped (missing api_key, brief, or sources)")
        return [], 0

    claims = _extract_claims(brief, max_claims)
    if not claims:
        logger.debug("check_source_grounding | no checkable claims extracted from brief")
        return [], 0

    # Build the sources block once and reuse it for every claim
    tavily_results = "\n".join(
        f"[{i+1}] {s.get('title', '')}: {s.get('content', '')[:300]}"
        for i, s in enumerate(all_sources[:max_sources])
    )

    logger.debug(
        "check_source_grounding | claims_to_check=%d | sources_in_context=%d",
        len(claims),
        min(len(all_sources), max_sources),
    )

    from groq import Groq
    client = Groq(api_key=groq_api_key)

    ungrounded: list[str] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    t_start = time.perf_counter()

    for claim in claims:
        prompt = _GROUNDING_PROMPT.format(
            sentence=claim,
            tavily_results=tavily_results,
        )
        try:
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,  # only "GROUNDED" or "UNGROUNDED" needed
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            usage = response.usage
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens

            verdict = response.choices[0].message.content.strip().upper()
            logger.debug(
                "grounding_check | verdict=%s | elapsed_ms=%.0f | claim=%r",
                verdict,
                elapsed_ms,
                claim[:80],
            )

            if "UNGROUNDED" in verdict:
                ungrounded.append(claim)

        except Exception as e:
            logger.error(
                "grounding_check | failed | claim=%r | error=%s", claim[:80], e
            )

    total_elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "LLM call | fn=groq_grounding_check | model=llama-3.3-70b-versatile"
        " | claims_checked=%d | ungrounded=%d"
        " | prompt_tokens=%d | completion_tokens=%d | total_tokens=%d | elapsed_ms=%.0f",
        len(claims),
        len(ungrounded),
        total_prompt_tokens,
        total_completion_tokens,
        total_prompt_tokens + total_completion_tokens,
        total_elapsed_ms,
    )
    return ungrounded, len(claims)


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
    - Grounding base: grounded_count / checked_count
      (only sentences actually sent to Groq count; skipped sentences
       are excluded from both numerator and denominator)
    - Each missing section   → -0.10
    - Each no-data dimension → -0.05
    """
    logger.debug("validate_report | starting all checks")

    ungrounded, checked_count = check_source_grounding(brief, all_sources, groq_api_key)
    incomplete = check_completeness(brief)
    no_data = check_staleness(
        news_results, funding_results, techstack_results,
        competitor_results, people_results, product_results,
    )

    # Grounding score: ratio of checked claims that passed
    if checked_count > 0:
        grounding_score = (checked_count - len(ungrounded)) / checked_count
    else:
        grounding_score = 1.0  # nothing was checkable → no deduction

    # Structural penalties applied on top of the grounding score
    penalty = len(incomplete) * 0.10 + len(no_data) * 0.05
    score = max(0.0, round(grounding_score - penalty, 2))
    is_valid = score >= 0.6 and not incomplete

    logger.debug(
        "validate_report | grounding_score=%.2f | checked=%d | ungrounded=%d"
        " | penalty=%.2f | score=%.2f | is_valid=%s",
        grounding_score,
        checked_count,
        len(ungrounded),
        penalty,
        score,
        is_valid,
    )
    return ValidationResult(
        is_valid=is_valid,
        ungrounded_claims=ungrounded,
        incomplete_sections=incomplete,
        no_data_sections=no_data,
        overall_score=score,
    )
