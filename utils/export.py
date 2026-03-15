import re


def parse_confidence_scores(brief: str) -> dict[str, dict]:
    """
    Scan the synthesized brief for section headings and their Confidence: X/5 lines.

    Returns a dict keyed by section heading text:
    {
        "Financial Health & Growth Stage": {
            "score": 4,
            "reason": "Two Crunchbase sources confirm the round...",
            "content": "Full section text including heading..."
        },
        ...
    }
    """
    # Split on ## headings (but not the title which uses #)
    section_pattern = re.compile(r"(^##\s+.+$)", re.MULTILINE)
    parts = section_pattern.split(brief)

    # parts alternates: [preamble, heading, body, heading, body, ...]
    sections: dict[str, dict] = {}

    i = 1  # skip preamble
    while i < len(parts) - 1:
        heading_raw = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        i += 2

        # Clean heading to plain text (strip ## and emoji)
        heading_text = re.sub(r"^##\s+", "", heading_raw).strip()

        # Find Confidence: X/5 — reason in this section's body
        conf_match = re.search(
            r"Confidence:\s*([1-5])/5\s*[—\-]\s*(.+)", body
        )

        if conf_match:
            score = int(conf_match.group(1))
            reason = conf_match.group(2).strip()
            # Remove the confidence line from the displayed content
            clean_body = re.sub(
                r"\n?Confidence:\s*[1-5]/5\s*[—\-]\s*.+\n?", "", body
            ).strip()
        else:
            score = None
            reason = None
            clean_body = body.strip()

        sections[heading_text] = {
            "score": score,
            "reason": reason,
            "content": f"{heading_raw}\n{clean_body}",
        }

    return sections


def confidence_badge(score: int | None, reason: str | None) -> tuple[str, str, str | None]:
    """
    Returns (color_hex, label, warning_text) for a given confidence score.
    warning_text is only set for scores 1-2.
    """
    if score is None:
        return "#9ca3af", "No score", None
    if score >= 4:
        return "#16a34a", f"{score}/5", None
    if score == 3:
        return "#d97706", f"{score}/5", None
    # 1-2
    return "#dc2626", f"{score}/5", "Low data coverage — verify manually"
