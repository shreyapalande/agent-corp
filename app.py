import streamlit as st
import httpx
from dotenv import load_dotenv
from datetime import datetime
from utils.export import parse_confidence_scores, confidence_badge
from utils.tracing import is_tracing_enabled, get_project_url, configure_tracing
from api.config import settings

load_dotenv()
configure_tracing()

API_BASE = settings.api_base_url

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="B2B Sales Intelligence Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Main header */
        .hero-title {
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .hero-sub {
            color: #6b7280;
            font-size: 1.05rem;
            margin-top: 0.2rem;
        }

        /* Node status cards */
        .node-card {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 12px 16px;
            margin: 6px 0;
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.92rem;
        }
        .node-card.running {
            border-color: #6366f1;
            background: #eef2ff;
        }
        .node-card.done {
            border-color: #10b981;
            background: #ecfdf5;
        }

        /* Brief container */
        .brief-box {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 28px 32px;
            line-height: 1.7;
        }

        /* Source pill */
        .source-pill {
            display: inline-block;
            background: #f3f4f6;
            border: 1px solid #d1d5db;
            border-radius: 20px;
            padding: 3px 12px;
            font-size: 0.78rem;
            color: #374151;
            margin: 3px;
        }

        /* Section tabs */
        .stTabs [data-baseweb="tab"] {
            font-size: 0.88rem;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
NODE_META = {
    "news_node":             {"label": "News & Announcements",   "icon": "📰"},
    "funding_node":          {"label": "Funding & Financials",   "icon": "💰"},
    "techstack_node":        {"label": "Tech Stack",             "icon": "🛠️"},
    "competitor_node":       {"label": "Competitive Landscape",  "icon": "⚔️"},
    "people_node":           {"label": "Key People",             "icon": "👥"},
    "product_node":          {"label": "Product & Sentiment",    "icon": "📦"},
    "synthesize_node":       {"label": "Synthesizing Brief",     "icon": "🧠"},
    "validation_node":       {"label": "Validating Report",      "icon": "✅"},
    "change_detection_node": {"label": "Detecting Changes",      "icon": "🔄"},
}

DIMENSION_LABELS = {
    "news":        ("📰 News & Announcements",  "news_results"),
    "funding":     ("💰 Funding & Financials",  "funding_results"),
    "techstack":   ("🛠️ Tech Stack",            "techstack_results"),
    "competitors": ("⚔️ Competitors",           "competitor_results"),
    "people":      ("👥 Key People",            "people_results"),
    "product":     ("📦 Product & Sentiment",   "product_results"),
}


# ── Helper functions ──────────────────────────────────────────────────────────

def render_result_card(r: dict):
    st.write(f"[{r['title']}]({r['url']})")
    st.caption(r["url"])

    content = r["content"]
    limit = 400

    if len(content) > limit:
        # Trim to last complete sentence within limit
        trimmed = content[:limit]
        last_period = max(trimmed.rfind(". "), trimmed.rfind("! "), trimmed.rfind("? "))
        if last_period > 200:
            trimmed = trimmed[: last_period + 1]
        snippet = trimmed + " ..."
    else:
        snippet = content

    # Strip markdown image syntax and HTML img tags before displaying
    import re
    snippet = re.sub(r"!\[.*?\]\(.*?\)", "", snippet)
    snippet = re.sub(r"<img[^>]*>", "", snippet, flags=re.IGNORECASE)
    snippet = snippet.strip()

    st.markdown(
        f'<p style="font-size:0.95rem;line-height:1.6;color:#374151;margin:6px 0 12px;">{snippet}</p>',
        unsafe_allow_html=True,
    )
    st.divider()


def build_export_markdown(company: str, brief: str, sources: list[dict]) -> str:
    lines = [
        f"# Sales Intelligence Brief — {company}",
        f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} by B2B Sales Intelligence Agent*",
        "",
        brief,
        "",
        "---",
        "## Sources",
        "",
    ]
    for s in sources:
        lines.append(f"- [{s['title']}]({s['url']})  `{s['dimension']}`")
    return "\n".join(lines)


def check_cache(company: str) -> dict | None:
    """GET /research/{company} — returns cached report dict or None."""
    try:
        resp = httpx.get(f"{API_BASE}/research/{company}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except httpx.RequestError:
        return None


def run_pipeline(company_name: str) -> dict:
    """POST /research — runs the full pipeline via the FastAPI backend."""

    # ── Pipeline progress UI (all nodes shown as running, then done) ─────────
    st.markdown("### ⚙️ Pipeline Running")
    progress_cols = st.columns(3)

    col_map = {
        "news_node": 0, "funding_node": 0, "techstack_node": 0,
        "competitor_node": 1, "people_node": 1, "product_node": 1,
        "synthesize_node": 2, "validation_node": 2, "change_detection_node": 2,
    }
    node_placeholders: dict = {}
    for node_id, meta in NODE_META.items():
        col = progress_cols[col_map[node_id]]
        with col:
            node_placeholders[node_id] = st.empty()
            node_placeholders[node_id].markdown(
                f'<div class="node-card running">'
                f'<span style="font-size:1.2rem">{meta["icon"]}</span>'
                f'<span style="color:#4f46e5">⏳ {meta["label"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Call the API (blocking — runs in the background via FastAPI thread pool)
    try:
        resp = httpx.post(
            f"{API_BASE}/research",
            json={"company": company_name},
            timeout=120,
        )
        resp.raise_for_status()
    except httpx.RequestError as e:
        raise ConnectionError(
            f"Cannot reach API at {API_BASE}. Is the server running?\n"
            f"Start it with:  uvicorn api.main:app --reload\n\nDetail: {e}"
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"API error {e.response.status_code}: {e.response.text}")

    data = resp.json()

    # ── Mark all nodes done ───────────────────────────────────────────────────
    for node_id, meta in NODE_META.items():
        node_placeholders[node_id].markdown(
            f'<div class="node-card done">'
            f'<span style="font-size:1.2rem">{meta["icon"]}</span>'
            f'<span style="color:#059669">✅ {meta["label"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Map API response → final_state shape the rest of the UI expects ───────
    raw = data.get("raw_results") or {}
    return {
        "brief":             data.get("brief", ""),
        "all_sources":       [dict(s) for s in data.get("sources", [])],
        "changes_detected":  data.get("changes", []),
        "is_first_run":      data.get("is_first_run", False),
        "last_searched":     data.get("timestamp", ""),
        "validation":        data.get("validation", {}),
        "news_results":      raw.get("news", []),
        "funding_results":   raw.get("funding", []),
        "techstack_results": raw.get("techstack", []),
        "competitor_results":raw.get("competitors", []),
        "people_results":    raw.get("people", []),
        "product_results":   raw.get("product", []),
    }


# ── App Layout ────────────────────────────────────────────────────────────────

# Header
st.markdown('<p class="hero-title">🔍 B2B Sales Intelligence Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Enter a company name and get a full AI-powered sales brief in seconds — '
    "powered by Tavily, LangGraph & Groq/Llama-3.3</p>",
    unsafe_allow_html=True,
)
st.divider()

# Input
col_input, col_btn = st.columns([5, 1])
with col_input:
    company_name = st.text_input(
        "Company name",
        placeholder="e.g. Notion, Stripe, Linear, Vercel...",
        label_visibility="collapsed",
    )
with col_btn:
    analyze_clicked = st.button("Analyze →", type="primary", use_container_width=True)

# Example pills
st.markdown(
    "**Try:** "
    + " · ".join(
        [
            f'<span class="source-pill">{c}</span>'
            for c in ["Notion", "Stripe", "Linear", "Vercel", "Figma", "Retool"]
        ]
    ),
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Cache check on input ───────────────────────────────────────────────────────
if company_name.strip() and not analyze_clicked:
    cached_preview = check_cache(company_name.strip())
    if cached_preview:
        ts = cached_preview.get("timestamp", "")
        formatted = ts.replace("T", " ").split(".")[0] if "T" in ts else ts
        st.markdown(
            '<div style="background:#eff6ff;border:1px solid #93c5fd;border-radius:8px;'
            'padding:10px 16px;font-size:0.88rem;color:#1d4ed8;margin-bottom:8px;">'
            f"💾 Cached report found from <strong>{formatted}</strong> — "
            "click <strong>Analyze →</strong> to run a fresh search, or scroll down to view cached brief."
            "</div>",
            unsafe_allow_html=True,
        )
        with st.expander("📋 View cached brief", expanded=False):
            st.markdown(cached_preview.get("brief", ""))

# ── Run pipeline ──────────────────────────────────────────────────────────────
if analyze_clicked:
    if not company_name.strip():
        st.warning("Please enter a company name.")
        st.stop()

    company_name = company_name.strip()

    try:
        with st.spinner(f"Gathering intelligence on **{company_name}**…"):
            final_state = run_pipeline(company_name)

        st.success(f"✅ Intelligence brief for **{company_name}** is ready!")

        # ── Change detection banner ────────────────────────────────────────────
        is_first_run = final_state.get("is_first_run", False)
        changes = final_state.get("changes_detected", [])
        last_searched = final_state.get("last_searched", "")

        if is_first_run:
            st.markdown(
                '<div style="background:#f3f4f6;border:1px solid #d1d5db;border-radius:8px;'
                'padding:10px 16px;font-size:0.88rem;color:#6b7280;margin:8px 0;">'
                "🆕 First report generated — no previous data to compare"
                "</div>",
                unsafe_allow_html=True,
            )
        elif changes:
            change_bullets = "".join(f"<li>{c}</li>" for c in changes)
            st.markdown(
                '<div style="background:#fffbeb;border:1px solid #f59e0b;border-radius:8px;'
                'padding:12px 18px;margin:8px 0;">'
                '<p style="font-weight:700;color:#b45309;margin:0 0 8px;">🔄 What changed since last search:</p>'
                f"<ul style='margin:0;padding-left:20px;color:#374151;font-size:0.9rem;'>{change_bullets}</ul>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            formatted_ts = last_searched.replace("T", " ").split(".")[0] if "T" in last_searched else last_searched
            st.markdown(
                '<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;'
                f'padding:10px 16px;font-size:0.88rem;color:#15803d;margin:8px 0;">'
                f"✅ No significant changes since last searched on {formatted_ts}"
                "</div>",
                unsafe_allow_html=True,
            )

        # ── Validation banner ─────────────────────────────────────────────────
        vr = final_state.get("validation", {})
        v_score = vr.get("overall_score", 1.0)
        v_ungrounded = vr.get("ungrounded_claims", [])
        v_incomplete = vr.get("incomplete_sections", [])
        v_no_data = vr.get("no_data_sections", [])

        if v_score >= 0.8:
            st.markdown(
                '<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;'
                'padding:10px 18px;font-size:0.9rem;color:#166534;margin:8px 0;">'
                f"🟢 <strong>High confidence report</strong> — all claims grounded in sources "
                f"<span style='float:right;font-weight:600;'>Score: {v_score:.0%}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        elif v_score >= 0.6:
            st.markdown(
                '<div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:8px;'
                'padding:10px 18px;font-size:0.9rem;color:#92400e;margin:8px 0;">'
                f"🟡 <strong>Moderate confidence</strong> — some claims could not be verified "
                f"<span style='float:right;font-weight:600;'>Score: {v_score:.0%}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;'
                'padding:10px 18px;font-size:0.9rem;color:#991b1b;margin:8px 0;">'
                f"🔴 <strong>Low confidence</strong> — verify key claims manually before use "
                f"<span style='float:right;font-weight:600;'>Score: {v_score:.0%}</span>"
                "</div>",
                unsafe_allow_html=True,
            )

        # Detail expander: show ungrounded claims, incomplete/no-data sections
        detail_items = len(v_ungrounded) + len(v_incomplete) + len(v_no_data)
        if detail_items > 0:
            with st.expander(f"🔍 Validation details ({detail_items} issue{'s' if detail_items != 1 else ''})", expanded=False):
                if v_ungrounded:
                    st.markdown("**⚠️ Claims not found in source snippets:**")
                    for claim in v_ungrounded:
                        st.markdown(
                            f'<div style="background:#fef9c3;border-left:3px solid #f59e0b;'
                            f'padding:6px 12px;margin:4px 0;border-radius:4px;font-size:0.88rem;">'
                            f"⚠️ {claim}</div>",
                            unsafe_allow_html=True,
                        )
                if v_incomplete:
                    st.markdown("**❌ Missing sections:**")
                    for section in v_incomplete:
                        st.markdown(f"- {section}")
                if v_no_data:
                    st.markdown("**📭 Dimensions with no search results:**")
                    for dim in v_no_data:
                        st.markdown(f"- {dim}")

        st.markdown("---")

        # ── Brief display ─────────────────────────────────────────────────────
        brief = final_state.get("brief", "")
        all_sources = final_state.get("all_sources", [])

        col_brief, col_meta = st.columns([3, 1])

        with col_brief:
            st.markdown("## 📄 Sales Brief")
            sections = parse_confidence_scores(brief)

            if sections:
                for heading, data in sections.items():
                    score = data["score"]
                    reason = data["reason"]
                    content = data["content"]

                    color, label, warning = confidence_badge(score, reason)

                    # Render section content as markdown
                    st.markdown(content)

                    # Confidence badge line
                    badge_html = (
                        f'<span style="display:inline-block;background:{color}22;'
                        f'border:1px solid {color};border-radius:6px;'
                        f'padding:2px 10px;font-size:0.8rem;color:{color};'
                        f'font-weight:600;">Confidence: {label}</span>'
                    )
                    if reason:
                        badge_html += (
                            f'<span style="font-size:0.82rem;color:#6b7280;'
                            f'margin-left:8px;">— {reason}</span>'
                        )
                    st.markdown(badge_html, unsafe_allow_html=True)

                    if warning:
                        st.warning(f"⚠️ {warning}", icon=None)

                    st.markdown("")
            else:
                # Fallback: render brief as-is if parsing finds no sections
                st.markdown(brief)

        with col_meta:
            st.markdown("### 📊 Run Stats")
            source_count = len(all_sources)
            result_counts = {
                "📰 News":        len(final_state.get("news_results", [])),
                "💰 Funding":     len(final_state.get("funding_results", [])),
                "🛠️ Tech Stack":  len(final_state.get("techstack_results", [])),
                "⚔️ Competitors": len(final_state.get("competitor_results", [])),
                "👥 People":      len(final_state.get("people_results", [])),
                "📦 Product":     len(final_state.get("product_results", [])),
            }
            for label, count in result_counts.items():
                st.metric(label, f"{count} results")
            st.metric("🔗 Total Sources", source_count)

            # Export
            st.markdown("### ⬇️ Export")
            export_md = build_export_markdown(company_name, brief, all_sources)
            st.download_button(
                label="Download as Markdown",
                data=export_md,
                file_name=f"sales_brief_{company_name.lower().replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

            # Tracing
            st.markdown("### 🔬 Tracing")
            if is_tracing_enabled():
                st.success("LangSmith active", icon="✅")
                st.markdown(
                    f"[View traces →]({get_project_url()})",
                    unsafe_allow_html=False,
                )
                st.caption(
                    "Every node, Tavily query, and LLM call is logged "
                    "with latency and token usage."
                )
            else:
                st.warning("Tracing off", icon="⚠️")
                st.caption(
                    "Add `LANGCHAIN_TRACING_V2=true` and "
                    "`LANGCHAIN_API_KEY` to `.env` to enable LangSmith tracing."
                )

        # ── Raw data tabs ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🗂️ Raw Intelligence Data")

        tabs = st.tabs([label for label, _ in DIMENSION_LABELS.values()])

        for i, (dim_key, (tab_label, state_key)) in enumerate(DIMENSION_LABELS.items()):
            with tabs[i]:
                results = final_state.get(state_key, [])
                if results:
                    for r in results:
                        render_result_card(r)
                else:
                    st.info("No results found for this dimension.")

        # ── Sources list ───────────────────────────────────────────────────────
        with st.expander(f"🔗 All Sources ({len(all_sources)})", expanded=False):
            dim_colors = {
                "news": "#dbeafe",
                "funding": "#dcfce7",
                "techstack": "#fef3c7",
                "competitors": "#fce7f3",
                "people": "#f3e8ff",
                "product": "#ffedd5",
            }
            for s in all_sources:
                color = dim_colors.get(s["dimension"], "#f3f4f6")
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;'
                    f"padding:6px 0;border-bottom:1px solid #f3f4f6;\">"
                    f'<span style="background:{color};padding:2px 8px;border-radius:12px;'
                    f'font-size:0.75rem;">{s["dimension"]}</span>'
                    f'<a href="{s["url"]}" target="_blank" style="color:#4f46e5;font-size:0.85rem;">'
                    f"{s['title']}</a>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    except ConnectionError as e:
        st.error("**Cannot reach the API server.**")
        st.code(str(e))
        st.info("Start the backend with:  `uvicorn api.main:app --reload`")
    except ValueError as e:
        st.error(f"**Configuration error:** {e}")
        st.info("Make sure your `.env` file has valid `TAVILY_API_KEY` and `GROQ_API_KEY` values.")
    except Exception as e:
        st.error(f"**Pipeline error:** {e}")
        st.exception(e)

else:
    # Landing state placeholder
    st.markdown(
        """
        <div style="text-align:center;padding:60px 20px;color:#9ca3af;">
            <div style="font-size:4rem;">🔍</div>
            <p style="font-size:1.1rem;margin-top:12px;">
                Enter a company name above and click <strong>Analyze</strong> to generate<br>
                a full B2B sales intelligence brief in ~30 seconds.
            </p>
            <div style="margin-top:28px;display:flex;justify-content:center;gap:20px;flex-wrap:wrap;">
                <div style="background:#f3f4f6;border-radius:10px;padding:14px 20px;font-size:0.88rem;">
                    📰 <strong>News</strong><br><span style="color:#9ca3af;">Announcements & triggers</span>
                </div>
                <div style="background:#f3f4f6;border-radius:10px;padding:14px 20px;font-size:0.88rem;">
                    💰 <strong>Funding</strong><br><span style="color:#9ca3af;">Financial health signals</span>
                </div>
                <div style="background:#f3f4f6;border-radius:10px;padding:14px 20px;font-size:0.88rem;">
                    🛠️ <strong>Tech Stack</strong><br><span style="color:#9ca3af;">Tools & infrastructure</span>
                </div>
                <div style="background:#f3f4f6;border-radius:10px;padding:14px 20px;font-size:0.88rem;">
                    ⚔️ <strong>Competitors</strong><br><span style="color:#9ca3af;">Market positioning</span>
                </div>
                <div style="background:#f3f4f6;border-radius:10px;padding:14px 20px;font-size:0.88rem;">
                    👥 <strong>People</strong><br><span style="color:#9ca3af;">Decision makers</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
