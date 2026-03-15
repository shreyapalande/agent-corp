import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

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
    "news_node":       {"label": "News & Announcements",   "icon": "📰"},
    "funding_node":    {"label": "Funding & Financials",   "icon": "💰"},
    "techstack_node":  {"label": "Tech Stack",             "icon": "🛠️"},
    "competitor_node": {"label": "Competitive Landscape",  "icon": "⚔️"},
    "people_node":     {"label": "Key People",             "icon": "👥"},
    "product_node":    {"label": "Product & Sentiment",    "icon": "📦"},
    "synthesize_node": {"label": "Synthesizing Brief",     "icon": "🧠"},
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


def run_pipeline(company_name: str):
    """Stream the LangGraph pipeline and update the UI in real-time."""
    from agent.graph import build_graph

    graph = build_graph()

    initial_state = {
        "company_name": company_name,
        "news_results": [],
        "funding_results": [],
        "techstack_results": [],
        "competitor_results": [],
        "people_results": [],
        "product_results": [],
        "brief": "",
        "all_sources": [],
    }

    # ── Pipeline progress UI ─────────────────────────────────────────────────
    st.markdown("### ⚙️ Pipeline Running")
    progress_cols = st.columns(3)

    node_placeholders: dict = {}
    col_map = {
        "news_node": 0, "funding_node": 0,
        "techstack_node": 1, "competitor_node": 1,
        "people_node": 2, "product_node": 2, "synthesize_node": 2,
    }
    for node_id, meta in NODE_META.items():
        col = progress_cols[col_map[node_id]]
        with col:
            node_placeholders[node_id] = st.empty()
            node_placeholders[node_id].markdown(
                f'<div class="node-card">'
                f'<span style="font-size:1.2rem">{meta["icon"]}</span>'
                f'<span style="color:#9ca3af">⏳ {meta["label"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    completed_nodes: set[str] = set()
    final_state: dict = {}

    # ── Stream events ────────────────────────────────────────────────────────
    for event in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "__end__":
                continue

            # Mark node as done in the UI
            if node_name in NODE_META:
                completed_nodes.add(node_name)
                meta = NODE_META[node_name]
                node_placeholders[node_name].markdown(
                    f'<div class="node-card done">'
                    f'<span style="font-size:1.2rem">{meta["icon"]}</span>'
                    f'<span style="color:#059669">✅ {meta["label"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Accumulate state
            final_state.update(node_output)

    return final_state


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
        st.markdown("---")

        # ── Brief display ─────────────────────────────────────────────────────
        brief = final_state.get("brief", "")
        all_sources = final_state.get("all_sources", [])

        col_brief, col_meta = st.columns([3, 1])

        with col_brief:
            st.markdown("## 📄 Sales Brief")
            st.markdown(
                f'<div class="brief-box">{brief}</div>',
                unsafe_allow_html=True,
            )

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
