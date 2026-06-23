# Agent Corp

> Give it a company name. Get a structured sales intelligence brief in under a minute.

Agent Corp is an AI research agent that orchestrates **6 parallel web searches**, synthesizes the results with Gemini Flash 2.5, validates every claim against its sources, and delivers a downloadable brief — complete with confidence scores and change detection against previous runs.

---

<!-- GIF DEMO PLACEHOLDER -->
<!-- ![Agent Corp Demo](img/demo.gif) -->

---

## How it works

```
                     company name
                          │
              ┌───────────┼───────────────────────┐
              ▼       ▼       ▼    ▼    ▼    ▼    ▼
           cache    news  funding tech comp ppl  product
           load     node   node  node node node  node
              │       │       │    │    │    │    │
              └───────┴───────┴────┴────┴────┴───┘
                               │
                        synthesize_node
                       (Gemini Flash 2.5)
                               │
                        validation_node
                   (grounding · completeness · staleness)
                               │
                      change_detection_node
                        (diff vs cache)
                               │
                           brief ✓
```

All 7 nodes at the top run **in parallel** via LangGraph. Synthesis waits for all of them to complete before writing the brief.

---

## Features

- **Parallel search across 6 dimensions** — news, funding, tech stack, competitors, leadership, product sentiment — each targeting domain-specific sources (Crunchbase, G2, LinkedIn, StackShare, etc.)
- **LLM synthesis** — Gemini Flash 2.5 writes a 9-section brief grounded in the retrieved sources
- **Source grounding validation** — claims are checked sentence-by-sentence against actual Tavily results; a composite score (0–1) flags low-confidence briefs
- **Per-section confidence scores** — every section gets a 1–5 score explaining how well-sourced it is
- **Change detection** — diffs fresh results against the cached report so repeat runs surface only what's new
- **API key rotation** — cycles through multiple Gemini keys automatically on rate limits
- **Live UI updates** — Streamlit status cards update as each node finishes
- **Downloadable brief** — export as Markdown with one click
- **FastAPI backend** — pipeline is fully decoupled from the UI; usable as a standalone REST API
- **LangSmith tracing** — every node, LLM call, and token count is traced (optional)

---

## Tech stack

|                     |                      |
| ------------------- | -------------------- |
| Agent orchestration | LangGraph            |
| Web search          | Tavily API           |
| LLM                 | Gemini Flash 2.5     |
| Backend             | FastAPI              |
| UI                  | Streamlit            |
| Tracing             | LangSmith (optional) |

---

## Getting started

```bash
git clone https://github.com/shreyapalande/agent-corp.git
cd agent-corp

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env       # add your API keys
streamlit run app.py       # opens at http://localhost:8501
```

**Required keys** (both have free tiers):

- `TAVILY_API_KEY` — [tavily.com](https://tavily.com)
- `GEMINI_API_KEY` — [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

---

## Observability

LangSmith traces every node with latency and token counts. Set `LANGCHAIN_TRACING_V2=true` to enable.

![LangSmith trace](img/LangSmith.png)

---

## Project structure

```
agent-corp/
├── app.py              # Streamlit UI
├── api/                # FastAPI backend (main.py, schemas.py, config.py)
├── agent/
│   ├── graph.py        # 10-node LangGraph DAG
│   ├── nodes.py        # all node implementations
│   ├── prompts.py      # synthesis prompt
│   └── state.py        # AgentState TypedDict
├── utils/
│   ├── gemini_client.py  # Gemini calls with API key rotation
│   ├── validator.py      # grounding, completeness, staleness checks
│   ├── cache.py          # local JSON cache
│   └── export.py         # confidence score parsing + Markdown export
└── tests/
    ├── test_nodes.py     # search node integration tests
    └── test_pipeline.py  # full end-to-end pipeline test
```
