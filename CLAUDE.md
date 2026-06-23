# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**agent-corp** is an AI-powered company research platform that generates structured sales intelligence briefs. A user enters a company name, and the system orchestrates 6 parallel web searches (news, funding, tech stack, competitors, leadership, product sentiment) via Tavily, synthesizes results with Groq's Llama-3.3-70b, validates claims against sources, detects changes from cached reports, and outputs a downloadable Markdown brief.

- **Frontend:** Streamlit UI (`app.py`)
- **Backend:** FastAPI REST API (`api/`)
- **Agent:** LangGraph orchestrator running 10 nodes (`agent/`)
- **Core Stack:** LangGraph, Tavily Search API, Groq LLM, FastAPI, Streamlit

---

## Setup & Environment

### Dependencies
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### Environment Variables
Copy `.env.example` to `.env` and fill in API keys:
```env
TAVILY_API_KEY=tvly-...         # Required
GROQ_API_KEY=gsk_...             # Required
LANGCHAIN_TRACING_V2=true        # Optional (LangSmith observability)
LANGCHAIN_API_KEY=lsv2_...       # Optional
LANGCHAIN_PROJECT=agentcorp      # Optional
```

Get keys from:
- [Tavily](https://tavily.com) — free tier: 1,000 searches/month
- [Groq Console](https://console.groq.com) — free tier available
- [LangSmith](https://smith.langchain.com) — optional tracing

---

## Commands

### Running the Application

**Streamlit UI (local development)**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Pipeline runs directly in the Streamlit process.

**FastAPI Backend (standalone API)**
```bash
uvicorn api.main:app --reload --port 8000
```
Health check: `GET http://localhost:8000/health`

### Testing

**Run all integration tests**
```bash
pytest tests/test_nodes.py -v -s
```
Tests make real Tavily API calls using "Notion" as the test company. Output shows result count and first result title per node.

**Run single test**
```bash
pytest tests/test_nodes.py::test_news_node -v -s
```

### Linting & Code Quality
No linting tools are currently configured. Project follows PEP 8 conventions.

### Building for Production

**Streamlit Community Cloud Deployment**
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repo, set main file to `app.py`
4. Add API keys in Settings → Secrets as TOML

**Docker / Dev Container**
A `.devcontainer/devcontainer.json` is included for VS Code Remote Containers development.

---

## Architecture

### LangGraph Pipeline: 10-Node DAG

The core orchestration is a stateful graph (`agent/graph.py`) with TypedDict state (`agent/state.py`):

```
START
  ├─ load_cache_node       (snapshot old cache before overwrite)
  ├─ news_node             (TechCrunch, Reuters, Forbes, etc. — 7 days)
  ├─ funding_node          (Crunchbase, SEC filings, etc. — 90 days)
  ├─ techstack_node        (StackShare, GitHub, dev.to — 180 days)
  ├─ competitor_node       (G2, Capterra, Gartner, etc. — 30 days)
  ├─ people_node           (LinkedIn, Crunchbase, etc. — 30 days)
  └─ product_node          (Product Hunt, G2, Reddit, etc. — 30 days)
        └─ synthesize_node      (Groq: writes structured brief)
              └─ validation_node  (LLM grounding, completeness, staleness checks)
                    └─ change_detection_node (diffs new vs cached news)
                          └─ END
```

**Key Points:**
- 7 nodes run **in parallel** at START (load_cache + 6 searches)
- `synthesize_node` waits for all 7 to complete before writing the brief
- `validation_node` performs 3 checks with a composite score (grounded/checked − penalties)
- `change_detection_node` compares fresh news against cached report
- All state flows through `AgentState` TypedDict (see `agent/state.py`)

### Search Node Pattern

Each search node (e.g., `news_node`) in `agent/nodes.py`:
1. Builds domain-specific Tavily queries
2. Calls `_run_searches()` with domain filtering, time window, and deduplication
3. Returns `{"<dimension>_results": [SearchResult, ...]}`
4. Logging captures query, domains, result count, elapsed time

**SearchResult format:** `{title, url, content, score}`

### Synthesis & Validation

**Synthesis** (`synthesize_node`):
- Uses `SYNTHESIS_PROMPT` from `agent/prompts.py`
- Feeds all 6 search results to Groq Llama-3.3-70b
- LLM writes 9-section brief with per-section confidence scores (1–5)
- Brief format: Markdown with embedded `Confidence: X/5 — [reason]` after each section

**Validation** (`validation_node` → `utils/validator.py`):
1. **Grounding:** LLM verifies claims in "Funding & Growth", "Key People", "Recent Signals" against actual Tavily sources
2. **Completeness:** Checks all 6 required sections present
3. **Staleness:** Flags dimensions where Tavily returned no results
4. Returns `ValidationResult` with `is_valid`, `ungrounded_claims`, `incomplete_sections`, `no_data_sections`, `overall_score`

### State & AgentState

`agent/state.py` defines `AgentState` as a TypedDict:
```python
company_name: str
news_results: list[SearchResult]           # populated by news_node
funding_results: list[SearchResult]         # populated by funding_node
# ... (4 more result dimensions)
brief: str                                  # written by synthesize_node
all_sources: list[dict]                     # extracted by export utils
is_first_run: bool                          # true if no cached_report
cached_report: str                          # loaded by load_cache_node
changes_detected: list[str]                 # populated by change_detection_node
last_searched: str                          # timestamp
validation_result: dict                     # from validation_node
```

### API Layer (`api/`)

- **`config.py`:** Pydantic `Settings` reads from `.env` via `pydantic-settings`
- **`schemas.py`:** Request/response models (`ResearchRequest`, `ResearchResponse`, `ValidationResult`)
- **`main.py`:** FastAPI app with `/health` and `/research` endpoints

Endpoints run the graph in a thread pool executor to avoid blocking.

### Streamlit UI (`app.py`, ~547 lines)

- Injects `st.secrets` into `os.environ` for cloud deployment
- Builds input form, runs graph on submit
- Streams node status cards with live updates as each node completes
- Renders brief with source pills, confidence badges, and download button
- Handles first-run vs cached report diff display

### Utilities

- **`cache.py`:** JSON-based local cache (`cache/<company>.json`)
- **`export.py`:** Confidence score parsing, Markdown export, badge rendering
- **`validator.py`:** Grounding, completeness, staleness checks (335 lines)
- **`logger.py`:** Centralized rotating file logger (`logs/agentcorp.log`, 5 MB × 3 backups)
- **`tracing.py`:** LangSmith tracing setup (optional)
- **`validation_result.py`:** `ValidationResult` dataclass

---

## Key Design Patterns

### 1. **Parallel Execution at Scale**
LangGraph's `StateGraph` automatically parallelizes all 7 initial nodes. Groq/Tavily calls within `_run_searches()` are sequential but domain-scoped. No explicit async/threading needed for the DAG.

### 2. **Structured Output Parsing**
The `SYNTHESIS_PROMPT` instructs Groq to append `Confidence: X/5 — [reason]` after each section. `utils/export.parse_confidence_scores()` extracts these via regex. No tool_use or structured output mode required.

### 3. **Deduplication & Domain Filtering**
`_run_searches()` deduplicates by URL across multiple queries. Tavily's `include_domains` parameter narrows results per node (e.g., news_node targets journalism sites, techstack_node targets dev-focused platforms).

### 4. **Lazy LLM Calls**
Validation's grounding check is only called if brief scores below 0.70, to conserve Groq tokens.

### 5. **Local Caching + Change Detection**
After synthesis, brief is saved to `cache/<company>.json`. Next run's `change_detection_node` diffs old vs new news results to highlight what changed, without re-running all 6 nodes.

---

## Testing

### Test Structure
- Single test file: `tests/test_nodes.py`
- Integration-style tests: each test calls one search node with `company_name="Notion"`
- Tests verify result shape: `[SearchResult]` with `title, url, content, score`
- No mocking; uses real Tavily API (free tier)

### Running Tests
```bash
pytest tests/test_nodes.py -v -s
```
Example output:
```
[test_news_node] results returned: 5
[test_news_node] first result title: "Notion releases new AI features"
```

### Cost
~12 Tavily searches + ~22 Groq calls per run = free tier available.

---

## Observability

### Logging
All modules import from `utils/logger.py`:
```python
from utils.logger import get_logger
logger = get_logger(__name__)
```

Logs go to:
- **Console:** INFO and above
- **File:** `logs/agentcorp.log` (DEBUG and above, rotating)

Tavily calls log: query, domains, time window, result count, elapsed time.
Groq calls log: prompt tokens, completion tokens, elapsed time.
Validation scores below 0.70 trigger WARNING level.

### LangSmith Tracing (Optional)
Set `LANGCHAIN_TRACING_V2=true` in `.env` to trace every node, every Tavily call, and every LLM decision with latency and token counts. Dashboard at [smith.langchain.com](https://smith.langchain.com).

---

## Costs & Quotas

| Service              | Per Run | Tier                   |
|----------------------|---------|------------------------|
| Tavily searches      | ~12     | Free: 1,000/month      |
| Groq tokens (in/out) | ~8k/5k  | Free tier available    |

---

## Common Issues

### "TAVILY_API_KEY not set in environment"
Ensure `.env` file exists and contains `TAVILY_API_KEY=...`. On Streamlit Cloud, add it under Settings → Secrets.

### Tests fail with Tavily 429 (rate limit)
Wait 60 seconds. Free tier allows ~1 request/second.

### Brief validation score below 0.70
Check logs for ungrounded claims or empty dimensions. May indicate weak Tavily coverage for that company.

### Streamlit session state resets on rerun
By design. The graph doesn't use `st.cache_data` or session keys; each submission re-runs the full pipeline.

---

## File Structure Summary

```
agentcorp/
├── app.py                          # Streamlit UI entry point
├── requirements.txt                 # Dependencies
├── .env.example                     # Template for API keys
├── README.md                        # User-facing docs
├── CLAUDE.md                        # This file
│
├── agent/
│   ├── graph.py                    # LangGraph DAG (10 nodes, ~47 lines)
│   ├── nodes.py                    # All node implementations (~457 lines)
│   ├── prompts.py                  # Synthesis prompt template (~77 lines)
│   ├── state.py                    # AgentState TypedDict (~25 lines)
│   └── __init__.py
│
├── api/
│   ├── main.py                     # FastAPI endpoints (~133 lines)
│   ├── schemas.py                  # Pydantic models (~55 lines)
│   ├── config.py                   # Settings from .env (~28 lines)
│   └── __init__.py
│
├── utils/
│   ├── cache.py                    # JSON caching (~39 lines)
│   ├── export.py                   # Score parsing, Markdown export (~72 lines)
│   ├── validator.py                # Grounding, completeness, staleness (~335 lines)
│   ├── logger.py                   # Rotating file logger (~67 lines)
│   ├── tracing.py                  # LangSmith setup (~30 lines)
│   ├── validation_result.py        # ValidationResult dataclass (~29 lines)
│   └── __init__.py
│
├── tests/
│   ├── test_nodes.py               # Integration tests for 6 search nodes
│   └── __init__.py
│
├── .streamlit/                     # Streamlit config (empty by default)
├── .devcontainer/                  # Dev container for VS Code
├── .github/                        # (not present; no CI/CD yet)
├── .gitignore                      # Excludes .env, cache/, logs/, venv/, __pycache__/
│
├── cache/                          # Local JSON report cache (auto-created, gitignored)
├── logs/                           # Log files (auto-created, gitignored)
└── img/                            # LangSmith trace screenshot
```

---

## Next Steps for Development

- **Add CI/CD:** GitHub Actions to run `pytest` on PR and deploy to Streamlit Cloud on merge
- **Expand tests:** Add unit tests for `utils/validator.py` and `utils/export.py`
- **Multi-user support:** Add auth layer to FastAPI
- **Distributed cache:** Replace local JSON with Redis or DynamoDB
- **Change detection refinement:** Currently only diffs news; extend to all 6 dimensions
