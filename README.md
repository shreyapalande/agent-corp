# AgentCorp — B2B Sales Intelligence Agent

An agentic pipeline that autonomously gathers real-time intelligence on any company and synthesizes it into a structured sales brief. Built with LangGraph, Tavily, Groq, FastAPI, and Streamlit.

---

## Demo

Enter a company name → the agent runs 6 parallel searches across distinct intelligence dimensions → Groq/Llama-3.3 synthesizes everything into a validated, actionable sales brief with confidence scores.

---

## How It Works

```
User Input (company name)
         │
         ▼
    LangGraph Graph
         │
  ┌──────┴────────────────────────────────────────────┐
  │               Fan-out (parallel)                  │
  ▼      ▼       ▼          ▼        ▼      ▼         ▼
load  news  funding  techstack  competitor  people  product
cache  node   node      node       node      node    node
  │      │       │         │           │       │       │
  └──────┴───────┴─────────┴───────────┴───────┴───────┘
                           │
                           ▼
                   synthesize_node
                 (Groq / Llama-3.3-70b)
                           │
                           ▼
                   validation_node
                (grounding + completeness
                     + staleness)
                           │
                           ▼
               change_detection_node
                (diff vs cached report)
                           │
                           ▼
               Sales Brief + Validation + Changes
```

### The 6 Search Nodes

| Node              | What It Searches                           | Sources                                           | Time Window |
| ----------------- | ------------------------------------------ | ------------------------------------------------- | ----------- |
| `news_node`       | Announcements, press releases, launches    | TechCrunch, Reuters, Bloomberg, Forbes, Wired...  | 7 days      |
| `funding_node`    | Funding rounds, investors, valuations      | Crunchbase, PitchBook, Tracxn, SEC filings...     | 90 days     |
| `techstack_node`  | Tech stack, engineering blog, job postings | StackShare, GitHub, dev.to, BuiltWith...          | 180 days    |
| `competitor_node` | Competitors, market positioning            | G2, Capterra, Gartner, SimilarWeb...              | 30 days     |
| `people_node`     | Executives, leadership changes             | LinkedIn, Crunchbase, Forbes, Substack...         | 30 days     |
| `product_node`    | Reviews, user sentiment, pricing           | Product Hunt, G2, Capterra, Reddit, Trustpilot... | 30 days     |

Each node runs **simultaneously** via LangGraph's parallel fan-out. `load_cache_node` also runs in parallel to fetch any previously cached report. All 7 results feed into `synthesize_node`, which waits for all to complete before running.

### Validation

After synthesis, `validation_node` runs three checks:

1. **Source grounding** — each claim in the *Funding & Growth*, *Key People*, and *Recent Signals* sections is checked sentence-by-sentence against the Tavily sources using Groq/Llama-3.3. Short sentences (< 8 words) and transition phrases are skipped.
2. **Completeness** — verifies all 6 required sections are present in the brief.
3. **Staleness** — flags any search dimension that returned zero results.

The overall score is `grounded / checked` (only sentences actually sent to Groq count) minus structural penalties: −0.10 per missing section, −0.05 per empty dimension. Minimum score: 0.0.

### Change Detection

On every run, the synthesized brief is cached locally (`cache/<company>.json`). On subsequent runs, `change_detection_node` compares the latest news against the cached version using Groq and surfaces what meaningfully changed — new funding, leadership changes, product launches, etc.

---

## Features

- **Parallel execution** — 7 nodes run simultaneously (6 searches + cache load)
- **Domain filtering** — each node targets trusted, relevant sources per intelligence type
- **Per-section confidence scores** — every section scored 1–5 based on data quality
- **Validation** — grounding check, completeness check, and staleness check with a composite score
- **Change detection** — diffs new results against cached report and highlights what changed
- **Source citations** — every result links back to its original source
- **Export** — download the full brief as a Markdown file
- **Live pipeline UI** — node status cards update in real-time as each search completes
- **Structured logging** — all Tavily and LLM calls logged with timing and token counts to `logs/agentcorp.log`
- **FastAPI backend** — REST API decouples the pipeline from the UI

---

## Tech Stack

| Layer               | Technology                                             |
| ------------------- | ------------------------------------------------------ |
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Web search          | [Tavily API](https://tavily.com)                       |
| LLM synthesis       | [Groq](https://groq.com) + Llama-3.3-70b-versatile     |
| Backend API         | [FastAPI](https://fastapi.tiangolo.com) + Uvicorn       |
| UI                  | [Streamlit](https://streamlit.io)                      |
| Tracing             | [LangSmith](https://smith.langchain.com) (optional)    |
| Caching             | Local JSON (`cache/`)                                  |

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Tavily API key](https://tavily.com) — free tier includes 1,000 searches/month
- [Groq API key](https://console.groq.com) — free tier, very fast inference

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/agentcorp.git
cd agentcorp

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Environment Variables

```env
TAVILY_API_KEY=tvly-...
GROQ_API_KEY=gsk_...

# Optional — enables LangSmith tracing
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentcorp
```

### Run

**Option 1 — Streamlit + FastAPI (recommended)**

```bash
# Terminal 1: start the API server
uvicorn api.main:app --reload --port 8000

# Terminal 2: start the Streamlit UI
streamlit run app.py
```

UI opens at `http://localhost:8501`. The Streamlit app calls the FastAPI backend at `http://localhost:8000`.

**Option 2 — Streamlit only**

```bash
streamlit run app.py
```

The UI will call the API, so make sure the API server is also running.

---

## API Reference

### `GET /health`

Returns `{"status": "ok"}`.

### `POST /research`

Run the full pipeline for a company.

**Request body:**
```json
{ "company_name": "Notion" }
```

**Response:**
```json
{
  "company_name": "Notion",
  "brief": "## Company Snapshot\n...",
  "changes": "No previous report found.",
  "sources": [...],
  "validation": {
    "is_valid": true,
    "ungrounded_claims": [],
    "incomplete_sections": [],
    "no_data_sections": [],
    "overall_score": 0.95
  },
  "cached": false,
  "timestamp": "2024-01-15T10:30:00"
}
```

### `GET /research/{company_name}`

Retrieve the cached report for a company (returns 404 if not cached).

---

## Project Structure

```
agentcorp/
├── app.py                      # Streamlit UI
├── requirements.txt
├── .env.example
│
├── agent/
│   ├── graph.py                # LangGraph graph definition (10 nodes)
│   ├── nodes.py                # All nodes: search, synthesis, validation, change detection
│   ├── prompts.py              # LLM prompt templates
│   └── state.py                # AgentState TypedDict
│
├── api/
│   ├── config.py               # Pydantic settings (env vars)
│   ├── main.py                 # FastAPI app + endpoints
│   └── schemas.py              # Request/response Pydantic models
│
├── utils/
│   ├── cache.py                # Local JSON caching (save/load/exists)
│   ├── export.py               # Confidence score parser + Markdown export
│   ├── logger.py               # Centralized logging config (RotatingFileHandler)
│   ├── tracing.py              # LangSmith tracing helpers
│   ├── validation_result.py    # ValidationResult dataclass
│   └── validator.py            # Grounding, completeness, and staleness checks
│
├── tests/
│   └── test_nodes.py           # pytest integration tests for all 6 search nodes
│
├── cache/                      # Auto-created, gitignored
│   └── <company>.json
│
└── logs/                       # Auto-created, gitignored
    └── agentcorp.log
```

---

## Running Tests

Integration tests make real Tavily API calls against company "Notion". Requires `TAVILY_API_KEY` in `.env`.

```bash
pytest tests/test_nodes.py -v -s
```

Each test prints the number of results returned and the first result title.

---

## Observability

When `LANGCHAIN_TRACING_V2=true` is set, every pipeline run is traced in LangSmith — full node-by-node breakdown, token counts, and latency per step.

![LangSmith trace showing the AgentCorp pipeline](img/LangSmith.png)

---

## Logging

All activity is logged to `logs/agentcorp.log` (rotating, 5 MB × 3 backups) and to the console.

Every Tavily search logs:
- Query string, target domains, time window
- Number of hits, new (deduplicated) results, elapsed ms

Every Groq LLM call logs:
- Function name, model, prompt tokens, completion tokens, total tokens, elapsed ms

Validation logs a WARNING if the brief scores below 0.70.

---

## Example Output

The generated brief includes:

- **Company Snapshot** — industry, business model, size/stage
- **Recent Signals** — latest news creating urgency
- **Tech Stack** — current tools, gaps, hiring signals
- **Funding & Growth** — funding rounds, investors, growth signals
- **Key People** — executives, leadership changes, LinkedIn signals
- **Product Sentiment** — reviews, pricing, recurring complaints

Each section ends with a **Confidence score (1–5)** based on source quality and recency.

---

## API Costs

A typical run makes ~12 Tavily searches and ~22 Groq calls (1 synthesis + 1 change detection + up to 20 grounding checks).

| Service              | Usage per run                        | Cost                   |
| -------------------- | ------------------------------------ | ---------------------- |
| Tavily               | ~12 searches                         | Free tier: 1,000/month |
| Groq (Llama-3.3-70b) | ~8,000 tokens in / ~5,000 out        | Free tier available    |

---

## Limitations

- Tavily free tier caps at 1,000 searches/month
- Domain filtering may reduce results for smaller/less-covered companies
- Change detection compares news only (not all 6 dimensions)
- Cache is local — not shared across machines or users
- No authentication or multi-user support
