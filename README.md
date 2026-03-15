# AgentCorp — B2B Sales Intelligence Agent

An agentic pipeline that autonomously gathers real-time intelligence on any company and synthesizes it into a structured sales brief. Built with LangGraph, Tavily, Groq, and Streamlit.

---

## Demo

Enter a company name → the agent runs 6 parallel searches across distinct intelligence dimensions → Groq/Llama-3.3 synthesizes everything into an actionable sales brief with confidence scores.

![Pipeline running with node status cards updating live]

---

## How It Works

```
User Input (company name)
         │
         ▼
    LangGraph Graph
         │
  ┌──────┴─────────────────────────────────────┐
  │              Fan-out (parallel)            │
  ▼       ▼        ▼          ▼       ▼    ▼  │
news  funding  techstack  competitor  people  product
node   node      node       node      node    node
  │       │        │          │       │    │  │
  └───────┴────────┴──────────┴───────┴────┘  │
                    │                         │
                    ▼                         │
            synthesize_node                  │
          (Groq / Llama-3.3-70b)            │
                    │
                    ▼
        change_detection_node
       (diff vs cached report)
                    │
                    ▼
           Sales Brief + Changes
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

Each node runs **simultaneously** via LangGraph's parallel fan-out. All results feed into `synthesize_node`, which waits for all 6 to complete before running.

### Change Detection

On every run, the synthesized brief is cached locally (`/cache/<company>.json`). On subsequent runs, `change_detection_node` compares the latest news against the cached version using Groq and surfaces what meaningfully changed — new funding, leadership changes, product launches, etc.

---

## Features

- **Parallel execution** — 6 search nodes run simultaneously, not sequentially
- **Domain filtering** — each node targets trusted, relevant sources per intelligence type
- **Per-section confidence scores** — every section of the brief is scored 1–5 based on data quality
- **Change detection** — diffs new results against cached report and highlights what changed
- **Source citations** — every result links back to its original source
- **Export** — download the full brief as a Markdown file
- **Live pipeline UI** — node status cards update in real-time as each search completes

---

## Tech Stack

| Layer               | Technology                                             |
| ------------------- | ------------------------------------------------------ |
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Web search          | [Tavily API](https://tavily.com)                       |
| LLM synthesis       | [Groq](https://groq.com) + Llama-3.3-70b-versatile     |
| UI                  | [Streamlit](https://streamlit.io)                      |
| Caching             | Local JSON (`/cache`)                                  |

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
```

### Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Project Structure

```
agentcorp/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .env.example
│
├── agent/
│   ├── graph.py            # LangGraph graph definition
│   ├── nodes.py            # All search nodes + synthesis + change detection
│   ├── prompts.py          # LLM prompt templates
│   └── state.py            # AgentState TypedDict
│
├── utils/
│   ├── cache.py            # Local JSON caching (save/load/exists)
│   └── export.py           # Confidence score parser + badge logic
│
└── cache/                  # Auto-created, gitignored
    └── <company>.json
```

---

## Example Output

The generated brief includes:

- **Executive Summary** — 2-3 sentence overview and core sales opportunity
- **Company Overview** — industry, business model, size/stage
- **Recent Activity & Sales Triggers** — latest news creating urgency
- **Financial Health & Growth Stage** — funding, investors, growth signals
- **Tech Stack & Infrastructure Insights** — current tools, gaps, hiring signals
- **Product & User Sentiment** — reviews, pricing, recurring complaints
- **Competitive Position** — key competitors, differentiation, weaknesses
- **Key Decision Makers** — names, titles, LinkedIn signals
- **Sales Opportunity Assessment** — why now, the hook, pain points, objections
- **Recommended Outreach Strategy** — 3 concrete next steps

Each section ends with a **Confidence score (1–5)** based on source quality and recency.

---

## API Costs

A typical run makes ~12 Tavily searches and 2 Groq calls (synthesis + change detection).

| Service              | Usage per run                 | Cost                   |
| -------------------- | ----------------------------- | ---------------------- |
| Tavily               | ~12 searches                  | Free tier: 1,000/month |
| Groq (Llama-3.3-70b) | ~6,000 tokens in / ~4,500 out | Free tier available    |

---

## Limitations

- Tavily free tier caps at 1,000 searches/month
- Domain filtering may reduce results for smaller/less-covered companies
- Change detection compares news only (not all 6 dimensions)
- Cache is local — not shared across machines or users
- No authentication or multi-user support

---
