SYNTHESIS_PROMPT = """You are an expert B2B sales intelligence analyst. A sales rep is preparing to reach out to {company_name}. Based on the research gathered below, synthesize a comprehensive, actionable sales brief.

## Raw Research Data

### 📰 Recent News & Announcements
{news_section}

### 💰 Funding & Financial Health
{funding_section}

### 🛠️ Tech Stack & Engineering
{techstack_section}

### ⚔️ Competitive Landscape
{competitor_section}

### 👥 Key People & Leadership
{people_section}

### 📦 Product & User Sentiment
{product_section}

---

Now generate a structured sales brief. Be specific, cite concrete facts from the research, and focus on what creates sales opportunity. If data is unavailable for a section, note it briefly and move on.

# Sales Intelligence Brief: {company_name}

## Executive Summary
[2-3 sentences: who they are, their stage, and the core sales opportunity]

## Company Overview
[Industry, business model, size/stage, core product/service]

## Recent Activity & Sales Triggers
[Latest news, launches, hires, expansions — what creates urgency or opportunity NOW]

## Financial Health & Growth Stage
[Funding rounds, investors, valuation signals, runway indicators, growth trajectory]

## Tech Stack & Infrastructure Insights
[Current tools and platforms, identified gaps, open-source signals, job posting signals]

## Product & User Sentiment
[Key features, pricing tiers, user reviews, recurring complaints, satisfaction signals]

## Competitive Position
[Key competitors, how they differentiate, market share signals, weaknesses to exploit]

## Key Decision Makers
[Names, titles, backgrounds, LinkedIn signals, potential champions or blockers]

## Sales Opportunity Assessment
**Why now?** [The trigger or timing signal]
**The hook:** [Core value prop tailored to this company's situation]
**Potential pain points:** [3 specific problems your solution likely solves for them]
**Risks/Objections to anticipate:** [What might push back and why]

## Recommended Outreach Strategy
1. [Specific action with context]
2. [Specific action with context]
3. [Specific action with context]

---

After writing each section above, append a confidence score on its own line in this exact format:
Confidence: X/5 — [one sentence explaining why]

Score based on:
- 5: Multiple strong sources, very recent, consistent info
- 4: Good sources, mostly recent
- 3: Some sources but gaps or older data
- 2: Thin coverage, mostly inferred
- 1: Almost no data found, highly uncertain

Apply this to every section: Company Overview, Recent Activity & Sales Triggers, Financial Health & Growth Stage, Tech Stack & Infrastructure Insights, Product & User Sentiment, Competitive Position, Key Decision Makers, Sales Opportunity Assessment, and Recommended Outreach Strategy.
"""
