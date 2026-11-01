# Contracts

What each stage takes in and gives back. Everything uses synthetic data.

## 1. Ingestion

Input: four CSVs in `data/raw/`: `candidates`, `offers`, `interview_feedback`, `interviewer_load`.
Output: `(datasets, reports)`. `datasets` is a dict of DataFrames. Each report has `name` and `passed`.
Rule: if any report fails, the run stops with `Data validation failed: [names]`. Optional date columns (`stage_entered_date`, `final_interview_date`) are validated when present.

## 2. Insight agents (deterministic, no LLM)

Each agent takes DataFrames and returns a dict with an `agent` name, its metrics, and `limitations` (plus `warnings` where relevant). Small samples are flagged in `warnings`.

| Agent | Main outputs |
|---|---|
| sourcing_quality | per-source candidates, rejection rate, offers, offer acceptance; stage distribution by source |
| rejection_patterns | reason counts, by role, `role_rejection_breakdown` (stage split, screen experience-mismatch share, `jd_problem_signal` over 30% with at least 5 rejections), `interviewer_rejection_rates` (at least 5 candidates), repeated feedback |
| panel_load | per interviewer-week assigned, capacity, utilization, status; `over_capacity_count` |
| offer_insights | totals, acceptance rate, decline reasons, salary summary, `offer_speed` (within 7 days, delayed over 14 days, declines in each) |
| pipeline_health | stage distribution, `stage_wait_summary` against SLA (applied 3, screen 7, interview 5 days; breach at 1.5x), `sla_breach_count`, `role_age`, `possible_bottlenecks` |

## 3. Retrieval

Input: a short per-agent query and a list of agent tags (the agent's own file plus `responsible_use`).
Output: up to 3 sections, each `{text, source, section}`. Each section carries an `agent` tag, so one agent's guidance never leaks into another's.

## 4. Routing

Input: agent key and the analysis.
Output: `{agent, model, complexity_score, est_input_tokens, reason}`, also appended to `logs/routing_log.jsonl`.
Rule: score below 3 uses `gpt-4o-mini`, 3 or more uses `gpt-4.1-mini`. Base complexity: offer_insights 2; the other four 3. A `force_model` override exists for baseline runs.

## 5. Insight (the required five-field output)

```json
{
  "recommendation": "string, one practical action",
  "evidence": ["string, each traceable to the analysis or retrieved guidance"],
  "confidence_score": 0.0,
  "cost_of_insight": 0.0,
  "alternative": "string, a reasonable other action"
}
```

- `confidence_score` is 0 to 1. The prompt caps it at 0.5 when a key figure rests on fewer than 10 records.
- `cost_of_insight` is calculated in Python from token usage and the per-model price table, not by the model. It is the USD cost of generating that insight and does not include the evaluation judge call.
- Wrapper fields added by the system: `agent_name`, `model`, `input_tokens`, `output_tokens`, `estimated_api_cost_usd`, `retrieved_context`, `routing`, `latency_seconds`.

## 6. Evaluation

Input: agent key, analysis, insight, retrieved context.
Output:
```json
{
  "agent": "panel_load",
  "badge": "Green | Amber | Red | Unrated",
  "score": 0.0,
  "accepted": true,
  "rules_failed": [],
  "rule_checks": [{"name": "schema", "passed": true}],
  "judge": {"actionability": 5, "grounding": 5, "hallucination_free": 5,
            "overconfident": false, "unsupported_claims": [], "summary": ""},
  "judge_error": null
}
```
Rules: any hard rule failure gives Red and caps the score at 0.5. A judge error gives Unrated and the insight is not accepted. Details are in `evaluation_notes.md`.

## 7. Run report

`reports/run_<label>.json` holds `analysis_report`, `llm_insights` and `usage_summary` (calls, tokens, cost, latency). `reports/eval_<label>.json` holds the evaluation and a summary (mean score, badge counts, judge cost).

## 8. Orchestration and failure behaviour

The pipeline is a LangGraph graph (`src/orchestration/insight_graph.py`):
`validate -> index knowledge -> analyze -> generate -> evaluate -> finalize`, with a conditional edge from `validate` that stops the run when data validation fails. Dependencies are injected, so the graph is tested with fakes.

| Failure | Behaviour |
|---|---|
| Data validation fails | graph ends with the named dataset in `errors`; nothing else runs |
| One analysis agent raises | recorded in `errors`; the other agents continue |
| Knowledge-base indexing or retrieval fails | that agent runs with empty context; the prompt forbids claiming guideline support |
| Transient API error (rate limit, timeout, connection, 5xx) | retried up to 3 times with exponential backoff and jitter |
| Spend limit, quota or invalid key | not retried (waiting cannot fix it) |
| Generation still fails | deterministic fallback insight: computed figures only, confidence 0, flagged `fallback: true`, with the cause; other agents unaffected |
| Judge call fails | that insight is Unrated and not accepted; the run continues |
| Fallback insight in evaluation | marked Unrated, "not evaluated" |
| Evaluation step fails as a whole | recorded in `errors`; insights are still returned |

The report carries `errors` (list), `fallback_agents` (list) and per-insight `attempts`.

## 9. Known gaps

- The five analysis agents run one after another inside one graph node, not as parallel graph branches.
- Retry covers the generation call only; indexing and the judge are not retried.
- `alternative` is free text. It is not a structured trade-off.
- A run with a fallback insight still writes its report; the dashboard should show the fallback warning (see README).
