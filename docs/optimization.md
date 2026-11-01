# Cost and latency optimization

Numbers come from saved runs in `reports/` (`run_baseline.json`, `run_routed.json`, `run_routed_compressed.json`). Cost is estimated from token usage and the per-model price table in `src/insight_generator.py`; verify current prices before relying on it. Same data and same five agents in every run.

## Before and after

| Run | Input tokens | Output tokens | Cost (USD) | vs baseline |
|---|---|---|---|---|
| Baseline: every agent on gpt-4.1-mini | 9,239 | 1,450 | 0.0060156 | n/a |
| Optimization 1: model routing | 9,239 | 1,391 | 0.0054662 | -9.1% |
| Optimization 2: routing + panel-load input compression | 7,270 | 1,427 | 0.0047902 | -20.4% |

Optimization 2 alone, against routing: -12.4% cost and -21.3% input tokens. All three runs use the same final code and prompt, so the comparison is like for like. Output tokens vary a little from run to run, which is why input tokens are the cleaner measure of compression.

A later live run from the dashboard (same code) gave 7,270 input, 1,642 output and 0.005096 USD, a 15.3% saving against the same baseline. The saving therefore sits between about 15% and 20% depending on how long the model's answers are. The input-token saving (-21.3%) does not move, because the input is fixed.

## Per agent

Routing only (`run_routed`):

| Agent | Model | Input tokens | Output tokens | Cost (USD) | Latency (s) |
|---|---|---|---|---|---|
| sourcing_quality | gpt-4.1-mini | 1,505 | 283 | 0.001055 | 4.94 |
| rejection_patterns | gpt-4.1-mini | 1,958 | 247 | 0.001178 | 3.71 |
| panel_load | gpt-4.1-mini | 3,359 | 338 | 0.001884 | 4.61 |
| offer_insights | gpt-4o-mini | 988 | 208 | 0.000273 | 3.99 |
| pipeline_health | gpt-4.1-mini | 1,429 | 315 | 0.001076 | 4.00 |

Routing + compression (`run_routed_compressed`):

| Agent | Model | Input tokens | Output tokens | Cost (USD) | Latency (s) |
|---|---|---|---|---|---|
| sourcing_quality | gpt-4.1-mini | 1,505 | 378 | 0.001207 | 5.18 |
| rejection_patterns | gpt-4.1-mini | 1,958 | 195 | 0.001095 | 3.28 |
| panel_load | gpt-4.1-mini | 1,390 | 308 | 0.001049 | 5.12 |
| offer_insights | gpt-4o-mini | 988 | 154 | 0.000241 | 3.69 |
| pipeline_health | gpt-4.1-mini | 1,429 | 392 | 0.001199 | 4.40 |

The rejection agent now sends the per-role breakdown (reasons, stages, interviewer rates), so its input is 1,958 tokens, up from about 1,100 earlier. That is the price of the correct role attribution and it is why the totals are higher than in my earlier runs.

## Optimization 1: model routing

Each agent gets a complexity score (a base per agent, adjusted by input size). Below 3 goes to `gpt-4o-mini`, 3 or more to `gpt-4.1-mini`. Every decision is logged to `logs/routing_log.jsonl` with the score and reason.

Result: the saving is modest (9.1%) because only the offers agent runs on the small model. Priced at the same tokens (988 in, 208 out), the offers call would cost 0.000728 on gpt-4.1-mini against 0.000273 on gpt-4o-mini, a 62.5% saving on that call. My first routing also sent panel-load and pipeline-health to the small model and the insights got worse, so I raised both to the large model. A saving that lowers quality is not a saving, and I kept the quality-safe version.

## Optimization 2: input compression for panel load

The panel-load agent produces one row per interviewer per week (32 rows). Sending all of them cost 3,359 input tokens. `summarize_panel_load` sends one row per interviewer plus totals instead.

- Panel-load input: 3,359 to 1,390 tokens (-58.6%)
- Panel-load cost: 0.001884 to 0.001049 (-44.3%)
- Quality: in the uncompressed run the model wrote "10 interviewers exceed capacity" (10 is interviewer-weeks; only 4 interviewers are over). In the compressed run it said "10 weeks over capacity across four interviewers" and "18 excess interviews", and named I003 and I001 with 8 excess interviews each. I checked these against the data: I001 is 2+3+2+1 = 8, I003 is 2+2+1+3 = 8, and the total with I002 and I007 (1 each) is 18. One run each, so indicative, but compression improved the answer rather than hurting it.

## Latency

Total generation latency was 20.25 s, 21.25 s and 21.67 s across baseline, routed and compressed. The differences are within run-to-run noise (earlier sets of runs on the same code varied by several seconds), so I do not claim a latency improvement. Latency is measured and shown per agent for visibility only. The agents run one after another, so total latency is the sum of the five calls.

## What evaluation costs

The LLM judge adds about 0.0050 USD per evaluated run (0.0050324 measured), roughly the same as generating the insights. The rule checks are free. Evaluation is the largest remaining cost item and a deliberate trade for catching errors; `--no-llm` runs the rules alone for free.

## What the accuracy rules cost

Adding the generation rules to the prompt added about 190 input tokens per call (940 per run), roughly 11% more per run, in exchange for fewer wrong insights.

## Not done

- RAG caching: retrieval is local and fast and was not a cost or latency driver, so I did not optimize it.
- Single runs per configuration. Generation varies run to run, so treat the percentages as indicative.
