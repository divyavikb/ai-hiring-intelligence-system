# Reflection

Draft in plain language. Edit it into your own voice before submitting.

## What I built

A system where five analysis agents read hiring data, a retrieval step pulls the matching hiring guidelines, a routing step picks a cheaper or stronger model per agent, an LLM writes one recommendation per agent, and an evaluation agent checks each recommendation before a manager sees it. A Streamlit dashboard shows the insights with their quality badge, the cost per agent, and the before/after cost comparison.

## What worked

- **Doing the maths in code, not in the model.** The agents compute every number with pandas. The LLM only explains and recommends. That is why most insights are grounded.
- **Per-agent knowledge base.** Giving each agent its own guidance file, and filtering retrieval by agent, stopped one agent from being fed another's guidance.
- **Cost.** Routing alone cut cost 7.2% against running everything on the large model. Compressing the panel-load input cut a further 16.5%, 22.5% in total, by sending a short per-interviewer summary instead of 32 weekly rows. That compression also fixed a real error: the model had read "10 interviewer-weeks over capacity" as 10 interviewers.
- **Evaluation caught real errors.** Rules caught an invented figure and a bottleneck claim when no bottleneck was listed. After I added generation rules to the prompt, the same runs went from 2 Green, 1 Amber, 2 Red to 5 Green (one run each, so a signal, not proof).

## What went wrong

- **I hit the API spend limit mid-project.** The run failed with a 429. That led to the retry and fallback design: transient errors retry with backoff, a spent quota is not retried, and a failed agent returns a flagged deterministic fallback instead of failing the whole run.

- **My first routing made quality worse.** I put panel-load and pipeline-health on the small model to save money. The insights got worse, so I moved both to the large model. Cost saving that lowers quality is not a saving.
- **Latency was noise.** The latency differences between runs were small enough that I did not claim an improvement from them.
- **The judge was too kind.** With real insights it first gave 1.0 to everything, including claims like "within expected ranges" that nothing supports. I made it adversarial and added deterministic rules for the mistakes it missed.
- **My own rule had a bug.** The people-count rule flagged a correct insight because it misread "interviewer weeks". The judge gave that insight full marks, so it was the rule that was wrong. Checking a flag against the text matters.
- **The generator is inconsistent.** The same prompt gave different mistakes on different runs. I would not quote any single run.

## Limits I would be upfront about

- The golden set is 20 scenarios I wrote. The rules were tuned after seeing the first misses, so 90% accuracy is a tuned result, not a held-out one. Two judge-only cases (a superlative on one offer, and generic advice) still slip through as Amber.
- All data is synthetic and small. Every conclusion is a lead for a human to check, not a finding.
- The analysis agents run one after another inside one graph node, and only the generation call is retried.
- Evaluation roughly doubles the per-run cost. I think it is worth it, but it is a real cost.

## What I would do next

1. Run the analysis agents as parallel graph branches and retry the indexing and judge calls too.
2. Move the offers agent to the large model if it stays the weakest in repeat runs.
3. Make `alternative` a structured trade-off, and break `cost_of_insight` into generation, retrieval and evaluation costs.
4. Build a held-out golden set written without looking at the rules.
5. Use real, anonymised data with proper consent before drawing any real conclusion.
