# Reflection

## What I built

Five analysis agents read the hiring data and compute every number with pandas. A retrieval step gives each agent its own hiring guidelines. A routing step picks a cheaper or stronger model per agent. An LLM writes one recommendation per agent, with evidence, a confidence score, a cost and an alternative. An evaluation agent checks each one before a manager sees it. A Streamlit dashboard shows the insights with a quality badge, the cost per agent, and the before and after cost comparison.

## What worked

- **Maths in code, not in the model.** The LLM only explains and recommends, so most insights are grounded in real figures.
- **A knowledge base per agent.** Filtering retrieval by agent stopped one agent being fed another's guidance.
- **Cost.** Routing cut cost 9.1%. Compressing the panel-load input took the total saving to 15 to 20% across two runs, with input tokens down 21.3%. The compression also removed a real error: the uncompressed run said "10 interviewers exceed capacity" when 10 was interviewer-weeks and only 4 interviewers were over.
- **Evaluation caught real errors,** such as an invented figure and a bottleneck claim when none was listed. After I tightened the generation prompt, the same runs went from 2 Green, 1 Amber, 2 Red to 5 Green. One run each, so a signal, not proof.

## What went wrong

- **My first routing made quality worse.** I put panel-load and pipeline-health on the small model to save money and the insights got worse, so I moved both back. A saving that lowers quality is not a saving.
- **The judge was too kind.** It first gave 1.0 to everything. I made it adversarial and added rules for the mistakes it missed.
- **It still misses comparisons.** In the final dashboard run, two insights scored Green 1.0 with a wrong comparison: "exceeds others" for the MLOps role, which at 68 days is the shortest, and three interviewers "higher than others" when two others are higher. So Green means no error detected, not verified correct.
- **My own rule had a bug.** The people-count rule flagged a correct insight because it misread "interviewer weeks". I now check every flag against the text.
- **I hit the API spend limit.** That led to the retry and fallback design: transient errors retry with backoff, a spent quota is not retried, and a failed agent returns a flagged fallback.
- **The generator varies.** The same prompt gave different mistakes on different runs, so I would not quote any single run. Latency differences were noise, so I claim no latency gain.

## Limits I would state

- The golden set is 20 scenarios I wrote. I tuned rules after seeing misses, so 90% is a tuned result, not a held-out one.
- All data is synthetic and small. Every conclusion is a lead for a human to check, not a finding.
- Evaluation roughly doubles the per-run cost. I think it is worth it, but it is a real cost.

## Next

1. Build a held-out golden set, and add a check for comparison claims.
2. Run the analysis agents as parallel branches, and retry the indexing and judge calls too.
3. Use real, anonymised data with consent before drawing real conclusions.
