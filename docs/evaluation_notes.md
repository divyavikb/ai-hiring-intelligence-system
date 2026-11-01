# Evaluation notes

Running log for the evaluation agent. Newest sections go at the bottom.

## 1. What the evaluation agent does

Every generated insight passes two layers before it reaches the dashboard.

**Layer 1: deterministic rules (free, repeatable)**

| Rule | Fails when | Result |
|---|---|---|
| schema | recommendation, evidence, confidence (0-1), alternative or cost missing | hard fail |
| responsible_use | automated-decision phrases, or bias/removal claims about a person | hard fail |
| numeric_grounding | 2 or more numbers appear in neither the analysis nor the retrieved guidelines (percent-to-fraction and whole-number ratios accepted) | hard fail |
| confidence_vs_sample | confidence above 0.6 while the analysis warns of small samples | hard fail |
| unsupported_bottleneck | "bottleneck" claimed while `possible_bottlenecks` is empty (negations allowed) | hard fail |
| people_count | a count of interviewers differs from people (over capacity / total), e.g. interviewer-weeks mistaken for people | hard fail |

**Layer 2: LLM judge** (gpt-4.1-mini, temperature 0). Scores actionability, grounding and hallucination-free from 1 to 5, flags overconfidence, and quotes each unsupported claim.

**Score and badge**
- score = (actionability + grounding + hallucination_free) / 15, minus 0.10 if overconfident
- Red: any hard rule failure, or hallucination_free <= 2, or score < 0.6
- Green: score >= 0.8, hallucination_free >= 4, not overconfident
- Amber: everything else. Red and Amber are shown flagged, not hidden.
- If the judge call fails the insight is Unrated and is never accepted.

## 2. Golden set

20 scenarios in `data/golden/golden_scenarios.json`, built by `scripts/build_golden_set.py` from real agent analyses: 8 good insights (should accept) and 12 defective ones (should reject). Each defective one is labelled `detectable_by`: "rule" (4 at first) or "llm" (8).

Defects covered: invented figures, automated-decision language, empty evidence, overconfidence on a small sample, false bottleneck, interviewer-weeks read as people, a superlative resting on 1 offer, "5.4 exceeds 7", generic advice, a figure below a threshold called above it, "all declines" when 3 of 4, and a bias claim from 14 candidates.

## 3. Results

| Run | Accuracy | Bad caught | Good wrongly rejected | Notes |
|---|---|---|---|---|
| Rules only (v1) | 0.55 | 4/12 | 0/8 | rules catch 4/4 rule-detectable defects |
| Rules + judge (v1) | 0.80 | 8/12 | 0/8 | missed B04, B05, B06, B12 |
| Rules + judge (v2) | 0.90 | 10/12 | 0/8 | missed B07, B09 (judge-only) |

## 4. Findings

1. **The judge alone was not enough.** In v1 it let through a false bottleneck at 0.85 confidence, a "10 interviewers" claim (actually 4 people, 10 interviewer-weeks), and a bias claim from 14 candidates. Each is now a deterministic rule.
2. **Rules fire the same way every time; the judge does not.** B07 (a 100% acceptance rate resting on 1 offer) and B09 (generic advice) were Red in the first judge run and Amber in the second, at temperature 0. Quote a range over several runs, not one score.
3. **Cheap checks first.** Rules cost nothing, so they run before the judge and set the hard fails.
4. **No false rejections** of good insights in any run.
5. **A real catch.** On the saved panel-load insight the numeric rule flagged "130%-160%" and "1.3", which are not in the analysis.

## 5. Limitations (state these in the demo)

- v2 rules were added after seeing the v1 misses, so 0.90 is a tuned result on a set the rules have seen, not a held-out one. A fair held-out number needs new bad scenarios written without touching the rules.
- The numeric rule checks that numbers exist, not that they are used correctly. Misuse (B08, B10) is the judge's job.
- The judge model is the same family as the generator, so it can share blind spots.
- The golden set is small (20) and synthetic. Treat accuracy as a smoke test, not a benchmark.
- Remaining judge-only gaps: superlatives on tiny samples, generic advice.

## 6. Log

- 2026-10-25: v1 golden run, 0.80. Added bottleneck, people_count, punitive-phrase rules and made overconfidence a hard fail. v2 golden run, 0.90.
- 2026-10-25: Fixed a people_count false positive ("10 instances of interviewer weeks" read as 10 people). Made the judge adversarial: benchmark claims such as "within expected ranges" need a source. A hard rule failure now caps the score at 0.5.

## 7. Prompt rules: before and after

Change: added confidence and claim rules to the insight generator prompt (confidence at most 0.5 when a key figure rests on under 10 records; "bottleneck" only if `possible_bottlenecks` is non-empty; no superlatives on tiny samples; correct comparison to targets; people vs records; no claims about bias; recommendations must name who does what).

| Agent | Before (old prompt) | After (new prompt) |
|---|---|---|
| sourcing_quality | Green 1.0 | Green 1.0 |
| rejection_patterns | Green 1.0 | Green 1.0 |
| panel_load | Red (numeric_grounding) | Green 1.0 |
| offer_insights | Amber 0.767 | Green 0.867 |
| pipeline_health | Red (unsupported_bottleneck) | Green 1.0 |
| Summary | 2 Green, 1 Amber, 2 Red | 5 Green, mean 0.973 |

Reading it honestly:
- One run each. The generator varies run to run (an earlier run of the new prompt had panel_load flagged by a rule that turned out to be a bug in the rule, since fixed), so repeat before quoting this as a rate.
- The two Reds before were genuine generator errors caught by deterministic rules: an invented figure and a bottleneck claimed when none was listed.
- Offers is the weakest agent in both runs (small model). It improved but scored lowest each time. Candidate for the large model.
- Judge cost is about 0.005 USD per evaluated run, roughly the cost of generating the insights, so evaluation doubles the per-run cost. Worth stating in the cost section.
