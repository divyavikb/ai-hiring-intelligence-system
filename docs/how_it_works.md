# How my system works

My system helps a hiring manager make sense of four spreadsheets: candidates, offers, interview feedback and interviewer workload. It has two paths, and the difference between them is whether AI is involved.

## The five agents and the CSVs they read

I have five analysis agents. Each one answers a single question about the hiring process, and each one is plain maths with no AI. Between them they use all four CSVs, and some share a file because they ask different questions of it.

**1. Sourcing quality agent**
- **CSV:** `candidates.csv` and `offers.csv`
- **Question it asks:** which hiring source (Referral, Job Board, Campus, Recruiter) brings good candidates?
- **What it calculates:** for each source, the number of candidates, the rejection rate, the offers made and the offer acceptance rate. It warns when a source has too few candidates to trust.
- **Example finding:** Job Board rejected 23 of its 32 candidates (72%), against about 31% for Referral and Recruiter.

**2. Rejection patterns agent**
- **CSV:** `candidates.csv` and `interview_feedback.csv`
- **Question it asks:** why are people rejected, and at which stage?
- **What it calculates:** the rejection reasons, rejections by role and by interview stage, repeated negative feedback, and rejection rates per interviewer. It flags a job description problem when more than 30% of a role's rejections are "experience mismatch" at screening.
- **Example finding:** 8 of Data Analyst's 11 rejections (72.7%) were experience mismatch at screening.

**3. Panel load agent**
- **CSV:** `interviewer_load.csv`
- **Question it asks:** are some interviewers overloaded?
- **What it calculates:** for each interviewer and each week, interviews assigned against capacity, the utilization percentage, and whether they're over, at or under capacity. It also says how many interviews could be reassigned.
- **Example finding:** I003 had 28 interviews against a capacity of 20, and was over capacity in all 4 weeks.

**4. Offer insights agent**
- **CSV:** `offers.csv`
- **Question it asks:** are offers being accepted, and if not, why?
- **What it calculates:** the total offers, accepted and declined counts, the acceptance rate, the decline reasons, the salary summary, and the offer speed (offers sent within 7 days, and offers delayed over 14 days).
- **Example finding:** 4 of 11 resolved offers were declined, and 3 of those 4 cited compensation.

**5. Pipeline health agent**
- **CSV:** `candidates.csv`
- **Question it asks:** where are candidates stuck, and for how long?
- **What it calculates:** the candidates in each stage, the days spent in each stage against a target (applied 3, screen 7, interview 5 days), the SLA breaches, the roles open too long, and any possible bottleneck.
- **Example finding:** 3 candidates in the Interview stage have waited 12 days on average, against a 5-day target.

So `candidates.csv` feeds three agents, `offers.csv` two, and the interview feedback and interviewer load files one each. Every agent also returns a list of limitations, such as "small samples don't prove trends," which is passed on to the AI later so it stays cautious.

## Path A: the manager opens the dashboard

When the dashboard opens, the app reads the four CSVs and checks that they're valid. If one is broken, it stops and says which. Then the five agents run one after another, each producing a small summary of numbers. The dashboard draws those as charts and tables. There's no LLM and no LangGraph here, and it costs nothing. The manager sees what the data says, and nothing more.

## Path B: the manager clicks Generate AI Insights

Now the numbers aren't enough, and the manager wants advice. This is where LangGraph starts. LangGraph is my orchestrator. It doesn't think. It runs a fixed list of steps in order and handles failures.

1. It validates the CSVs again.
2. It loads my guideline documents into Chroma, the small local database. These are six markdown files with things like SLA targets and capacity limits.
3. It runs the five agents again to get their numbers.
4. For each agent, it fetches the 3 guideline sections that match that agent's topic. This is plain code, not an agent and not an AI.
5. The routing agent looks at each agent's analysis and decides how hard it is. Easy ones go to the cheaper model, hard ones to the stronger model, and every choice is logged.
6. LLM1 receives the agent's numbers plus the 3 guideline sections. It writes the recommendation, the evidence, a confidence score and an alternative action. Python adds the cost, not the AI. If the call fails, the system retries up to 3 times, and if it still fails that agent shows a flagged fallback card and the others carry on.

## How each agent gets its guidelines

The guidelines live in six markdown files, one per agent plus one shared file on responsible use. When Generate is clicked, the system splits each file at its headings into 21 sections, stores them in Chroma, and tags each section with its file name, such as `offers` or `panel_load`.

Each agent has one fixed question written in the code. For example, the panel load agent asks "interviewer capacity utilisation rebalancing." The system sends that question to Chroma with a filter, so only sections tagged with that agent's own file plus the shared `responsible_use` file are searched. Chroma returns the 3 sections closest in meaning to the question. This is plain code with no AI involved, and the agents themselves don't do it.

Those 3 sections are attached to the agent's numbers and sent to LLM1. If retrieval fails, the agent still runs without guidelines, and the AI is told not to claim guideline support. Because the questions are fixed, the same sections come back each run. Building the question from each analysis's findings would be a possible improvement.

## The checker

After LLM1 has written all five insights, the checker reviews them, because AI sometimes makes things up. It has two parts. First, fixed code rules check for invented numbers, claims that don't match the data, and wording like "automatically reject this candidate". Second, LLM2 acts as a judge and scores each insight for how actionable it is, how well it's grounded in the data, and how likely it is to contain made-up claims. A broken rule makes an insight Red on its own. Otherwise LLM2's scores decide Green, Amber or Red. LLM2 doesn't fix or rewrite anything. It only marks.

## The golden set

The golden set is a separate test of the checker. It is not part of the manager's flow and it never runs when the dashboard is used. It answers one question: is my checker any good?

**What it contains.** It has 20 rows in `data/golden/golden_scenarios.json`. Each row has:
- the **numbers**: a real analysis from one of the five agents
- an **insight** written by me for that agent: 8 are good insights, and 12 have a deliberate mistake (an invented number, a false bottleneck, "10 interviewers" when only 4 are over capacity, a bias claim, vague advice)
- the **correct answer** that I decided in advance: accept or reject
- a note on what is wrong, and whether code rules or LLM2 should catch it

**How it is run.** I start it by hand with `python run_golden_eval.py`. For each of the 20 rows, the numbers and the insight go into the checker, the same checker that reviews real insights: the code rules, then LLM2. The checker marks the insight Green, Amber or Red. Green and Amber count as **accept**, and Red counts as **reject**.

**How it is scored.** The checker's answer for each row is compared with the correct answer I wrote in advance. The result is:
- **accuracy:** the share of the 20 rows where the checker agreed with me
- **bad insights caught:** how many of the 12 bad ones it rejected
- **good insights wrongly rejected:** how many of the 8 good ones it rejected by mistake

**My results.** 90% accuracy: it caught 10 of the 12 bad insights and wrongly rejected none of the 8 good ones. The two it missed are a "highest acceptance" claim that rested on one offer, and some vague generic advice. Both only got an Amber.

**When to run it.** After any change to the checker (the rules, the judge prompt or the judge model), and once before a demo. It costs a few cents. `python run_golden_eval.py --no-llm` runs only the code rules, for free.

**Honest limits.** Only 20 rows, all written by me. I added some rules after seeing the first misses, so 90% is a tuned result, not a test on unseen cases.

## What the manager sees

Each of the five insights appears with its Green, Amber or Red mark, so the manager knows which ones to trust. Next to it are the cost, the speed and the model used for each agent, and a table showing how much money routing and input compression saved: 9.1% from routing and 15 to 20% in total.

## In one line

Code does the counting so the numbers are right, LLM1 writes the advice, the checker catches its mistakes, and LangGraph keeps the whole thing in order and stops one failure from breaking the rest.
