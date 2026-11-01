# AI-Powered Hiring Intelligence System

Five deterministic hiring-analytics agents, per-agent RAG, model routing, LLM-written insights, and an evaluation agent that checks every insight. Synthetic data only.

Docs: `docs/architecture.png`, `docs/contracts.md`, `docs/evaluation_notes.md`, `docs/reflection.md`.

## Setup (VS Code on Windows / PowerShell)
1. Install Python 3.11 or 3.12 and the VS Code Python extension.
2. Open this folder in VS Code.
3. Run:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. If Python 3.12 isn't installed, use an installed version such as `py -3.11`.

## Configure
Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Data ingestion and the deterministic agents need no key. Never commit `.env`.

## Run
```powershell
python -m src.data_ingestion                       # validate data
python -m scripts.generate_synthetic_data          # regenerate the synthetic CSVs (optional)
python run_llm_hiring_intelligence.py --baseline   # all agents on the large model
python run_llm_hiring_intelligence.py              # routed
python run_llm_hiring_intelligence.py --compress   # routed + panel-load input compression
python run_evaluation.py routed                    # evaluate a saved run (add --no-llm for rules only)
python run_golden_eval.py                          # measure the evaluation agent on the golden set
python -m streamlit run src/dashboard.py           # dashboard
python -m pytest
```

## Layout
- `src/agents/`: five insight agents, routing agent, evaluation agent
- `src/rag/`: knowledge-base indexing and retrieval; `data/knowledge_base/`: one file per agent plus responsible use
- `src/insight_generator.py`: structured insight generation and cost calculation
- `data/golden/`: 20-scenario golden set for the evaluation agent
- `reports/`, `logs/`: run reports, evaluations, routing log

## Cost
Estimated from token usage and per-model prices in `src/insight_generator.py`. Verify current prices before relying on them. Evaluation adds roughly the same cost again per evaluated run.
