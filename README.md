# Text-to-SQL Evaluation Harness

A LangGraph SQL agent scored on **BIRD Mini-Dev**, shipped with an honest failure
analysis of both the agent and the benchmark itself.

Status: scaffolding in progress. This README fills in as each stage lands - see
`EVAL_FINDINGS.md` (once written) for the benchmark audit.

---

## Why this exists

Most text-to-SQL demos stop at "it answered a question correctly once." This project
instead asks: how often is it *actually* correct, at what cost, at what latency - and
how much of the benchmark used to answer that question is itself wrong.

## What's here

1. **The agent** - a LangGraph tool-calling loop: retrieve schema, execute a guarded
   read-only SQL query, self-repair on error, return an answer.
2. **The eval** - the agent run against all 500 SQLite instances in BIRD Mini-Dev,
   scored with the official Execution Accuracy (EX) metric, for two models (a
   mini-class default and one frontier ceiling run), with $/query and p50 latency.
3. **The audit** - a hand-labeled review of 100 Mini-Dev examples against the CIDR
   2026 error taxonomy from *"Text-to-SQL Benchmarks are Broken"*, which found a
   52.8% annotation error rate in this benchmark. The corrected-subset delta is
   reported alongside the raw score, not instead of it.

## Guardrails

- SQLite connection opened in a read-only role at the connection level, not enforced
  by a prompt.
- `sqlglot`-based AST check: the statement must parse as a single `SELECT`.
- `LIMIT` injection on unbounded queries, statement timeout, schema allowlist, and a
  max tool-call depth cap on the agent loop.

## Stack

LangGraph, SQLite/SQLAlchemy, `sqlglot`, DeepEval (CI-gated regression tests),
Langfuse (tracing + cost), Streamlit (eval dashboard).

## Results

TODO - filled in once `eval/run_bird_eval.py` has a completed run against all 500
instances for both models, and the 100-example audit is done.

## Run it

```bash
python -m venv .venv311
.venv311\Scripts\activate      # Windows
pip install -r requirements.txt
cp .env.example .env            # set ANTHROPIC_API_KEY
python scripts/fetch_bird_data.py
python eval/run_bird_eval.py --model mini
```

## Limitations

TODO - fill in honestly once the eval has run: known agent failure modes, benchmark
caveats (mini-dev vs full-dev numbers are not comparable - see BIRD Mini-Dev section
above), judge calibration ceiling.
