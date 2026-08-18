# Text-to-SQL Evaluation Harness

A LangGraph SQL agent scored on **BIRD Mini-Dev**, shipped with an honest failure
analysis of both the agent and the benchmark itself.

---

## Why this exists

Most text-to-SQL demos stop at "it answered a question correctly once." This project
instead asks: how often is it *actually* correct, at what cost, at what latency, and
how much of the benchmark used to answer that question is itself wrong.

## What's here

1. **The agent** - a LangGraph tool-calling loop: retrieve schema, execute a guarded
   read-only SQL query, self-repair on error, return an answer.
2. **The eval** - the agent run against all 500 SQLite instances in BIRD Mini-Dev,
   scored with the official Execution Accuracy (EX) metric, for two models (a
   mini-class default and one frontier ceiling run), with $/query and p50 latency.
3. **The audit** - a systematic review of 100 Mini-Dev examples against the CIDR 2026
   error taxonomy from *"Text-to-SQL Benchmarks are Broken"* (Jin, Choi, Zhu & Kang),
   which found a 52.8% annotation error rate in this benchmark. This project's audit
   uses an LLM auditor applying the paper's exact taxonomy, not a human reviewer - see
   `eval/audit_taxonomy.py`'s docstring for exactly what that does and doesn't mean for
   how much to trust the resulting number.

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

Full 500-question BIRD Mini-Dev run, official Execution Accuracy (EX) metric (exact
set-equality between the agent's SQL result and the gold SQL result).

| Model | EX | Correct | Avg. cost/query | Total cost | p50 latency |
|---|---|---|---|---|---|
| Claude Haiku 4.5 (mini) | 53.0% | 265/500 | $0.0096 | $4.57 | 4.8s |
| Claude Sonnet 5 (frontier) | **61.8%** | 309/500 | $0.0190 | $9.37 | 7.4s |

Roughly 2x the cost buys +8.8 points of EX - a real cost/accuracy tradeoff, not just
"bigger model wins," reported as both numbers rather than one.

By difficulty:

| Difficulty | n | Mini EX | Frontier EX |
|---|---|---|---|
| Simple | 148 | 74.3% | 77.7% |
| Moderate | 250 | 48.0% | 57.2% |
| Challenging | 102 | 34.3% | 50.0% |

For reference, the [BIRD Mini-Dev README](https://github.com/bird-bench/mini_dev)
cites a GPT-4 baseline of 47.8% EX and a top reported mini-dev SQLite score around
58% (confirm which exact split before quoting either figure elsewhere - mini-dev and
full BIRD dev numbers are not comparable, and full-dev's leaderboard frontier has
sat around 81-82% EX against a 92.96% human baseline for over a year). This
project's frontier run (61.8%) is above that ~58% figure - a genuinely strong
result, though comparing across different agent harnesses is not perfectly
apples-to-apples, so treat it as encouraging rather than as a leaderboard claim.

## Benchmark audit

100-question stratified sample, audited against the four-way annotation-error
taxonomy from Jin, Choi, Zhu & Kang, *"Text-to-SQL Benchmarks are Broken"* (CIDR
2026), which found a 52.8% annotation error rate on their own BIRD Mini-Dev sample.

**This project's audit found 21% (21/100)** - E1 (semantic mismatch): 15, E2
(schema misunderstanding): 5, E3 (domain knowledge): 1, E4 (ambiguity): 0.

This is a lower bound, not a contradiction of the paper's number: the audit here is
a single LLM-auditor pass (schema + gold query + gold result only), not four human
researchers cross-referencing external resources. The pattern distribution shows
why - E1/E2 are detectable from the SQL and schema alone; E3/E4 typically need
outside knowledge or human judgment the auditor doesn't have. Full breakdown and
methodology caveats in [EVAL_FINDINGS.md](EVAL_FINDINGS.md).

## Judge calibration

Execution Accuracy is *programmatic* ground truth, which makes it possible to
validate an LLM judge against it directly rather than trust the judge on faith.
Given only the question, the gold execution result, and the agent's natural-language
answer (not the SQL, not the EX verdict), a frontier-model judge was scored against
the objective EX outcome on 100 examples:

- **Raw agreement: 80.0%**
- **Cohen's kappa: 0.551**

By the conventional reading (>0.6 acceptable, >0.8 production-ready, <0.5 rewrite
the rubric), this sits below "acceptable" - the judge is measurably more lenient
than exact-match EX, mainly on floating-point precision near-misses (see
[EVAL_FINDINGS.md](EVAL_FINDINGS.md) for the specific disagreement pattern - it is
leniency, not unreliability, but that distinction matters for how much to trust it
unsupervised).

## Run it

```bash
python -m venv .venv311
.venv311\Scripts\activate           # Windows
pip install -r requirements.txt
cp .env.example .env                # set ANTHROPIC_API_KEY
python scripts/fetch_bird_data.py   # ~3.3GB, one time

python -m eval.run_bird_eval --n 500 --model mini --model frontier --out eval/results.json
python -m eval.audit_taxonomy --n 100 --out eval/audit_results.json
python -m eval.judge_calibration --results eval/results.json --n 100

streamlit run dashboard/app.py
```

CI runs a smaller, self-contained regression gate on every push (`tests/test_deepeval_regression.py`)
against 4 committed fixture databases, not the full dataset - see `tests/fixtures/README.md`.

## Deliberately not built

No multi-database connectors. No semantic modelling layer. No chart generation.
No auth or multi-user support. No Spider 1.0 comparison - that leaderboard has been
frozen since February 2024 and citing it would misrepresent the current state of the
field.

## Limitations

- **The benchmark audit is LLM-driven, not human-reviewed.** It systematically
  under-detects domain-knowledge and ambiguity errors (E3/E4) relative to the
  paper's human-audited methodology - treat the 21% figure as a lower bound.
- **The judge calibration measures leniency, not general reliability**, on this
  particular 100-example sample. Its main failure mode is being *too generous* on
  floating-point near-misses, not hallucinating verdicts.
- **Recursion-limit failures count as wrong answers.** A question the agent could
  plausibly have solved with a slightly larger tool-call budget shows up identically
  to a genuine reasoning failure in the EX number.
- **Mini-Dev EX is not comparable to full BIRD Dev EX.** Different question set,
  different difficulty mix - don't read this project's 53% against full-dev
  leaderboard numbers without accounting for that.
- Guardrails prevent unsafe SQL; they do not and cannot improve query correctness.
  A rejected query and a wrong-but-safe query are different failure modes, both
  counted as incorrect by EX.

## Screening-call talking point

*"I scored an agent on BIRD Mini-Dev, then found the benchmark's own gold labels
were unreliable - there's a CIDR 2026 paper putting the annotation error rate at
52.8%. My own audit found 21%, but that's a lower bound: I used an LLM auditor
instead of a human research team, and the errors it systematically misses -
domain knowledge and ambiguity - are exactly the categories that need outside
judgment to catch. That's why I don't trust a single leaderboard number, mine
included, and why the README says so."*
