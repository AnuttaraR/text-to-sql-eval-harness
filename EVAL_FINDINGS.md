# Eval findings

Running the eval harness surfaced several real bugs and a few methodology decisions
worth recording plainly. This is that record.

## Agent bugs found and fixed

1. **Answering outside SQL.** The agent would sometimes run an exploratory query,
   read the result, then compute the actual answer itself in prose (e.g. subtracting
   two numbers it had just seen in a result table) instead of writing one SQL query
   whose result *was* the answer. Since Execution Accuracy only credits the last
   executed query's result set, this systematically under-scored otherwise-correct
   reasoning. Fixed by adding an explicit instruction to the system prompt: the last
   `run_sql` call must be a query whose result is the final answer, arithmetic and
   all - see `agent/graph.py`.
2. **Extra convenience columns.** Even after fix #1, the agent would often `SELECT`
   an extra descriptive column alongside the requested value (e.g. `Segment` next to
   `CustomerID`, or a subtotal next to a final total) - helpful for a human reading
   chat output, but a guaranteed EX mismatch, since BIRD's gold queries never do
   this. Fixed with an explicit "select only the columns asked for" rule.
3. **`claude-sonnet-5` rejects an explicit `temperature` parameter** ("deprecated
   for this model") - broke every frontier-model call outright until the parameter
   was made conditional on model name.
4. **`claude-sonnet-5` returns thinking blocks in its response content**, not just
   text blocks. Code that assumed `response.content[0].text` crashed; fixed by
   filtering for `type == "text"` blocks specifically.

## What's expected behavior, not a bug

- **Recursion-limit errors on harder questions.** 4 of the first 50 questions hit
  `MAX_TOOL_CALLS_PER_QUERY`'s recursion cap rather than converging. That is the
  guardrail working as designed: failing bounded rather than looping indefinitely
  or executing something outside its permitted scope. It does mean some genuinely
  answerable questions get counted as failures.
- **Floating-point exact-match "failures."** Two early misses turned out to be the
  same value differing at the 8th decimal place (`459.9562642871061` vs
  `459.95626428710585`) - float drift from a different join/computation order in
  SQLite, not a wrong answer. The official BIRD evaluator (`evaluation_ex.py`)
  uses exact set equality with no float tolerance, so this scoring artifact exists
  in the published baseline numbers too. Adding tolerance here would make this
  project's number *not* comparable to the cited baselines, so it was left as-is
  deliberately - see the judge calibration section below for how this shows up.

## Benchmark audit: 21% error rate found, not 52.8%

The CIDR 2026 paper audited BIRD Mini-Dev and found a 52.8% annotation error rate
across their own 100-question sample. This project's audit (`eval/audit_taxonomy.py`,
`eval/audit_results.json`) found 21% (21/100) on a separately drawn 100-question
sample, using the same four-way taxonomy.

That gap is itself informative, not just noise. The paper's audit was done by four
human researchers cross-referencing schemas, external domain resources, and each
other; this project's audit is a single frontier-model pass with only the schema and
the gold query's execution result as context. The pattern distribution makes the gap
legible: this audit found mostly **E1** (semantic mismatch - 15/21) and **E2**
(schema misunderstanding - 5/21) errors, both detectable from the SQL and schema
alone, and almost no **E3** (domain knowledge - 1/21) or **E4** (ambiguity - 0/21),
both of which typically require outside knowledge or human judgment about what a
question *could* reasonably mean. An LLM auditor without external research and
without a second opinion should be expected to systematically under-catch exactly
those two categories - which is what happened.

Reported honestly: this is a **lower bound** on the true annotation error rate, not
a disagreement with the paper's finding.

## Judge calibration: kappa = 0.551

Comparing an LLM judge's verdict (given only the question, the gold execution
result, and the agent's natural-language answer) against the objective EX ground
truth on 100 examples: **80% raw agreement, Cohen's kappa = 0.551**
(`eval/judge_calibration_results.json`).

By the conventional reading (kappa > 0.6 acceptable, > 0.8 production-ready, < 0.5
rewrite the rubric), 0.551 is below "acceptable" - moderate agreement, not
production-grade. Inspecting the 20 disagreements: the judge is consistently more
*lenient* than strict EX, specifically on the floating-point precision cases
described above (it correctly treats `459.9562642871061` and `459.95626428710585`
as the same answer, which a human grader would too) and on minor formatting
differences. It essentially never disagrees with EX in the other direction (calling
something wrong that EX scored correct). That means kappa here is measuring judge
leniency on near-misses, not judge unreliability in general - worth knowing before
using this judge to gate anything automatically, but not the same failure mode as a
judge that hallucinates verdicts.

## Bugs in the harness itself

- **CI `ModuleNotFoundError: No module named 'agent'`.** `python -m pytest` (used
  locally throughout development) adds the working directory to `sys.path`
  automatically; bare `pytest` (what CI actually runs) does not. Fixed with a
  `pytest.ini` (`pythonpath = .`), not by changing every invocation to `-m pytest`,
  so the fix holds regardless of how tests are run.
