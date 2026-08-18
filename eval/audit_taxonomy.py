"""
Benchmark audit: applies the four-way annotation-error taxonomy from Jin, Choi,
Zhu & Kang, "Text-to-SQL Benchmarks are Broken: An In-Depth Analysis of Annotation
Errors" (CIDR 2026, vldb.org/cidrdb/papers/2026/p5-jin.pdf) to a 100-question
stratified sample of BIRD Mini-Dev, and reports a corrected-subset EX delta for
this project's agent.

METHODOLOGY NOTE - read before citing this number: the CIDR paper's audit was
done by the paper's human authors, cross-checking each example against the
schema and (for domain-specific questions) external resources. This script
instead uses a frontier LLM (JUDGE_MODEL) as the auditor, given the exact
taxonomy definitions verbatim from the paper. That is a faster, cheaper, and
less rigorous process than the paper's - it will disagree with a careful human
reviewer on genuinely ambiguous cases. Treat this as a systematic, reproducible
approximation of the paper's method, not a replication of it. A sample of the
LLM auditor's own judgments should be spot-checked by a human before the
resulting numbers are treated as final.

Taxonomy (quoted from the paper, Section 1):
  E1. Mismatches between semantics of the question Q and the intended logic of
      the task T (the annotator's SQL doesn't implement what the question asks).
  E2. Mismatches between semantics of Q and the data/schema D, due to limited
      understanding of the data and schema (e.g. missing a filter the schema
      requires, or joining on the wrong column).
  E3. Mismatches between semantics of Q and domain knowledge relevant to T, or
      misannotated domain knowledge (e.g. misunderstanding what "K-12" means).
  E4. Ambiguity in T - the question or evidence doesn't clearly specify which
      column/table/interpretation is intended.

Usage:
    python -m eval.audit_taxonomy --n 100 --out eval/audit_results.json
"""
import argparse
import json
import os
import random
import sqlite3

from dotenv import load_dotenv

load_dotenv()

import anthropic

from agent.graph import FRONTIER_MODEL
from agent.tools import get_schema_text

DATA_DIR = os.getenv("BIRD_DATA_DIR", "./data/minidev/MINIDEV")
QUESTIONS_PATH = os.path.join(DATA_DIR, "mini_dev_sqlite.json")
DB_DIR = os.path.join(DATA_DIR, "dev_databases")

AUDITOR_SYSTEM_PROMPT = """You are auditing a text-to-SQL benchmark's gold-label quality.

You will be shown a natural-language question, an optional evidence/hint string, the
database schema, the annotated "gold" SQL query, and the result that gold query
actually produces. Your job is to judge whether the gold SQL correctly implements
what the question asks - not whether it is stylistically ideal.

Classify using EXACTLY this taxonomy (quoted from Jin, Choi, Zhu & Kang, "Text-to-SQL
Benchmarks are Broken", CIDR 2026):

E1. Mismatches between semantics of the question and the intended logic of the SQL -
    the query does not implement what the question literally asks (e.g. wrong
    comparison operator, wrong aggregation, BETWEEN used where strict inequality
    was needed).
E2. Mismatches between the question's semantics and the data/schema, due to limited
    understanding of the schema - e.g. a required WHERE filter that the schema
    demands is missing, or a join uses the wrong key.
E3. Mismatches between the question's semantics and domain knowledge - the query
    reflects a misunderstanding of what a domain term means (e.g. "K-12" wrongly
    scoped, a specialized field misread).
E4. Ambiguity - the question or evidence does not clearly specify which column,
    table, or interpretation is intended, so multiple SQL queries could be
    considered valid answers.

If the gold SQL correctly implements the question with no such issue, classify as VALID.

Respond with ONLY a JSON object, no markdown:
{
  "verdict": "VALID" | "E1" | "E2" | "E3" | "E4",
  "explanation": "one or two sentences, citing the specific mismatch if any",
  "corrected_sql": "a corrected SQL query if verdict is not VALID, else null"
}"""

AUDITOR_USER_TEMPLATE = """SCHEMA:
{schema}

QUESTION: {question}
EVIDENCE: {evidence}

GOLD SQL:
{sql}

GOLD SQL RESULT (first 10 rows, or an error):
{gold_result}
"""

_client = None


def get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def run_sql_for_audit(db_path: str, sql: str) -> str:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        try:
            rows = conn.execute(sql).fetchmany(10)
            if not rows:
                return "(0 rows)"
            return "\n".join(str(r) for r in rows)
        finally:
            conn.close()
    except sqlite3.Error as e:
        return f"ERROR: {e}"


def stratified_sample(questions: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample proportionally to the difficulty distribution already present in
    the dataset, mirroring the paper's own 62/28/10 simple/moderate/challenging
    split of their 100-example sample (Section 5.1)."""
    rng = random.Random(seed)
    by_difficulty: dict[str, list[dict]] = {}
    for q in questions:
        by_difficulty.setdefault(q["difficulty"], []).append(q)

    total = len(questions)
    sample = []
    for difficulty, group in by_difficulty.items():
        share = round(n * len(group) / total)
        sample.extend(rng.sample(group, min(share, len(group))))
    # Top up/trim to exactly n if rounding left us short/over.
    rng.shuffle(sample)
    if len(sample) > n:
        sample = sample[:n]
    elif len(sample) < n:
        remaining = [q for q in questions if q not in sample]
        sample.extend(rng.sample(remaining, n - len(sample)))
    return sample


def audit_one(question: dict) -> dict:
    db_path = os.path.join(DB_DIR, question["db_id"], f"{question['db_id']}.sqlite")
    schema, _ = get_schema_text(db_path)
    gold_result = run_sql_for_audit(db_path, question["SQL"])

    prompt = AUDITOR_USER_TEMPLATE.format(
        schema=schema[:6000],
        question=question["question"],
        evidence=question.get("evidence") or "(none)",
        sql=question["SQL"],
        gold_result=gold_result,
    )
    response = get_client().messages.create(
        model=FRONTIER_MODEL,
        max_tokens=1024,
        system=AUDITOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    raw = "".join(text_blocks).strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        verdict = {"verdict": "PARSE_ERROR", "explanation": raw[:300], "corrected_sql": None}

    return {
        "question_id": question["question_id"],
        "db_id": question["db_id"],
        "difficulty": question["difficulty"],
        "question": question["question"],
        "gold_sql": question["SQL"],
        **verdict,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", default="eval/audit_results.json")
    args = parser.parse_args()

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        all_questions = json.load(f)

    sample = stratified_sample(all_questions, args.n)
    by_diff = {}
    for q in sample:
        by_diff[q["difficulty"]] = by_diff.get(q["difficulty"], 0) + 1
    print(f"Auditing {len(sample)} questions: {by_diff}")

    results = []
    for i, q in enumerate(sample, 1):
        r = audit_one(q)
        results.append(r)
        print(f"  [{i}/{len(sample)}] q{r['question_id']}: {r['verdict']}")

    n_erroneous = sum(1 for r in results if r["verdict"] != "VALID" and r["verdict"] != "PARSE_ERROR")
    by_pattern = {}
    for r in results:
        v = r["verdict"]
        if v not in ("VALID", "PARSE_ERROR"):
            by_pattern[v] = by_pattern.get(v, 0) + 1

    summary = {
        "n": len(results),
        "error_rate": n_erroneous / len(results),
        "n_erroneous": n_erroneous,
        "by_pattern": by_pattern,
    }
    print(f"\n=== Audit summary ===")
    print(f"Error rate: {summary['error_rate']*100:.1f}% ({n_erroneous}/{len(results)})")
    print(f"By pattern: {by_pattern}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
