"""
Calibrates an LLM-as-judge against objective ground truth.

Execution accuracy (EX) is *programmatic* ground truth - we can execute both the
gold and predicted SQL and compare result sets exactly, no judgment call needed.
That makes it a rare case where an LLM judge's verdicts can be validated directly,
rather than trusted on faith. This script:

  1. Takes a completed eval run (from eval/run_bird_eval.py, which already has the
     objective EX verdict per question).
  2. Shows a judge model ONLY the question, the gold execution result, and the
     agent's final natural-language answer (not the SQL, not the EX verdict) -
     the realistic inputs available in a setting where you can't execute SQL
     yourself and have to judge from the answer text alone.
  3. Compares the judge's verdict to the objective EX verdict and reports
     Cohen's kappa (agreement corrected for chance).

Convention (see e.g. Landis & Koch 1977): kappa > 0.6 is usually read as
acceptable agreement, > 0.8 as production-ready, < 0.5 as "rewrite the rubric."

Usage:
    python -m eval.judge_calibration --results eval/results_mini_full.json --n 100
"""
import argparse
import json
import os
import sqlite3

from dotenv import load_dotenv

load_dotenv()

import anthropic
from sklearn.metrics import cohen_kappa_score

from agent.graph import FRONTIER_MODEL

DATA_DIR = os.getenv("BIRD_DATA_DIR", "./data/minidev/MINIDEV")
DB_DIR = os.path.join(DATA_DIR, "dev_databases")
QUESTIONS_PATH = os.path.join(DATA_DIR, "mini_dev_sqlite.json")


def load_question_text_by_id() -> dict[int, str]:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    return {q["question_id"]: q["question"] for q in questions}

JUDGE_SYSTEM_PROMPT = """You are grading whether an AI assistant's answer to a
database question is correct. You are given the question and the correct answer
(computed by executing the gold SQL query), plus the assistant's answer in its
own words. Judge whether the assistant's answer conveys the same information as
the correct answer - allow for reasonable differences in formatting, rounding to
a sensible precision, or ordering, but not differences in the actual value(s).

Respond with ONLY a JSON object, no markdown:
{"correct": true | false, "reason": "one sentence"}"""

JUDGE_USER_TEMPLATE = """QUESTION: {question}

CORRECT ANSWER (from executing the gold query): {gold_answer}

ASSISTANT'S ANSWER: {agent_answer}"""

_client = None


def get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def gold_answer_text(db_id: str, gold_sql: str) -> str:
    db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        try:
            rows = conn.execute(gold_sql).fetchmany(20)
            return "; ".join(str(r) for r in rows) if rows else "(0 rows)"
        finally:
            conn.close()
    except sqlite3.Error as e:
        return f"(gold query errored: {e})"


def judge_one(question: str, gold_answer: str, agent_answer: str) -> bool:
    prompt = JUDGE_USER_TEMPLATE.format(
        question=question, gold_answer=gold_answer, agent_answer=agent_answer
    )
    response = get_client().messages.create(
        model=FRONTIER_MODEL,
        max_tokens=512,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    raw = "".join(text_blocks).strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])
    try:
        return json.loads(raw)["correct"]
    except (json.JSONDecodeError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to a run_bird_eval.py output JSON")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", default="eval/judge_calibration_results.json")
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)
    results = [r for r in data["results"] if not r.get("error")][: args.n]
    print(f"Calibrating judge on {len(results)} examples from {args.results}")
    question_text = load_question_text_by_id()

    ground_truth = []
    judge_verdicts = []
    details = []
    for i, r in enumerate(results, 1):
        gold_ans = gold_answer_text(r["db_id"], r["gold_sql"])
        judge_says_correct = judge_one(
            question=question_text.get(r["question_id"], ""),
            gold_answer=gold_ans,
            agent_answer=r["agent_answer"],
        )
        ex_says_correct = bool(r["correct"])
        ground_truth.append(ex_says_correct)
        judge_verdicts.append(judge_says_correct)
        details.append({
            "question_id": r["question_id"],
            "ex_correct": ex_says_correct,
            "judge_correct": judge_says_correct,
            "agree": ex_says_correct == judge_says_correct,
        })
        print(f"  [{i}/{len(results)}] q{r['question_id']}: EX={ex_says_correct} judge={judge_says_correct}"
              f"{'  <-- DISAGREE' if ex_says_correct != judge_says_correct else ''}")

    kappa = cohen_kappa_score(ground_truth, judge_verdicts)
    agreement = sum(d["agree"] for d in details) / len(details)

    print(f"\n=== Judge calibration ===")
    print(f"Raw agreement: {agreement*100:.1f}%")
    print(f"Cohen's kappa: {kappa:.3f}")

    with open(args.out, "w") as f:
        json.dump({
            "n": len(details), "kappa": kappa, "raw_agreement": agreement,
            "details": details,
        }, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
