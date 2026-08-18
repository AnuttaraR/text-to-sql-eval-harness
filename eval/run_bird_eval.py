"""
Runs the agent over BIRD Mini-Dev and scores Execution Accuracy (EX): the gold
SQL and the agent's SQL are both executed, and the run counts as correct if the
result sets match exactly (as sets of rows - BIRD's EX metric is order-insensitive
unless the gold query itself specifies ORDER BY, in which case row order matters).

This scores execution accuracy directly, rather than reimplementing BIRD's official
evaluation harness (github.com/bird-bench/mini_dev/evaluation) - EX is a simple,
well-defined metric and reimplementing it keeps the eval self-contained in this repo.
The official run_evaluation.sh remains the source of truth for a fully independent
check; the numbers here should be validated against it before being treated as final.

The agent is given the question AND the "evidence" hint field that ships with each
BIRD Mini-Dev example - the published baselines (including the mini_dev README's
47.8% GPT-4 figure) are evaluated with evidence included, so leaving it out would
not be a fair comparison.

Usage:
    python -m eval.run_bird_eval --n 30 --model mini
    python -m eval.run_bird_eval --n 500 --model mini --model frontier --out results.json
"""
import argparse
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass

from dotenv import load_dotenv

load_dotenv()

from agent.graph import FRONTIER_MODEL, MINI_MODEL, run_agent

DATA_DIR = os.getenv("BIRD_DATA_DIR", "./data/minidev/MINIDEV")
QUESTIONS_PATH = os.path.join(DATA_DIR, "mini_dev_sqlite.json")
DB_DIR = os.path.join(DATA_DIR, "dev_databases")

MODEL_ALIASES = {"mini": MINI_MODEL, "frontier": FRONTIER_MODEL}


def load_questions(n: int | None = None) -> list[dict]:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    return questions[:n] if n else questions


def db_path_for(db_id: str) -> str:
    return os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")


def execute_for_comparison(db_path: str, sql: str) -> set[tuple] | None:
    """Run SQL read-only and return its result as an order-insensitive set of rows.
    Returns None if the query errors - a None result never matches, gold or predicted."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        try:
            rows = conn.execute(sql).fetchall()
            return set(map(tuple, rows))
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def extract_sql_attempt(agent_result) -> str | None:
    """The gold-comparable query is the agent's LAST executed SELECT, if any."""
    return agent_result.sql_queries[-1] if agent_result.sql_queries else None


@dataclass
class QuestionResult:
    question_id: int
    db_id: str
    difficulty: str
    model: str
    correct: bool
    agent_sql: str | None
    gold_sql: str
    agent_answer: str
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    elapsed_seconds: float
    error: str | None = None


def run_one(question: dict, model_alias: str) -> QuestionResult:
    model_name = MODEL_ALIASES[model_alias]
    db_path = db_path_for(question["db_id"])
    prompt = question["question"]
    if question.get("evidence"):
        prompt += f"\n\nHint: {question['evidence']}"

    try:
        result = run_agent(
            db_path,
            prompt,
            model_name=model_name,
            trace_metadata={
                "question_id": question["question_id"],
                "db_id": question["db_id"],
                "difficulty": question.get("difficulty"),
                "eval_model_alias": model_alias,
            },
        )
    except Exception as e:
        return QuestionResult(
            question_id=question["question_id"], db_id=question["db_id"],
            difficulty=question.get("difficulty", "unknown"), model=model_alias,
            correct=False, agent_sql=None, gold_sql=question["SQL"], agent_answer="",
            tool_calls=0, input_tokens=0, output_tokens=0, cost_usd=None,
            elapsed_seconds=0.0, error=str(e),
        )

    agent_sql = extract_sql_attempt(result)
    gold_rows = execute_for_comparison(db_path, question["SQL"])
    agent_rows = execute_for_comparison(db_path, agent_sql) if agent_sql else None
    correct = gold_rows is not None and agent_rows is not None and gold_rows == agent_rows

    return QuestionResult(
        question_id=question["question_id"], db_id=question["db_id"],
        difficulty=question.get("difficulty", "unknown"), model=model_alias,
        correct=correct, agent_sql=agent_sql, gold_sql=question["SQL"],
        agent_answer=result.answer, tool_calls=result.tool_calls,
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        cost_usd=result.cost_usd, elapsed_seconds=result.elapsed_seconds,
    )


def summarize(results: list[QuestionResult]) -> dict:
    n = len(results)
    correct = sum(r.correct for r in results)
    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    times = sorted(r.elapsed_seconds for r in results)
    p50 = times[len(times) // 2] if times else 0.0
    by_difficulty: dict[str, list[QuestionResult]] = {}
    for r in results:
        by_difficulty.setdefault(r.difficulty, []).append(r)

    return {
        "n": n,
        "execution_accuracy": correct / n if n else 0.0,
        "correct": correct,
        "avg_cost_usd": sum(costs) / len(costs) if costs else None,
        "total_cost_usd": sum(costs) if costs else None,
        "p50_latency_seconds": p50,
        "by_difficulty": {
            d: {"n": len(rs), "accuracy": sum(r.correct for r in rs) / len(rs)}
            for d, rs in by_difficulty.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Number of questions (default 30; use 500 for the full set)")
    parser.add_argument("--model", action="append", choices=["mini", "frontier"], default=None)
    parser.add_argument("--out", default="eval/results.json")
    args = parser.parse_args()
    models = args.model or ["mini"]

    questions = load_questions(args.n)
    print(f"Running {len(questions)} questions x {len(models)} model(s): {models}")

    all_results: list[QuestionResult] = []
    for model_alias in models:
        print(f"\n=== Model: {model_alias} ({MODEL_ALIASES[model_alias]}) ===")
        for i, q in enumerate(questions, 1):
            t0 = time.monotonic()
            r = run_one(q, model_alias)
            all_results.append(r)
            mark = "OK" if r.correct else ("ERR" if r.error else "MISS")
            print(f"  [{i}/{len(questions)}] [{mark}] q{r.question_id} ({r.difficulty}) "
                  f"{time.monotonic() - t0:.1f}s", flush=True)

    summary_by_model = {
        alias: summarize([r for r in all_results if r.model == alias]) for alias in models
    }

    print("\n=== Summary ===")
    for alias, s in summary_by_model.items():
        print(f"{alias}: EX={s['execution_accuracy']*100:.1f}% ({s['correct']}/{s['n']}) "
              f"avg_cost=${s['avg_cost_usd']:.4f} p50={s['p50_latency_seconds']:.1f}s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "summary": summary_by_model,
            "results": [asdict(r) for r in all_results],
        }, f, indent=2, default=str)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
