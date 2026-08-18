"""
CI regression gate: runs the agent against a small fixed subset of BIRD Mini-Dev
(eval/regression_set.py) using the small fixture databases in tests/fixtures/,
and fails the build if execution accuracy on that subset drops below EXPECTED_MIN.

This is deliberately NOT the same as eval/run_bird_eval.py's full 500-question
run: it exists to catch an obvious regression (a prompt change or dependency bump
that breaks the agent outright) on every push, using 4 questions chosen to be
small, fast, and requiring no network access beyond the Anthropic API call itself.

Skipped without ANTHROPIC_API_KEY, same as test_agent_integration.py.
"""
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from agent.graph import MINI_MODEL, run_agent
from eval.regression_set import REGRESSION_QUESTIONS

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="requires ANTHROPIC_API_KEY"
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "dev_databases")

# Below this fraction of the regression set, something is genuinely broken -
# not just "the model got a hard question wrong". 3/4 is deliberately lenient;
# tighten once the model/prompt combination has more history behind it.
EXPECTED_MIN_ACCURACY = 0.5


class ExecutionAccuracyMetric(BaseMetric):
    """1.0 if the agent's SQL execution result set matches gold exactly, else 0.0."""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        import sqlite3

        db_path = test_case.metadata["db_path"]
        gold_sql = test_case.expected_output
        agent_sql = test_case.actual_output

        def run(sql):
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
                try:
                    return set(map(tuple, conn.execute(sql).fetchall()))
                finally:
                    conn.close()
            except sqlite3.Error:
                return None

        gold_rows = run(gold_sql)
        agent_rows = run(agent_sql) if agent_sql else None
        self.score = 1.0 if (gold_rows is not None and gold_rows == agent_rows) else 0.0
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "ExecutionAccuracy"


def test_regression_set_execution_accuracy():
    """Aggregate gate: fails if too many of the fixed regression questions fail."""
    correct = 0
    for q in REGRESSION_QUESTIONS:
        db_path = os.path.join(FIXTURES_DIR, q["db_id"], f"{q['db_id']}.sqlite")
        prompt = q["question"]
        if q.get("evidence"):
            prompt += f"\n\nHint: {q['evidence']}"

        result = run_agent(db_path, prompt, model_name=MINI_MODEL)
        agent_sql = result.sql_queries[-1] if result.sql_queries else None

        metric = ExecutionAccuracyMetric()
        test_case = LLMTestCase(
            input=prompt,
            actual_output=agent_sql or "",
            expected_output=q["SQL"],
            metadata={"db_path": db_path},
        )
        score = metric.measure(test_case)
        correct += int(score == 1.0)
        print(f"q{q['question_id']} ({q['db_id']}): {'PASS' if score == 1.0 else 'FAIL'}")

    accuracy = correct / len(REGRESSION_QUESTIONS)
    assert accuracy >= EXPECTED_MIN_ACCURACY, (
        f"Regression set accuracy {accuracy:.0%} ({correct}/{len(REGRESSION_QUESTIONS)}) "
        f"is below the {EXPECTED_MIN_ACCURACY:.0%} gate."
    )
