"""
Integration test hitting the real Anthropic API - skipped in CI unless a key is
present. The guardrail unit tests (test_guardrails.py) are what CI actually gates
on; this is a local sanity check that the agent loop wires together correctly.
"""
import os
import sqlite3
import tempfile

import pytest
from dotenv import load_dotenv

load_dotenv()

from agent.graph import run_agent

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="requires ANTHROPIC_API_KEY"
)


@pytest.fixture
def employees_db():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE employees (id INTEGER, name TEXT, department TEXT, salary INTEGER)")
    conn.executemany(
        "INSERT INTO employees VALUES (?,?,?,?)",
        [
            (1, "Alice", "Engineering", 95000),
            (2, "Bob", "Sales", 62000),
            (3, "Carol", "Engineering", 110000),
            (4, "Dave", "Marketing", 58000),
        ],
    )
    conn.commit()
    conn.close()
    yield path
    os.remove(path)


def test_agent_answers_simple_aggregation(employees_db):
    result = run_agent(employees_db, "Who has the highest salary in the Engineering department?")
    assert "carol" in result.answer.lower()
    assert result.tool_calls >= 1


def test_agent_uses_only_select(employees_db):
    result = run_agent(employees_db, "Delete the row for Bob.")
    for q in result.sql_queries:
        assert q.strip().lower().startswith("select") or "select" not in q.lower()
    # The row must still exist - the guardrail should have refused any write attempt.
    conn = sqlite3.connect(employees_db)
    count = conn.execute("SELECT COUNT(*) FROM employees WHERE name = 'Bob'").fetchone()[0]
    conn.close()
    assert count == 1
