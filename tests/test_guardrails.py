import os
import sqlite3
import tempfile

import pytest

from agent.guardrails import (
    GuardrailViolation,
    assert_single_select,
    enforce_row_limit,
    run_guarded_query,
)


@pytest.fixture
def sample_db():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.executemany("INSERT INTO users VALUES (?, ?)", [(i, f"user{i}") for i in range(500)])
    conn.execute("CREATE TABLE secrets (id INTEGER, value TEXT)")
    conn.executemany("INSERT INTO secrets VALUES (?, ?)", [(1, "top-secret")])
    conn.commit()
    conn.close()
    yield path
    os.remove(path)


def test_rejects_multi_statement():
    with pytest.raises(GuardrailViolation):
        assert_single_select("SELECT * FROM users; DROP TABLE users;")


def test_rejects_ddl():
    with pytest.raises(GuardrailViolation):
        assert_single_select("DROP TABLE users")


def test_rejects_dml():
    with pytest.raises(GuardrailViolation):
        assert_single_select("DELETE FROM users WHERE id = 1")


def test_accepts_plain_select():
    stmt = assert_single_select("SELECT id, name FROM users WHERE id > 10")
    assert stmt is not None


def test_row_limit_injected_when_absent():
    stmt = assert_single_select("SELECT * FROM users")
    stmt = enforce_row_limit(stmt, max_rows=50)
    assert "LIMIT 50" in stmt.sql()


def test_row_limit_clamped_when_too_large():
    stmt = assert_single_select("SELECT * FROM users LIMIT 100000")
    stmt = enforce_row_limit(stmt, max_rows=50)
    assert "LIMIT 50" in stmt.sql()


def test_schema_allowlist_blocks_unlisted_table(sample_db):
    with pytest.raises(GuardrailViolation):
        run_guarded_query(sample_db, "SELECT * FROM secrets", allowed_tables={"users"})


def test_schema_allowlist_allows_listed_table(sample_db):
    result = run_guarded_query(sample_db, "SELECT * FROM users LIMIT 5", allowed_tables={"users"})
    assert len(result.rows) == 5


def test_row_limit_enforced_end_to_end(sample_db):
    result = run_guarded_query(
        sample_db, "SELECT * FROM users", allowed_tables={"users"}, max_rows=10
    )
    assert len(result.rows) == 10


def test_write_through_select_is_rejected(sample_db):
    # A crafted "SELECT" that isn't actually read-only should never parse to a
    # single clean SELECT AST - but also verify the read-only connection itself
    # would refuse a write if one somehow got through.
    with pytest.raises((GuardrailViolation, sqlite3.OperationalError)):
        run_guarded_query(
            sample_db,
            "UPDATE users SET name = 'x' WHERE id = 1",
            allowed_tables={"users"},
        )
