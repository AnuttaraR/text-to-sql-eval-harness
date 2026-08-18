"""
Guardrails for the SQL execution tool - enforced structurally, not by prompt.

Deliberately NOT `if "DROP" in sql`: string matching is trivially bypassed
(comments, whitespace, case, encoding). Every check here either runs on the
parsed AST or is enforced by the database connection itself.
"""
import sqlite3
import time
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

DEFAULT_ROW_LIMIT = 200
STATEMENT_TIMEOUT_SECONDS = 8
MAX_TOOL_CALLS_PER_QUERY = 6


class GuardrailViolation(Exception):
    """Raised when a query fails a guardrail check before it reaches the DB."""


@dataclass
class GuardedQueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool
    elapsed_seconds: float


def assert_single_select(sql: str, dialect: str = "sqlite") -> exp.Select:
    """
    Parse `sql` and reject anything that is not exactly one SELECT statement.
    Rejects multi-statement payloads (`; DROP TABLE ...`), DDL/DML, and PRAGMA -
    by AST shape, not substring search.
    """
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except Exception as e:
        raise GuardrailViolation(f"SQL failed to parse: {e}")

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise GuardrailViolation(
            f"Expected exactly one statement, got {len(statements)}. "
            "Multi-statement payloads are rejected."
        )

    stmt = statements[0]
    if not isinstance(stmt, exp.Select):
        raise GuardrailViolation(
            f"Only SELECT statements are permitted, got {type(stmt).__name__}."
        )
    return stmt


def enforce_row_limit(stmt: exp.Select, max_rows: int = DEFAULT_ROW_LIMIT) -> exp.Select:
    """Inject or clamp a LIMIT clause so a runaway SELECT can't return the whole table."""
    existing = stmt.args.get("limit")
    if existing is not None:
        try:
            n = int(existing.expression.this)
            if n > max_rows:
                stmt.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
        except (AttributeError, ValueError):
            stmt.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    else:
        stmt.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    return stmt


def assert_schema_allowlist(stmt: exp.Select, allowed_tables: set[str]) -> None:
    """Reject queries that touch a table outside the schema the agent was given."""
    referenced = {t.name.lower() for t in stmt.find_all(exp.Table)}
    disallowed = referenced - {t.lower() for t in allowed_tables}
    if disallowed:
        raise GuardrailViolation(
            f"Query references table(s) outside the allowed schema: {sorted(disallowed)}"
        )


def run_guarded_query(
    db_path: str,
    sql: str,
    allowed_tables: set[str],
    max_rows: int = DEFAULT_ROW_LIMIT,
    timeout_seconds: int = STATEMENT_TIMEOUT_SECONDS,
    dialect: str = "sqlite",
) -> GuardedQueryResult:
    """
    Validate `sql` against every guardrail, then execute it against a read-only
    connection with a wall-clock statement timeout.
    """
    stmt = assert_single_select(sql, dialect=dialect)
    assert_schema_allowlist(stmt, allowed_tables)
    stmt = enforce_row_limit(stmt, max_rows=max_rows)
    guarded_sql = stmt.sql(dialect=dialect)

    # mode=ro opens the file read-only at the SQLite level - a second, structural
    # enforcement point independent of the AST check above.
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)

    deadline = time.monotonic() + timeout_seconds

    def _progress_handler():
        return time.monotonic() > deadline

    # Called periodically during query execution; returning non-zero aborts the query.
    conn.set_progress_handler(_progress_handler, 1000)

    t0 = time.monotonic()
    try:
        cursor = conn.execute(guarded_sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        rows = rows[:max_rows]
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e).lower():
            raise GuardrailViolation(
                f"Query exceeded the {timeout_seconds}s statement timeout."
            )
        raise
    finally:
        conn.close()

    return GuardedQueryResult(
        columns=columns,
        rows=rows,
        truncated=truncated,
        elapsed_seconds=time.monotonic() - t0,
    )
