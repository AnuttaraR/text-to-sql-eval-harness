"""
Tools available to the SQL agent: schema introspection and guarded query execution.

Built per-database via `build_tools(db_path)` rather than as free functions, since
each BIRD Mini-Dev question targets a different SQLite file and a different table
allowlist. The allowlist is derived from the database's own schema, not passed in
by the caller - the agent can query any table that actually exists in this database,
just not tables from other databases or anything outside SELECT.
"""
import sqlite3

from langchain_core.tools import tool

from agent.guardrails import GuardrailViolation, run_guarded_query


def get_schema_text(db_path: str) -> tuple[str, set[str]]:
    """Return (human-readable schema description, set of table names) for a SQLite file."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    tables = {name for name, _ in rows}
    schema_text = "\n\n".join(sql for _, sql in rows if sql)
    return schema_text, tables


def build_tools(db_path: str) -> list:
    """Return the [get_schema, run_sql] tool pair bound to one SQLite database."""
    schema_text, allowed_tables = get_schema_text(db_path)

    @tool
    def get_schema() -> str:
        """Return the CREATE TABLE statements for every table in this database."""
        return schema_text

    @tool
    def run_sql(query: str) -> str:
        """
        Execute a read-only SELECT query against the database and return the result
        rows. The query must be a single SELECT statement - no writes, no DDL, no
        multiple statements. Results are capped at 200 rows.
        """
        try:
            result = run_guarded_query(db_path, query, allowed_tables=allowed_tables)
        except GuardrailViolation as e:
            return f"QUERY REJECTED: {e}"
        except sqlite3.OperationalError as e:
            return f"SQL ERROR: {e}"

        if not result.rows:
            return "Query returned 0 rows."

        header = ", ".join(result.columns)
        body = "\n".join(", ".join(str(v) for v in row) for row in result.rows)
        note = " (truncated to 200 rows)" if result.truncated else ""
        return f"{header}\n{body}{note}"

    return [get_schema, run_sql]
