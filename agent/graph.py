"""
The text-to-SQL agent: a bounded tool-calling loop over one SQLite database.

Uses LangGraph's prebuilt ReAct agent rather than a hand-rolled graph - the loop
itself (call model, run tool, feed result back, repeat until a final answer) is
exactly the ReAct pattern, and there's nothing project-specific about the control
flow. What IS project-specific lives in agent/guardrails.py and agent/tools.py:
the guardrails run inside the tool, not the graph, so they apply no matter how the
agent decides to call them.
"""
import os
import time
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langfuse.langchain import CallbackHandler

from agent.guardrails import MAX_TOOL_CALLS_PER_QUERY
from agent.tools import build_tools

# USD per million tokens. Verified against https://platform.claude.com/docs/en/about-claude/pricing
# (accessed 2026-08-18) - re-check before relying on these for anything beyond this project's
# own README, since Anthropic revises pricing periodically.
PRICING_PER_MTOK = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
}


def estimate_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float | None:
    rates = PRICING_PER_MTOK.get(model_name)
    if rates is None:
        return None
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


SYSTEM_PROMPT = """You are a SQL analyst answering questions against a single SQLite database.

Rules:
- Call get_schema first if you have not already seen this database's schema.
- Only SELECT statements are permitted. Never attempt INSERT, UPDATE, DELETE, or DDL.
- If run_sql returns "QUERY REJECTED" or "SQL ERROR", read the message, fix the query, and retry.
  Do not repeat the exact same failing query.
- CRITICAL: your LAST run_sql call must be a single query whose result IS the final
  answer. Do not run an exploratory query, then compute the answer yourself in
  prose (e.g. subtracting two numbers you read from a result table). If the
  question needs an aggregation, filter, ORDER BY, or arithmetic between values,
  do that inside the SQL itself - write one query that returns exactly the
  requested value(s), and run it last.
- SELECT only the column(s) the question actually asks for. Do not add extra
  descriptive/context columns (e.g. a name or category alongside an id, or an
  intermediate subtotal alongside a final total) even if they seem helpful -
  they change the result set and must not be present.
- Once you have the data you need, answer with ONLY the final value or row(s) requested.
  Do not explain your reasoning in the final answer. Do not wrap it in a full sentence.
- If the question asks for a single number or name, return just that value.
"""

MINI_MODEL = os.getenv("AGENT_MODEL", "claude-haiku-4-5-20251001")
FRONTIER_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5")


@dataclass
class AgentResult:
    answer: str
    sql_queries: list[str]
    tool_calls: int
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    cost_usd: float | None


def _extract_sql_calls(messages) -> list[str]:
    queries = []
    for m in messages:
        tool_calls = getattr(m, "tool_calls", None) or []
        for call in tool_calls:
            if call.get("name") == "run_sql":
                queries.append(call.get("args", {}).get("query", ""))
    return queries


def run_agent(
    db_path: str,
    question: str,
    model_name: str = MINI_MODEL,
    trace_metadata: dict | None = None,
) -> AgentResult:
    """
    Run the agent against one question. Bounded by MAX_TOOL_CALLS_PER_QUERY.

    Every run is traced to Langfuse (a no-op if LANGFUSE_PUBLIC_KEY isn't set - see
    .env.example). trace_metadata is attached to the trace for filtering in the
    Langfuse UI, e.g. {"db_id": ..., "question_id": ..., "eval_run": "mini-vs-frontier"}.
    """
    tools = build_tools(db_path)
    model = ChatAnthropic(model=model_name, temperature=0, max_tokens=1024)
    agent = create_agent(model, tools, system_prompt=SYSTEM_PROMPT)

    langfuse_handler = CallbackHandler()
    config = {
        "recursion_limit": MAX_TOOL_CALLS_PER_QUERY * 2 + 2,
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_tags": ["text-to-sql-agent", model_name],
            **(trace_metadata or {}),
        },
    }

    t0 = time.monotonic()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)
    elapsed = time.monotonic() - t0

    messages = result["messages"]
    final = messages[-1].content
    if isinstance(final, list):
        final = "".join(part.get("text", "") for part in final if isinstance(part, dict))

    input_tokens = output_tokens = 0
    for m in messages:
        usage = getattr(m, "usage_metadata", None)
        if usage:
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)

    sql_calls = _extract_sql_calls(messages)

    return AgentResult(
        answer=final.strip(),
        sql_queries=sql_calls,
        tool_calls=len(sql_calls),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=elapsed,
        cost_usd=estimate_cost_usd(model_name, input_tokens, output_tokens),
    )
