"""
Eval dashboard: reads the JSON output of eval/run_bird_eval.py,
eval/audit_taxonomy.py, and eval/judge_calibration.py, and renders them.
This is a read-only viewer over already-computed results, not a place that
re-runs the eval - keeps the dashboard free of API keys and fast to load.

Run: streamlit run dashboard/app.py
"""
import json
import os

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "eval")

st.set_page_config(page_title="Text-to-SQL Eval Dashboard", page_icon="📊", layout="wide")


def load_json(filename):
    path = os.path.join(EVAL_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_run_files():
    if not os.path.isdir(EVAL_DIR):
        return []
    return sorted(
        f for f in os.listdir(EVAL_DIR)
        if f.startswith("results") and f.endswith(".json")
    )


st.title("Text-to-SQL Evaluation Harness")
st.caption("BIRD Mini-Dev execution accuracy, benchmark audit, and judge calibration")

tab_results, tab_audit, tab_judge = st.tabs(
    ["Execution Accuracy", "Benchmark Audit", "Judge Calibration"]
)

# ── Execution Accuracy ────────────────────────────────────────────────────────
with tab_results:
    run_files = find_run_files()
    if not run_files:
        st.info("No eval run found yet. Run `python -m eval.run_bird_eval --n 500 --model mini` first.")
    else:
        selected = st.selectbox("Eval run", run_files, index=len(run_files) - 1)
        data = load_json(selected)

        summary = data["summary"]
        cols = st.columns(len(summary))
        for col, (model_alias, s) in zip(cols, summary.items()):
            with col:
                st.metric(f"{model_alias} - Execution Accuracy", f"{s['execution_accuracy']*100:.1f}%",
                          help=f"{s['correct']}/{s['n']} correct")
                st.metric("Avg. cost / query", f"${s['avg_cost_usd']:.4f}" if s['avg_cost_usd'] else "n/a")
                st.metric("p50 latency", f"{s['p50_latency_seconds']:.1f}s")

        st.subheader("By difficulty")
        for model_alias, s in summary.items():
            st.write(f"**{model_alias}**")
            diff_df = pd.DataFrame([
                {"difficulty": d, "n": v["n"], "accuracy": f"{v['accuracy']*100:.1f}%"}
                for d, v in s["by_difficulty"].items()
            ])
            st.dataframe(diff_df, hide_index=True, use_container_width=True)

        st.subheader("Per-question results")
        results_df = pd.DataFrame(data["results"])
        model_filter = st.multiselect("Model", results_df["model"].unique().tolist(),
                                        default=results_df["model"].unique().tolist())
        show_only_misses = st.checkbox("Show only misses", value=False)
        filtered = results_df[results_df["model"].isin(model_filter)]
        if show_only_misses:
            filtered = filtered[~filtered["correct"]]
        st.dataframe(
            filtered[["question_id", "db_id", "difficulty", "model", "correct",
                      "cost_usd", "elapsed_seconds", "error"]],
            hide_index=True, use_container_width=True,
        )

        with st.expander("Inspect one question"):
            qid = st.number_input("question_id", min_value=0, step=1)
            match = results_df[results_df["question_id"] == qid]
            if not match.empty:
                row = match.iloc[0]
                st.code(row["gold_sql"], language="sql")
                st.caption("Gold SQL")
                st.code(row["agent_sql"] or "(no SQL executed)", language="sql")
                st.caption("Agent's last SQL")
                st.write("**Agent's final answer:**", row["agent_answer"])

# ── Benchmark Audit ────────────────────────────────────────────────────────────
with tab_audit:
    audit = load_json("audit_results.json")
    if not audit:
        st.info("No audit found yet. Run `python -m eval.audit_taxonomy --n 100` first.")
    else:
        s = audit["summary"]
        st.metric("Annotation error rate", f"{s['error_rate']*100:.1f}%",
                   help=f"{s['n_erroneous']}/{s['n']} examples flagged by the LLM auditor")
        st.caption(
            "Methodology: an LLM auditor (not a human) applies the exact four-way taxonomy "
            "from Jin, Choi, Zhu & Kang, \"Text-to-SQL Benchmarks are Broken\" (CIDR 2026). "
            "This approximates the paper's method; it does not replicate it. Full explanation "
            "in eval/audit_taxonomy.py's module docstring."
        )

        if s["by_pattern"]:
            pattern_df = pd.DataFrame(
                [{"pattern": k, "count": v} for k, v in s["by_pattern"].items()]
            ).sort_values("count", ascending=False)
            st.bar_chart(pattern_df.set_index("pattern"))

        st.subheader("Flagged examples")
        audit_df = pd.DataFrame(audit["results"])
        flagged = audit_df[~audit_df["verdict"].isin(["VALID", "PARSE_ERROR"])]
        st.dataframe(
            flagged[["question_id", "db_id", "difficulty", "verdict", "explanation"]],
            hide_index=True, use_container_width=True,
        )

# ── Judge Calibration ───────────────────────────────────────────────────────────
with tab_judge:
    judge = load_json("judge_calibration_results.json")
    if not judge:
        st.info("No judge calibration found yet. Run `python -m eval.judge_calibration "
                 "--results eval/results_mini_full.json --n 100` first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cohen's kappa (judge vs. execution ground truth)", f"{judge['kappa']:.3f}")
        with col2:
            st.metric("Raw agreement", f"{judge['raw_agreement']*100:.1f}%")

        kappa = judge["kappa"]
        if kappa > 0.8:
            st.success("kappa > 0.8: production-ready agreement")
        elif kappa > 0.6:
            st.warning("kappa > 0.6: acceptable, but worth tightening the judge rubric")
        else:
            st.error("kappa < 0.6: judge disagrees with ground truth too often to trust unsupervised")

        st.subheader("Disagreements")
        details_df = pd.DataFrame(judge["details"])
        disagreements = details_df[~details_df["agree"]]
        st.dataframe(disagreements, hide_index=True, use_container_width=True)
