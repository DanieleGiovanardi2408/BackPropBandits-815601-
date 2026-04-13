"""Multi-agent pipeline orchestrator (LangGraph).

Graph with conditional edges:

    START → DataAgent → [ok?] → FeatureAgent → [ok?] → BaselineAgent
                ↓ err                ↓ err                   ↓ err
               END                  END                     END
                                                             ↓ ok
                                                        OutlierAgent
                                                          ↓        ↓ err
                                                    [report?]     END
                                                     ↓    ↓
                                                   yes    no
                                                    ↓     ↓
                                              ReportAgent  END
                                                    ↓
                                                   END

If continue_on_error=True, the conditional edges do not stop the graph
on the first error: each agent handles internally the absence of data
from its predecessor.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.graph import StateGraph, END

from multiagent_pipeline.agents.data_agent import data_agent_node
from multiagent_pipeline.agents.feature_agent import run_feature_agent
from multiagent_pipeline.agents.baseline_agent import run_baseline_agent
from multiagent_pipeline.agents.outlier_agent import run_outlier_agent
from multiagent_pipeline.agents.report_agent import run_report_agent
from multiagent_pipeline.config import get_dry_run, get_use_llm
from multiagent_pipeline.state import AgentState

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _init_state(perimeter: dict) -> AgentState:
    return {
        "perimeter": perimeter or {},
        "df_raw": None,
        "df_allarmi": None,
        "df_viaggiatori": None,
        "data_meta": None,
        "df_features": None,
        "feature_meta": None,
        "df_baseline": None,
        "baseline_meta": None,
        "df_anomalies": None,
        "anomaly_meta": None,
        "report": None,
        "report_path": None,
    }


def _has_error(state: AgentState, meta_key: str) -> bool:
    """Checks whether a meta-dict contains an error."""
    meta = state.get(meta_key) or {}
    return isinstance(meta, dict) and bool(meta.get("error"))


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _build_graph(
    *,
    save_outputs: bool,
    run_report: bool,
    use_llm: bool,
    dry_run: bool,
    continue_on_error: bool,
) -> Any:
    """Builds and compiles the LangGraph graph.

    Nodes wrap the existing agent functions, extracting only the fields
    that each agent writes to the state (delta update, not full state).

    Conditional edges implement the stop-on-error logic
    and the skip of ReportAgent when not needed.
    """

    # ── Nodes ─────────────────────────────────────────────────────────────────
    # Each node returns ONLY the keys it writes, not the full state.
    # LangGraph performs the merge automatically.

    def node_data(state: AgentState) -> dict:
        result = data_agent_node(state, save_artifacts=save_outputs)
        return {
            "df_raw": result["df_raw"],
            "df_allarmi": result["df_allarmi"],
            "df_viaggiatori": result["df_viaggiatori"],
            "data_meta": result["data_meta"],
        }

    def node_feature(state: AgentState) -> dict:
        result = run_feature_agent(state, save_output=save_outputs)
        return {
            "df_features": result["df_features"],
            "feature_meta": result["feature_meta"],
        }

    def node_baseline(state: AgentState) -> dict:
        result = run_baseline_agent(state, save_output=save_outputs)
        return {
            "df_baseline": result["df_baseline"],
            "baseline_meta": result["baseline_meta"],
        }

    def node_outlier(state: AgentState) -> dict:
        result = run_outlier_agent(state, save_output=save_outputs)
        return {
            "df_anomalies": result["df_anomalies"],
            "anomaly_meta": result["anomaly_meta"],
        }

    def node_report(state: AgentState) -> dict:
        result = run_report_agent(
            state,
            save_output=save_outputs,
            use_llm=use_llm,
            dry_run=dry_run,
        )
        return {
            "report": result["report"],
            "report_path": result["report_path"],
        }

    # ── Conditional edges ────────────────────────────────────────────────────
    # If continue_on_error=True, always proceeds to the next node.
    # If False, stops at the first error.

    def after_data(state: AgentState) -> str:
        if not continue_on_error and _has_error(state, "data_meta"):
            return "end"
        return "feature"

    def after_feature(state: AgentState) -> str:
        if not continue_on_error and _has_error(state, "feature_meta"):
            return "end"
        return "baseline"

    def after_baseline(state: AgentState) -> str:
        if not continue_on_error and _has_error(state, "baseline_meta"):
            return "end"
        return "outlier"

    def after_outlier(state: AgentState) -> str:
        if not continue_on_error and _has_error(state, "anomaly_meta"):
            return "end"
        if not run_report:
            return "end"
        # Skip report if there are no anomalous routes to explain
        df = state.get("df_anomalies")
        if df is not None and hasattr(df, "columns") and "risk_label" in df.columns:
            if not df["risk_label"].isin(["ALTA", "MEDIA"]).any():
                logger.info("Orchestrator -> no ALTA/MEDIA routes, skipping ReportAgent")
                return "end"
        elif df is None and not continue_on_error:
            return "end"
        return "report"

    # ── Graph assembly ───────────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("data", node_data)
    graph.add_node("feature", node_feature)
    graph.add_node("baseline", node_baseline)
    graph.add_node("outlier", node_outlier)

    graph.set_entry_point("data")

    graph.add_conditional_edges(
        "data", after_data,
        {"feature": "feature", "end": END},
    )
    graph.add_conditional_edges(
        "feature", after_feature,
        {"baseline": "baseline", "end": END},
    )
    graph.add_conditional_edges(
        "baseline", after_baseline,
        {"outlier": "outlier", "end": END},
    )

    if run_report:
        graph.add_node("report", node_report)
        graph.add_conditional_edges(
            "outlier", after_outlier,
            {"report": "report", "end": END},
        )
        graph.add_edge("report", END)
    else:
        graph.add_edge("outlier", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY BUILDER
# ══════════════════════════════════════════════════════════════════════════════

_STAGE_META_KEYS = [
    ("data",     "data_meta"),
    ("feature",  "feature_meta"),
    ("baseline", "baseline_meta"),
    ("outlier",  "anomaly_meta"),
    ("report",   "report"),
]


def _build_summary(
    state: AgentState,
    started_at: float,
    run_config: dict,
) -> dict[str, Any]:
    """Builds the summary dict from the final state.

    Format identical to the previous version for compatibility with
    Streamlit and e2e tests.
    """
    stage_results: dict[str, dict[str, Any]] = {}
    step_errors: dict[str, str] = {}

    for stage_name, meta_key in _STAGE_META_KEYS:
        meta = state.get(meta_key)
        if meta is None:
            continue  # stage not executed
        err = meta.get("error") if isinstance(meta, dict) else None
        elapsed = meta.get("elapsed_s", 0) if isinstance(meta, dict) else 0
        stage_results[stage_name] = {
            "ok": err is None,
            "error": err,
            "elapsed_s": elapsed,
        }
        if err:
            step_errors[stage_name] = err

    return {
        "perimeter": state.get("perimeter"),
        "report_path": state.get("report_path"),
        "stages": stage_results,
        "step_errors": step_errors,
        "completed_stages": [k for k, v in stage_results.items() if v["ok"]],
        "failed_stages":    [k for k, v in stage_results.items() if not v["ok"]],
        "run_config": run_config,
        "runtime_s": round(time.perf_counter() - started_at, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    perimeter: dict | None = None,
    *,
    run_report: bool = False,
    use_llm: bool | None = None,
    dry_run: bool | None = None,
    continue_on_error: bool = False,
    save_outputs: bool = False,
) -> tuple[AgentState, dict[str, Any]]:
    """Runs the multi-agent pipeline as a LangGraph graph.

    API identical to the previous version for compatibility with
    Streamlit, script runner and e2e tests.

    Args:
        perimeter: user filters (anno, aeroporto, paese, zona).
        run_report: if True, also runs ReportAgent.
        use_llm: enables LLM calls in ReportAgent.
        dry_run: generates placeholder explanations without API calls.
        continue_on_error: if True, continues even after errors.
        save_outputs: saves CSV/JSON artefacts to disk.

    Returns:
        (final_state, summary) — same structure as the previous version.
    """
    use_llm_effective = get_use_llm(False) if use_llm is None else use_llm
    dry_run_effective = get_dry_run(False) if dry_run is None else dry_run

    run_config = {
        "run_report": run_report,
        "use_llm": use_llm_effective,
        "dry_run": dry_run_effective,
        "continue_on_error": continue_on_error,
        "save_outputs": save_outputs,
    }

    graph = _build_graph(
        save_outputs=save_outputs,
        run_report=run_report,
        use_llm=use_llm_effective,
        dry_run=dry_run_effective,
        continue_on_error=continue_on_error,
    )

    initial_state = _init_state(perimeter or {})
    started_at = time.perf_counter()

    logger.info(
        "Orchestrator LangGraph -> starting pipeline | perimeter=%s | config=%s",
        perimeter, run_config,
    )

    final_state = graph.invoke(initial_state)

    summary = _build_summary(final_state, started_at, run_config)

    logger.info(
        "Orchestrator LangGraph -> completed in %.2fs | stages=%s",
        summary["runtime_s"],
        {k: v["ok"] for k, v in summary["stages"].items()},
    )

    return final_state, summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    state, summary = run_pipeline(
        {"anno": 2024},
        run_report=False,
        save_outputs=False,
    )
    print("\n=== MULTIAGENT ORCHESTRATOR (LangGraph) ===")
    print(summary)
    if state.get("df_anomalies") is not None:
        print("df_anomalies shape:", state["df_anomalies"].shape)
