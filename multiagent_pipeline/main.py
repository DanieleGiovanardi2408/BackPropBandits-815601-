"""Orchestrator della pipeline multi-agent.

Step 1 del backend finale:
    DataAgent -> FeatureAgent -> BaselineAgent -> OutlierAgent -> (ReportAgent opzionale)
"""
from __future__ import annotations

import logging
import time
from typing import Any

from multiagent_pipeline.agents.data_agent import data_agent_node
from multiagent_pipeline.agents.feature_agent import run_feature_agent
from multiagent_pipeline.agents.baseline_agent import run_baseline_agent
from multiagent_pipeline.agents.outlier_agent import run_outlier_agent
from multiagent_pipeline.agents.report_agent import run_report_agent
from multiagent_pipeline.config import get_dry_run, get_use_llm
from multiagent_pipeline.state import AgentState

logger = logging.getLogger(__name__)


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


def _extract_error(state: AgentState, stage: str) -> str | None:
    meta_key = {
        "data": "data_meta",
        "feature": "feature_meta",
        "baseline": "baseline_meta",
        "outlier": "anomaly_meta",
        "report": "report",
    }[stage]
    meta = state.get(meta_key) or {}
    if isinstance(meta, dict):
        return meta.get("error")
    return None


def run_pipeline(
    perimeter: dict | None = None,
    *,
    run_report: bool = False,
    use_llm: bool | None = None,
    dry_run: bool | None = None,
    continue_on_error: bool = False,
    save_outputs: bool = False,
) -> tuple[AgentState, dict[str, Any]]:
    """Esegue la catena multi-agent in ordine.

    Args:
        perimeter: filtri utente.
        run_report: se True esegue anche ReportAgent (richiede API key LLM).
        continue_on_error: se False si ferma al primo errore.
        save_outputs: salva output su disco dove supportato.
    """
    use_llm_effective = get_use_llm(False) if use_llm is None else use_llm
    dry_run_effective = get_dry_run(False) if dry_run is None else dry_run
    state = _init_state(perimeter or {})
    stages: list[tuple[str, Any]] = [
        ("data", lambda s: data_agent_node(s, save_artifacts=save_outputs)),
        ("feature", lambda s: run_feature_agent(s, save_output=save_outputs)),
        ("baseline", lambda s: run_baseline_agent(s, save_output=save_outputs)),
        ("outlier", lambda s: run_outlier_agent(s, save_output=save_outputs)),
    ]
    if run_report:
        stages.append(
            (
                "report",
                lambda s: run_report_agent(
                    s,
                    save_output=save_outputs,
                    use_llm=use_llm_effective,
                    dry_run=dry_run_effective,
                ),
            )
        )

    stage_results: dict[str, dict[str, Any]] = {}
    step_errors: dict[str, str] = {}
    started_at = time.perf_counter()
    for stage_name, fn in stages:
        stage_start = time.perf_counter()
        logger.info("Orchestrator -> avvio stage: %s", stage_name)
        state = fn(state)
        err = _extract_error(state, stage_name)
        elapsed = round(time.perf_counter() - stage_start, 3)
        stage_results[stage_name] = {"ok": err is None, "error": err, "elapsed_s": elapsed}
        if err:
            step_errors[stage_name] = err
            logger.error("Orchestrator -> stage '%s' failed: %s", stage_name, err)
            if not continue_on_error:
                break

    report_path = state.get("report_path")
    summary = {
        "perimeter": state.get("perimeter"),
        "report_path": report_path,
        "stages": stage_results,
        "step_errors": step_errors,
        "completed_stages": [k for k, v in stage_results.items() if v["ok"]],
        "failed_stages": [k for k, v in stage_results.items() if not v["ok"]],
        "run_config": {
            "run_report": run_report,
            "use_llm": use_llm_effective,
            "dry_run": dry_run_effective,
            "continue_on_error": continue_on_error,
            "save_outputs": save_outputs,
        },
        "runtime_s": round(time.perf_counter() - started_at, 3),
    }
    return state, summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    state, summary = run_pipeline({"anno": 2024}, run_report=False, save_outputs=False)
    print("\n=== MULTIAGENT ORCHESTRATOR ===")
    print(summary)
    if state.get("df_anomalies") is not None:
        print("df_anomalies shape:", state["df_anomalies"].shape)
