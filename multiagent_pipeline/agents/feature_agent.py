"""FeatureAgent — second node of the multi-agent graph.

Responsibilities (from the Reply slide):
    "Builds aggregated features per route from cleaned datasets"

Wraps the FeatureBuilder class (multiagent_pipeline.src.features) to
transform it into a LangGraph node that reads/writes AgentState.

Unlike DataAgent (which uses dataset_merged.csv), FeatureAgent
needs the two separate clean datasets because the aggregation logic
treats them differently. It loads directly from disk and applies
the same perimeter present in the state.
"""
from __future__ import annotations

# ── Bootstrap for direct execution ───────────────────────────────────────────
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "multiagent_pipeline.agents"

import json
import logging
import time
from pathlib import Path

import pandas as pd

from multiagent_pipeline.state import AgentState, PATHS
from multiagent_pipeline.src.features import FeatureBuilder
from multiagent_pipeline.tools.data_tools import filter_by_perimeter

logger = logging.getLogger(__name__)

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
ALLARMI_PATH    = _PROJECT_ROOT / "data" / "processed" / "allarmi_clean.csv"
VIAGGIATORI_PATH = _PROJECT_ROOT / "data" / "processed" / "viaggiatori_clean.csv"

# Artefacts written by DataAgent (cross-process handoff).
DATA_AGENT_OUTPUT_JSON = _PROJECT_ROOT / "data" / "processed" / "data_agent_output.json"


def _load_from_data_agent_artifact() -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """Attempts to load filtered allarmi/viaggiatori and perimeter from the DataAgent output.

    Returns (df_allarmi, df_viaggiatori, perimeter) or None if the artefact
    does not exist or is incomplete.
    """
    if not DATA_AGENT_OUTPUT_JSON.exists():
        return None
    try:
        manifest = json.loads(DATA_AGENT_OUTPUT_JSON.read_text())
        outputs  = manifest.get("outputs", {})
        a_rel    = outputs.get("allarmi")
        v_rel    = outputs.get("viaggiatori")
        if not (a_rel and v_rel):
            return None
        a_path = _PROJECT_ROOT / a_rel
        v_path = _PROJECT_ROOT / v_rel
        if not (a_path.exists() and v_path.exists()):
            return None
        df_a = pd.read_csv(a_path)
        df_v = pd.read_csv(v_path)
        return df_a, df_v, manifest.get("perimeter", {})
    except Exception as e:
        logger.warning("Unable to read DataAgent artefact: %s", e)
        return None


def run_feature_agent(
    state: AgentState,
    allarmi_path: Path | str = ALLARMI_PATH,
    viaggiatori_path: Path | str = VIAGGIATORI_PATH,
    save_output: bool = False,
    output_path: Path | str | None = None,
) -> AgentState:
    """Runs the FeatureAgent: load clean -> filter perimeter -> aggregate features.

    Args:
        state: current state. Uses `state["perimeter"]` (optional).
        allarmi_path / viaggiatori_path: override for testing.

    Args:
        save_output: if True, saves the final DataFrame to disk.
        output_path: CSV output path. If None, uses PATHS["features"].

    Returns:
        New AgentState with df_features and feature_meta populated.
    """
    logger.info("FeatureAgent ── Starting")
    started_at = time.perf_counter()
    perimeter = state.get("perimeter") or {}
    logger.info("FeatureAgent start | perimeter=%s", perimeter)

    try:
        # Priority 1: dataframes already in state (in-process chain with DataAgent).
        df_a = state.get("df_allarmi")
        df_v = state.get("df_viaggiatori")

        if isinstance(df_a, pd.DataFrame) and isinstance(df_v, pd.DataFrame):
            logger.info("Inputs received from DataAgent (state): allarmi=%s viaggiatori=%s", df_a.shape, df_v.shape)
        else:
            # Priority 2: artefact on disk written by DataAgent (cross-process handoff).
            artifact = _load_from_data_agent_artifact()
            if artifact is not None:
                df_a, df_v, da_perimeter = artifact
                logger.info(
                    "Inputs read from DataAgent artefact: allarmi=%s viaggiatori=%s | perimeter=%s",
                    df_a.shape, df_v.shape, da_perimeter,
                )
                # Align perimeter to the one actually applied by DataAgent
                if not perimeter:
                    perimeter = da_perimeter
            else:
                # Priority 3: full fallback — original clean files + local filter.
                df_a = pd.read_csv(allarmi_path)
                df_v = pd.read_csv(viaggiatori_path)
                logger.info("Clean files loaded from disk (fallback): allarmi=%s viaggiatori=%s", df_a.shape, df_v.shape)
                df_a = filter_by_perimeter(df_a, perimeter)
                df_v = filter_by_perimeter(df_v, perimeter)
                logger.info("After local filter: allarmi=%s viaggiatori=%s", df_a.shape, df_v.shape)

        if df_a.empty and df_v.empty:
            raise ValueError(f"No data found with filters: {perimeter}")

        builder = FeatureBuilder()
        df_features = builder.build(df_a, df_v)

        if df_features.empty:
            raise ValueError(f"No features generated with filters: {perimeter}")

        quality = builder.quality_report(df_features)
        logger.info("Features: %d routes x %d columns", df_features.shape[0], df_features.shape[1])
        logger.info("Quality: %s", quality)

        saved_to = None
        if save_output:
            default_out = _PROJECT_ROOT / PATHS["features"]
            out_path = Path(output_path) if output_path is not None else default_out
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_features.to_csv(out_path, index=False)
            saved_to = str(out_path)
            logger.info("FeatureAgent output saved to: %s", saved_to)

        feature_meta = {
            "n_rotte": int(df_features.shape[0]),
            "n_features": int(df_features.shape[1]),
            "feature_cols": df_features.select_dtypes(include="number").columns.tolist(),
            "quality": quality,
            "saved_to": saved_to,
            "elapsed_s": round(time.perf_counter() - started_at, 3),
        }

        logger.info("FeatureAgent ✓ Completed")
        return {
            **state,
            "df_features": df_features,
            "feature_meta": feature_meta,
        }
    except Exception as e:
        logger.error("FeatureAgent ✗ Error: %s", e)
        return {
            **state,
            "df_features": None,
            "feature_meta": {
                "error": str(e),
                "user_message": "Feature extraction failed: check filters and input datasets.",
                "elapsed_s": round(time.perf_counter() - started_at, 3),
            },
        }


if __name__ == "__main__":
    from multiagent_pipeline.tools.data_tools import load_last_perimeter
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    _perimeter = load_last_perimeter() or {"anno": 2024}
    print(f"  Perimeter: {_perimeter}")
    out = run_feature_agent({"perimeter": _perimeter})
    print("\n=== FeatureAgent RESULT ===")
    print("df_features shape:", out["df_features"].shape)
    print("n_numeric_features:", len(out["feature_meta"]["feature_cols"]))
    print("quality:", out["feature_meta"]["quality"])
