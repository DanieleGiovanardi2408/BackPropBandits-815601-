"""FeatureAgent — secondo nodo del grafo multi-agent.

Responsabilità (dalla slide Reply):
    "Builds aggregated features per route from cleaned datasets"

Wrappa la classe FeatureBuilder (multiagent_pipeline.src.features) per
trasformarla in un nodo LangGraph che legge/scrive AgentState.

A differenza di DataAgent (che usa dataset_merged.csv), FeatureAgent
ha bisogno dei due dataset clean separati perche' la logica di
aggregazione li tratta diversamente. Carica direttamente da disco e
applica lo stesso perimetro presente in state.
"""
from __future__ import annotations

# ── Bootstrap per esecuzione diretta ─────────────────────────────────────────
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "multiagent_pipeline.agents"

import logging
from pathlib import Path

import pandas as pd

from multiagent_pipeline.state import AgentState, PATHS
from multiagent_pipeline.src.features import FeatureBuilder
from multiagent_pipeline.tools.data_tools import filter_by_perimeter

logger = logging.getLogger(__name__)

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
ALLARMI_PATH    = _PROJECT_ROOT / "data" / "processed" / "allarmi_clean.csv"
VIAGGIATORI_PATH = _PROJECT_ROOT / "data" / "processed" / "viaggiatori_clean.csv"


def run_feature_agent(
    state: AgentState,
    allarmi_path: Path | str = ALLARMI_PATH,
    viaggiatori_path: Path | str = VIAGGIATORI_PATH,
) -> AgentState:
    """Esegue il FeatureAgent: carica clean -> filtra perimetro -> aggrega feature.

    Args:
        state: stato corrente. Usa `state["perimeter"]` (opzionale).
        allarmi_path / viaggiatori_path: override per test.

    Returns:
        Nuovo AgentState con df_features e feature_meta popolati.
    """
    perimeter = state.get("perimeter") or {}
    logger.info("FeatureAgent start | perimeter=%s", perimeter)

    df_a = pd.read_csv(allarmi_path)
    df_v = pd.read_csv(viaggiatori_path)
    logger.info("Clean caricati: allarmi=%s viaggiatori=%s", df_a.shape, df_v.shape)

    df_a = filter_by_perimeter(df_a, perimeter)
    df_v = filter_by_perimeter(df_v, perimeter)
    logger.info("Dopo filtro: allarmi=%s viaggiatori=%s", df_a.shape, df_v.shape)

    builder = FeatureBuilder()
    df_features = builder.build(df_a, df_v)
    quality     = builder.quality_report(df_features)
    logger.info("Features: %d rotte x %d colonne", df_features.shape[0], df_features.shape[1])
    logger.info("Quality: %s", quality)

    feature_meta = {
        "n_rotte"      : int(df_features.shape[0]),
        "n_features"   : int(df_features.shape[1]),
        "feature_cols" : df_features.select_dtypes(include="number").columns.tolist(),
        "quality"      : quality,
    }

    return {
        **state,
        "df_features": df_features,
        "feature_meta": feature_meta,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    out = run_feature_agent({"perimeter": {"anno": 2024}})
    print("\n=== RISULTATO FeatureAgent ===")
    print("df_features shape:", out["df_features"].shape)
    print("n_features numeriche:", len(out["feature_meta"]["feature_cols"]))
    print("quality:", out["feature_meta"]["quality"])
