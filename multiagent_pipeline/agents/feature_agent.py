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

# Artefatti scritti dal DataAgent (handoff cross-process).
DATA_AGENT_OUTPUT_JSON = _PROJECT_ROOT / "data" / "processed" / "data_agent_output.json"


def _load_from_data_agent_artifact() -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """Tenta di caricare allarmi/viaggiatori filtrati e perimetro dall'output del DataAgent.

    Restituisce (df_allarmi, df_viaggiatori, perimeter) oppure None se l'artefatto
    non esiste o e' incompleto.
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
        logger.warning("Impossibile leggere artefatto DataAgent: %s", e)
        return None


def run_feature_agent(
    state: AgentState,
    allarmi_path: Path | str = ALLARMI_PATH,
    viaggiatori_path: Path | str = VIAGGIATORI_PATH,
    save_output: bool = False,
    output_path: Path | str | None = None,
) -> AgentState:
    """Esegue il FeatureAgent: carica clean -> filtra perimetro -> aggrega feature.

    Args:
        state: stato corrente. Usa `state["perimeter"]` (opzionale).
        allarmi_path / viaggiatori_path: override per test.

    Args:
        save_output: se True salva il DataFrame finale su disco.
        output_path: path CSV di output. Se None usa PATHS["features"].

    Returns:
        Nuovo AgentState con df_features e feature_meta popolati.
    """
    logger.info("FeatureAgent ── Avvio")
    started_at = time.perf_counter()
    perimeter = state.get("perimeter") or {}
    logger.info("FeatureAgent start | perimeter=%s", perimeter)

    try:
        # Priorità 1: dataframe gia' nello stato (catena in-process con DataAgent).
        df_a = state.get("df_allarmi")
        df_v = state.get("df_viaggiatori")

        if isinstance(df_a, pd.DataFrame) and isinstance(df_v, pd.DataFrame):
            logger.info("Input ricevuti da DataAgent (state): allarmi=%s viaggiatori=%s", df_a.shape, df_v.shape)
        else:
            # Priorità 2: artefatto su disco scritto dal DataAgent (handoff cross-process).
            artifact = _load_from_data_agent_artifact()
            if artifact is not None:
                df_a, df_v, da_perimeter = artifact
                logger.info(
                    "Input letti dall'artefatto DataAgent: allarmi=%s viaggiatori=%s | perimetro=%s",
                    df_a.shape, df_v.shape, da_perimeter,
                )
                # Allinea il perimetro a quello effettivamente applicato dal DataAgent
                if not perimeter:
                    perimeter = da_perimeter
            else:
                # Priorità 3: fallback completo — clean originali + filtro locale.
                df_a = pd.read_csv(allarmi_path)
                df_v = pd.read_csv(viaggiatori_path)
                logger.info("Clean caricati da disco (fallback): allarmi=%s viaggiatori=%s", df_a.shape, df_v.shape)
                df_a = filter_by_perimeter(df_a, perimeter)
                df_v = filter_by_perimeter(df_v, perimeter)
                logger.info("Dopo filtro locale: allarmi=%s viaggiatori=%s", df_a.shape, df_v.shape)

        if df_a.empty and df_v.empty:
            raise ValueError(f"Nessun dato trovato con i filtri: {perimeter}")

        builder = FeatureBuilder()
        df_features = builder.build(df_a, df_v)

        if df_features.empty:
            raise ValueError(f"Nessuna feature generata con i filtri: {perimeter}")

        quality = builder.quality_report(df_features)
        logger.info("Features: %d rotte x %d colonne", df_features.shape[0], df_features.shape[1])
        logger.info("Quality: %s", quality)

        saved_to = None
        if save_output:
            default_out = _PROJECT_ROOT / PATHS["features"]
            out_path = Path(output_path) if output_path is not None else default_out
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_features.to_csv(out_path, index=False)
            saved_to = str(out_path)
            logger.info("FeatureAgent output salvato in: %s", saved_to)

        feature_meta = {
            "n_rotte": int(df_features.shape[0]),
            "n_features": int(df_features.shape[1]),
            "feature_cols": df_features.select_dtypes(include="number").columns.tolist(),
            "quality": quality,
            "saved_to": saved_to,
            "elapsed_s": round(time.perf_counter() - started_at, 3),
        }

        logger.info("FeatureAgent ✓ Completato")
        return {
            **state,
            "df_features": df_features,
            "feature_meta": feature_meta,
        }
    except Exception as e:
        logger.error("FeatureAgent ✗ Errore: %s", e)
        return {
            **state,
            "df_features": None,
            "feature_meta": {
                "error": str(e),
                "user_message": "Feature extraction non riuscita: controlla filtri e dataset di input.",
                "elapsed_s": round(time.perf_counter() - started_at, 3),
            },
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
