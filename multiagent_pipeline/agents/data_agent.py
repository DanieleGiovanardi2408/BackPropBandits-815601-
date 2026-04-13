"""
data_agent.py
─────────────
Agent 1 — DataAgent

Responsibilities:
    Loads the already-cleaned merged dataset from preprocessing, applies the
    user-defined filters (year, airport, country, zone) and returns the
    filtered DataFrame with its statistics in the shared state.

Architecture:
    DETERMINISTIC agent — does not use an LLM.
    Always executes the same 3 tools in sequence. This is a deliberate
    choice: the DataAgent knows exactly what to do; an LLM would be a
    waste of resources and latency.

    load_dataset → filter_by_perimeter → get_dataset_stats

Input  (from AgentState): state["perimeter"]
Output (to AgentState):   state["df_raw"], state["df_allarmi"],
                          state["df_viaggiatori"], state["data_meta"]
"""

# ── Bootstrap for direct execution (python data_agent.py) ────────────────────
# Allows running the file both as a module (-m) and as a script (▶ VSCode).
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "multiagent_pipeline.agents"

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Optional

from multiagent_pipeline.state import AgentState, Perimeter, PATHS

logger = logging.getLogger(__name__)

# Resolve dataset paths relative to the project root, so it works
# from any cwd (terminal, VSCode "Run File", debugger, etc.)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS = {k: str(_PROJECT_ROOT / v) if not Path(v).is_absolute() else v
         for k, v in PATHS.items()}

# DataAgent outputs (artefacts for audit + cross-process handoff to FeatureAgent)
DATA_AGENT_OUTPUT_JSON       = _PROJECT_ROOT / "data" / "processed" / "data_agent_output.json"
DATA_AGENT_OUTPUT_CSV        = _PROJECT_ROOT / "data" / "processed" / "data_agent_filtered.csv"
DATA_AGENT_ALLARMI_CSV       = _PROJECT_ROOT / "data" / "processed" / "data_agent_allarmi.csv"
DATA_AGENT_VIAGGIATORI_CSV   = _PROJECT_ROOT / "data" / "processed" / "data_agent_viaggiatori.csv"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — load_dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str) -> str:
    """
    Loads the airport transit dataset from a CSV file.
    Returns the data in JSON format (orient=records).
    Always use as the first step before applying filters.
    """
    try:
        p = Path(path)
        if not p.exists():
            return json.dumps({"error": f"File not found: {path}"})

        df = pd.read_csv(p)
        logger.info("[load_dataset] Loaded '%s': %d rows × %d columns", p.name, df.shape[0], df.shape[1])
        return df.to_json(orient="records", date_format="iso")

    except Exception as e:
        return json.dumps({"error": f"Error during loading: {str(e)}"})


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — filter_by_perimeter
# ══════════════════════════════════════════════════════════════════════════════

def filter_by_perimeter(
    data_json: str,
    anno: Optional[int] = None,
    aeroporto_arrivo: Optional[str] = None,
    aeroporto_partenza: Optional[str] = None,
    paese_partenza: Optional[str] = None,
    zona: Optional[int] = None,
) -> str:
    """
    Filters the dataset by the user-defined analysis perimeter.
    Only applies filters whose parameters are not None.
    All string comparisons are case-insensitive.
    Returns the filtered dataset in JSON format (orient=records).
    """
    try:
        parsed = json.loads(data_json)
        if isinstance(parsed, dict) and "error" in parsed:
            return data_json  # propagate previous error

        df = pd.DataFrame(parsed)
        applied_filters = []

        if anno is not None:
            df = df[df["ANNO_PARTENZA"] == anno]
            applied_filters.append(f"anno={anno}")

        if aeroporto_arrivo is not None:
            df = df[df["AREOPORTO_ARRIVO"].str.upper() == aeroporto_arrivo.upper()]
            applied_filters.append(f"aeroporto_arrivo={aeroporto_arrivo}")

        if aeroporto_partenza is not None:
            df = df[df["AREOPORTO_PARTENZA"].str.upper() == aeroporto_partenza.upper()]
            applied_filters.append(f"aeroporto_partenza={aeroporto_partenza}")

        if paese_partenza is not None:
            df = df[df["PAESE_PART"].str.upper() == paese_partenza.upper()]
            applied_filters.append(f"paese_partenza={paese_partenza}")

        if zona is not None:
            df = df[df["ZONA"] == zona]
            applied_filters.append(f"zona={zona}")

        if df.empty:
            return json.dumps({
                "error": f"No data found with filters: {', '.join(applied_filters)}"
            })

        label = ', '.join(applied_filters) if applied_filters else "none"
        logger.info("[filter_by_perimeter] Applied filters: %s -> %d rows remaining", label, len(df))
        return df.to_json(orient="records", date_format="iso")

    except Exception as e:
        return json.dumps({"error": f"Error during filtering: {str(e)}"})


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — get_dataset_stats
# ══════════════════════════════════════════════════════════════════════════════

def get_dataset_stats(data_json: str) -> str:
    """
    Computes descriptive statistics on the filtered dataset.
    Returns n_righe, n_rotte_uniche, anni_presenti,
    paesi_partenza_top5 and n_con_allarmi (rows with TOT > 0).
    Use after filter_by_perimeter to get an overview of the data.
    """
    try:
        parsed = json.loads(data_json)
        if isinstance(parsed, dict) and "error" in parsed:
            return data_json  # propagate previous error

        df = pd.DataFrame(parsed)

        # Build ROTTA column if not present
        if "ROTTA" not in df.columns:
            df["ROTTA"] = (
                df["AREOPORTO_PARTENZA"].str.upper() + "-" +
                df["AREOPORTO_ARRIVO"].str.upper()
            )

        # Numeric TOT to count alarms
        tot = pd.to_numeric(df["TOT"], errors="coerce").fillna(0)

        stats = {
            "n_righe"            : int(len(df)),
            "n_rotte_uniche"     : int(df["ROTTA"].nunique()),
            "anni_presenti"      : sorted(df["ANNO_PARTENZA"].dropna().unique().tolist()),
            "paesi_partenza_top5": df["PAESE_PART"].value_counts().head(5).index.tolist(),
            "n_con_allarmi"      : int((tot > 0).sum()),
        }

        logger.info(
            "[get_dataset_stats] %d rows, %d routes, %d with alarms",
            stats["n_righe"],
            stats["n_rotte_uniche"],
            stats["n_con_allarmi"],
        )
        return json.dumps(stats)

    except Exception as e:
        return json.dumps({"error": f"Error computing statistics: {str(e)}"})


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH NODE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def data_agent_node(state: AgentState, save_artifacts: bool = False) -> AgentState:
    """
    LangGraph node for the DataAgent.

    Reads state["perimeter"], executes the tools in sequence and writes:
      - state["df_raw"]          (filtered dataset_merged)
      - state["df_allarmi"]      (filtered allarmi_clean)
      - state["df_viaggiatori"]  (filtered viaggiatori_clean)
      - state["data_meta"]       (main statistics)

    Args:
        save_artifacts: if True, saves debug JSON/CSV files to data/processed.

    On error, does not raise exceptions: populates data_meta["error"]
    and sets df_raw = None, so the graph can handle the failure.
    """
    logger.info("DataAgent -- Starting")

    try:
        # 1. Read and validate the perimeter
        raw_perimeter = state.get("perimeter", {})
        perimeter = Perimeter(**raw_perimeter)

        # 2. Load the datasets (merged + separate clean files)
        merged_json = load_dataset(PATHS["dataset_merged"])
        allarmi_json = load_dataset(PATHS["allarmi_clean"])
        viaggiatori_json = load_dataset(PATHS["viaggiatori_clean"])

        # 3. Apply filters to all datasets
        merged_json = filter_by_perimeter(
            data_json=merged_json,
            anno=perimeter.anno,
            aeroporto_arrivo=perimeter.aeroporto_arrivo,
            aeroporto_partenza=perimeter.aeroporto_partenza,
            paese_partenza=perimeter.paese_partenza,
            zona=perimeter.zona,
        )
        allarmi_json = filter_by_perimeter(
            data_json=allarmi_json,
            anno=perimeter.anno,
            aeroporto_arrivo=perimeter.aeroporto_arrivo,
            aeroporto_partenza=perimeter.aeroporto_partenza,
            paese_partenza=perimeter.paese_partenza,
            zona=perimeter.zona,
        )
        viaggiatori_json = filter_by_perimeter(
            data_json=viaggiatori_json,
            anno=perimeter.anno,
            aeroporto_arrivo=perimeter.aeroporto_arrivo,
            aeroporto_partenza=perimeter.aeroporto_partenza,
            paese_partenza=perimeter.paese_partenza,
            zona=perimeter.zona,
        )

        # 4. Compute statistics
        stats_json = get_dataset_stats(merged_json)

        # 5. Check propagated errors
        stats = json.loads(stats_json)
        if "error" in stats:
            raise ValueError(stats["error"])

        for payload in [allarmi_json, viaggiatori_json]:
            parsed = json.loads(payload)
            if isinstance(parsed, dict) and "error" in parsed:
                raise ValueError(parsed["error"])

        # 6. Deserialize DataFrames
        df_raw = pd.DataFrame(json.loads(merged_json))
        df_allarmi = pd.DataFrame(json.loads(allarmi_json))
        df_viaggiatori = pd.DataFrame(json.loads(viaggiatori_json))
        stats["n_righe_allarmi"] = int(len(df_allarmi))
        stats["n_righe_viaggiatori"] = int(len(df_viaggiatori))

        # 7. Save artefacts to disk (audit + cross-process handoff)
        if save_artifacts:
            DATA_AGENT_OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
            df_raw.to_csv(DATA_AGENT_OUTPUT_CSV, index=False)
            df_allarmi.to_csv(DATA_AGENT_ALLARMI_CSV, index=False)
            df_viaggiatori.to_csv(DATA_AGENT_VIAGGIATORI_CSV, index=False)
            artifact = {
                "perimeter": perimeter.model_dump(),
                "data_meta": stats,
                "outputs": {
                    "merged":      str(DATA_AGENT_OUTPUT_CSV.relative_to(_PROJECT_ROOT)),
                    "allarmi":     str(DATA_AGENT_ALLARMI_CSV.relative_to(_PROJECT_ROOT)),
                    "viaggiatori": str(DATA_AGENT_VIAGGIATORI_CSV.relative_to(_PROJECT_ROOT)),
                },
            }
            DATA_AGENT_OUTPUT_JSON.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
            logger.info("[save] %s + 3 csv (merged/allarmi/viaggiatori)", DATA_AGENT_OUTPUT_JSON.name)

        logger.info(
            "DataAgent ✓ Completed — %d rows, %d unique routes",
            stats["n_righe"],
            stats["n_rotte_uniche"],
        )

        return {
            **state,
            "df_raw"   : df_raw,
            "df_allarmi": df_allarmi,
            "df_viaggiatori": df_viaggiatori,
            "data_meta": stats,
        }

    except Exception as e:
        logger.error("DataAgent ✗ Error: %s", e)
        return {
            **state,
            "df_raw"   : None,
            "df_allarmi": None,
            "df_viaggiatori": None,
            "data_meta": {"error": str(e)},
        }


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

def _pick_value(df, col: str, label: str, cast=str, top: int = 30):
    """Shows unique values for the column and lets the user pick one.

    The user can type the number (from the list) or enter the value directly.
    """
    if col not in df.columns:
        print(f"  ⚠ Column {col} not found, skipping.")
        return None

    values = (
        df[col].dropna().value_counts().head(top).index.tolist()
    )
    if not values:
        print(f"  ⚠ No values available for {col}.")
        return None

    print(f"\n  ── Available values for {label} (top {len(values)}) ──")
    for i, v in enumerate(values, 1):
        n = int((df[col] == v).sum())
        print(f"    {i:>3}. {v}  ({n} rows)")
    raw = input(f"  Choose (number or exact value): ").strip()
    if not raw:
        return None

    # Number from the list?
    if raw.isdigit() and 1 <= int(raw) <= len(values):
        return cast(values[int(raw) - 1])

    # Value typed manually
    try:
        return cast(raw)
    except Exception as e:
        print(f"  ⚠ invalid value: {e}")
        return None


def _interactive_perimeter() -> dict:
    """Interactive CLI: shows available filters and actual dataset values."""
    # Load the dataset once to populate the menus
    try:
        df_preview = pd.read_csv(PATHS["dataset_merged"])
    except Exception as e:
        print(f"  ⚠ Unable to load dataset for preview: {e}")
        df_preview = None

    # (key, label, csv_column, cast)
    fields = [
        ("anno",               "Year",                    "ANNO_PARTENZA",      int),
        ("aeroporto_partenza", "Departure airport",       "AREOPORTO_PARTENZA", str),
        ("aeroporto_arrivo",   "Arrival airport",         "AREOPORTO_ARRIVO",   str),
        ("paese_partenza",     "Departure country",       "PAESE_PART",         str),
        ("zona",               "Geographic zone",         "ZONA",               int),
    ]

    print("\n── Available filters (perimeter) ────────────────────")
    for i, (key, label, *_) in enumerate(fields, 1):
        print(f"  {i}. {key:20s} — {label}")
    print("──────────────────────────────────────────────────────")

    raw = input("Which filters do you want to apply? (numbers or names separated by comma, empty = none): ").strip()
    if not raw:
        return {}

    keys = [k for k, *_ in fields]
    idx = []
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok.isdigit():
            i = int(tok) - 1
            if 0 <= i < len(fields):
                idx.append(i)
        elif tok in keys:
            idx.append(keys.index(tok))
        else:
            print(f"  ⚠ ignored: '{tok}' (not a number or a valid name)")

    perimeter = {}
    for i in idx:
        if i < 0 or i >= len(fields):
            continue
        key, label, col, cast = fields[i]
        if df_preview is not None:
            v = _pick_value(df_preview, col, label, cast=cast)
        else:
            raw_val = input(f"  {label} = ").strip()
            v = cast(raw_val) if raw_val else None
        if v is not None:
            perimeter[key] = v
    return perimeter


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print("=" * 55)
    print("  DataAgent — interactive mode")
    print("=" * 55)

    perimeter = _interactive_perimeter()
    print(f"\n  Selected perimeter: {perimeter or '(none)'}")

    stato_iniziale: AgentState = {
        "perimeter"    : perimeter,
        "df_raw"       : None,
        "df_allarmi"   : None,
        "df_viaggiatori": None,
        "data_meta"    : None,
        "df_features"  : None,
        "feature_meta" : None,
        "df_baseline"  : None,
        "baseline_meta": None,
        "df_anomalies" : None,
        "anomaly_meta" : None,
        "report"       : None,
        "report_path"  : None,
    }

    stato_finale = data_agent_node(stato_iniziale, save_artifacts=True)

    print("── Result ─────────────────────────────────────────")
    if stato_finale["data_meta"].get("error"):
        print(f"  ERROR: {stato_finale['data_meta']['error']}")
        print("=" * 55)
    else:
        meta = stato_finale["data_meta"]
        print(f"  Rows loaded:       {meta['n_righe']}")
        print(f"  Unique routes:     {meta['n_rotte_uniche']}")
        print(f"  Years present:     {meta['anni_presenti']}")
        print(f"  Top 5 countries:   {meta['paesi_partenza_top5']}")
        print(f"  Rows with alarms:  {meta['n_con_allarmi']}")
        print(f"  df_raw shape:      {stato_finale['df_raw'].shape}")
        print(f"  df_allarmi shape:  {stato_finale['df_allarmi'].shape}")
        print(f"  df_viag shape:     {stato_finale['df_viaggiatori'].shape}")
        print("=" * 55)

        # ── Interactive chain ──────────────────────────────────
        CHAIN = [
            ("FeatureAgent",  "run_feature_agent",  "multiagent_pipeline.agents.feature_agent"),
            ("BaselineAgent", "run_baseline_agent", "multiagent_pipeline.agents.baseline_agent"),
            ("OutlierAgent",  "run_outlier_agent",  "multiagent_pipeline.agents.outlier_agent"),
            ("ReportAgent",   "run_report_agent",   "multiagent_pipeline.agents.report_agent"),
        ]

        state = stato_finale
        for agent_name, fn_name, module_path in CHAIN:
            risposta = input(f"\nDo you want to run {agent_name}? [s/N] ").strip().lower()
            if risposta not in ("s", "si", "sì", "y", "yes"):
                print(f"  ↳ {agent_name} skipped. Done.")
                break
            import importlib
            mod = importlib.import_module(module_path)
            fn  = getattr(mod, fn_name)
            print(f"\n── {agent_name} ──────────────────────────────────────")
            state = fn(state)
            # Print a brief result for each agent
            if agent_name == "FeatureAgent":
                fm = state.get("feature_meta") or {}
                if "error" in fm:
                    print(f"  ERROR: {fm['error']}")
                    break
                print(f"  df_features: {state['df_features'].shape} | quality: {fm.get('quality',{}).get('null_totali','?')} null")
            elif agent_name == "BaselineAgent":
                bm = state.get("baseline_meta") or {}
                if "error" in bm:
                    print(f"  ERROR: {bm['error']}")
                    break
                print(f"  df_baseline: {state['df_baseline'].shape} | soglia_alta={bm.get('soglia_alta')} | soglia_media={bm.get('soglia_media')}")
            elif agent_name == "OutlierAgent":
                am = state.get("anomaly_meta") or {}
                if "error" in am:
                    print(f"  ERROR: {am['error']}")
                    break
                print(f"  ALTA={am.get('n_alta')} | MEDIA={am.get('n_media')} | NORMALE={am.get('n_normale')}")
            elif agent_name == "ReportAgent":
                rp = state.get("report_path")
                rs = (state.get("report") or {}).get("summary", "N/A")
                print(f"  Report saved: {rp}")
                print(f"  Summary: {rs}")
