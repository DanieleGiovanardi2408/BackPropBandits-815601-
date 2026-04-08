"""
data_agent.py
─────────────
Agent 1 — DataAgent

Responsabilità:
    Carica il dataset merged già pulito dal preprocessing, applica i filtri
    definiti dall'utente (anno, aeroporto, paese, zona) e restituisce il
    DataFrame filtrato con le sue statistiche nello stato condiviso.

Architettura:
    Agente DETERMINISTICO — non usa un LLM.
    Esegue sempre gli stessi 3 tool in sequenza. Questa è una scelta
    deliberata: il DataAgent sa esattamente cosa fare, un LLM sarebbe
    spreco di risorse e latenza.

    load_dataset → filter_by_perimeter → get_dataset_stats

Input  (da AgentState): state["perimeter"]
Output (su AgentState): state["df_raw"], state["df_allarmi"],
                        state["df_viaggiatori"], state["data_meta"]
"""

# ── Bootstrap per esecuzione diretta (python data_agent.py) ──────────────────
# Permette di lanciare il file sia come modulo (-m) sia come script (▶ VSCode).
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

from langchain_core.tools import tool

from multiagent_pipeline.state import AgentState, Perimeter, PATHS

logger = logging.getLogger(__name__)

# Risolvi i path del dataset rispetto alla root del progetto, cosi' funziona
# da qualsiasi cwd (terminale, "Run File" di VSCode, debugger, ecc.)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS = {k: str(_PROJECT_ROOT / v) if not Path(v).is_absolute() else v
         for k, v in PATHS.items()}

# Output del DataAgent (artefatti per audit / handoff al prossimo agente)
DATA_AGENT_OUTPUT_JSON = _PROJECT_ROOT / "data" / "processed" / "data_agent_output.json"
DATA_AGENT_OUTPUT_CSV  = _PROJECT_ROOT / "data" / "processed" / "data_agent_filtered.csv"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — load_dataset
# ══════════════════════════════════════════════════════════════════════════════

@tool
def load_dataset(path: str) -> str:
    """
    Carica il dataset dei transiti aeroportuali da un file CSV.
    Restituisce i dati in formato JSON (orient=records).
    Usare sempre come primo step prima di applicare filtri.
    """
    try:
        p = Path(path)
        if not p.exists():
            return json.dumps({"error": f"File non trovato: {path}"})

        df = pd.read_csv(p)
        logger.info("[load_dataset] Caricato '%s': %d righe × %d colonne", p.name, df.shape[0], df.shape[1])
        return df.to_json(orient="records", date_format="iso")

    except Exception as e:
        return json.dumps({"error": f"Errore durante il caricamento: {str(e)}"})


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — filter_by_perimeter
# ══════════════════════════════════════════════════════════════════════════════

@tool
def filter_by_perimeter(
    data_json: str,
    anno: Optional[int] = None,
    aeroporto_arrivo: Optional[str] = None,
    aeroporto_partenza: Optional[str] = None,
    paese_partenza: Optional[str] = None,
    zona: Optional[int] = None,
) -> str:
    """
    Filtra il dataset per il perimetro di analisi definito dall'utente.
    Applica solo i filtri i cui parametri non sono None.
    Tutti i confronti su stringhe sono case-insensitive.
    Restituisce il dataset filtrato in formato JSON (orient=records).
    """
    try:
        parsed = json.loads(data_json)
        if isinstance(parsed, dict) and "error" in parsed:
            return data_json  # propaga errore precedente

        df = pd.DataFrame(parsed)
        filtri_applicati = []

        if anno is not None:
            df = df[df["ANNO_PARTENZA"] == anno]
            filtri_applicati.append(f"anno={anno}")

        if aeroporto_arrivo is not None:
            df = df[df["AREOPORTO_ARRIVO"].str.upper() == aeroporto_arrivo.upper()]
            filtri_applicati.append(f"aeroporto_arrivo={aeroporto_arrivo}")

        if aeroporto_partenza is not None:
            df = df[df["AREOPORTO_PARTENZA"].str.upper() == aeroporto_partenza.upper()]
            filtri_applicati.append(f"aeroporto_partenza={aeroporto_partenza}")

        if paese_partenza is not None:
            df = df[df["PAESE_PART"].str.upper() == paese_partenza.upper()]
            filtri_applicati.append(f"paese_partenza={paese_partenza}")

        if zona is not None:
            df = df[df["ZONA"] == zona]
            filtri_applicati.append(f"zona={zona}")

        if df.empty:
            return json.dumps({
                "error": f"Nessun dato trovato con i filtri: {', '.join(filtri_applicati)}"
            })

        label = ', '.join(filtri_applicati) if filtri_applicati else "nessuno"
        logger.info("[filter_by_perimeter] Filtri applicati: %s -> %d righe rimaste", label, len(df))
        return df.to_json(orient="records", date_format="iso")

    except Exception as e:
        return json.dumps({"error": f"Errore durante il filtraggio: {str(e)}"})


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — get_dataset_stats
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_dataset_stats(data_json: str) -> str:
    """
    Calcola statistiche descrittive sul dataset filtrato.
    Restituisce n_righe, n_rotte_uniche, anni_presenti,
    paesi_partenza_top5 e n_con_allarmi (righe con TOT > 0).
    Usare dopo filter_by_perimeter per ottenere una panoramica dei dati.
    """
    try:
        parsed = json.loads(data_json)
        if isinstance(parsed, dict) and "error" in parsed:
            return data_json  # propaga errore precedente

        df = pd.DataFrame(parsed)

        # Costruisce colonna ROTTA se non presente
        if "ROTTA" not in df.columns:
            df["ROTTA"] = (
                df["AREOPORTO_PARTENZA"].str.upper() + "-" +
                df["AREOPORTO_ARRIVO"].str.upper()
            )

        # TOT numerico per contare gli allarmi
        tot = pd.to_numeric(df["TOT"], errors="coerce").fillna(0)

        stats = {
            "n_righe"            : int(len(df)),
            "n_rotte_uniche"     : int(df["ROTTA"].nunique()),
            "anni_presenti"      : sorted(df["ANNO_PARTENZA"].dropna().unique().tolist()),
            "paesi_partenza_top5": df["PAESE_PART"].value_counts().head(5).index.tolist(),
            "n_con_allarmi"      : int((tot > 0).sum()),
        }

        logger.info(
            "[get_dataset_stats] %d righe, %d rotte, %d con allarmi",
            stats["n_righe"],
            stats["n_rotte_uniche"],
            stats["n_con_allarmi"],
        )
        return json.dumps(stats)

    except Exception as e:
        return json.dumps({"error": f"Errore nel calcolo statistiche: {str(e)}"})


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE NODO LANGGRAPH
# ══════════════════════════════════════════════════════════════════════════════

def data_agent_node(state: AgentState) -> AgentState:
    """
    Nodo LangGraph per il DataAgent.

    Legge state["perimeter"], esegue i tool in sequenza e scrive:
      - state["df_raw"]          (dataset_merged filtrato)
      - state["df_allarmi"]      (allarmi_clean filtrato)
      - state["df_viaggiatori"]  (viaggiatori_clean filtrato)
      - state["data_meta"]       (statistiche principali)

    In caso di errore non lancia eccezioni: popola data_meta["error"]
    e lascia df_raw = None, così il grafo può gestire il fallimento.
    """
    logger.info("DataAgent -- Avvio")

    try:
        # 1. Leggi e valida il perimetro
        raw_perimeter = state.get("perimeter", {})
        perimeter = Perimeter(**raw_perimeter)

        # 2. Carica i dataset (merged + clean separati)
        merged_json = load_dataset.invoke({"path": PATHS["dataset_merged"]})
        allarmi_json = load_dataset.invoke({"path": PATHS["allarmi_clean"]})
        viaggiatori_json = load_dataset.invoke({"path": PATHS["viaggiatori_clean"]})

        # 3. Applica i filtri su tutti i dataset
        merged_json = filter_by_perimeter.invoke({
            "data_json"         : merged_json,
            "anno"              : perimeter.anno,
            "aeroporto_arrivo"  : perimeter.aeroporto_arrivo,
            "aeroporto_partenza": perimeter.aeroporto_partenza,
            "paese_partenza"    : perimeter.paese_partenza,
            "zona"              : perimeter.zona,
        })
        allarmi_json = filter_by_perimeter.invoke({
            "data_json"         : allarmi_json,
            "anno"              : perimeter.anno,
            "aeroporto_arrivo"  : perimeter.aeroporto_arrivo,
            "aeroporto_partenza": perimeter.aeroporto_partenza,
            "paese_partenza"    : perimeter.paese_partenza,
            "zona"              : perimeter.zona,
        })
        viaggiatori_json = filter_by_perimeter.invoke({
            "data_json"         : viaggiatori_json,
            "anno"              : perimeter.anno,
            "aeroporto_arrivo"  : perimeter.aeroporto_arrivo,
            "aeroporto_partenza": perimeter.aeroporto_partenza,
            "paese_partenza"    : perimeter.paese_partenza,
            "zona"              : perimeter.zona,
        })

        # 4. Calcola statistiche
        stats_json = get_dataset_stats.invoke({"data_json": merged_json})

        # 5. Controlla errori propagati
        stats = json.loads(stats_json)
        if "error" in stats:
            raise ValueError(stats["error"])

        for payload in [allarmi_json, viaggiatori_json]:
            parsed = json.loads(payload)
            if isinstance(parsed, dict) and "error" in parsed:
                raise ValueError(parsed["error"])

        # 6. Deserializza DataFrame
        df_raw = pd.DataFrame(json.loads(merged_json))
        df_allarmi = pd.DataFrame(json.loads(allarmi_json))
        df_viaggiatori = pd.DataFrame(json.loads(viaggiatori_json))
        stats["n_righe_allarmi"] = int(len(df_allarmi))
        stats["n_righe_viaggiatori"] = int(len(df_viaggiatori))

        # 7. Salva artefatti su disco (audit + handoff)
        DATA_AGENT_OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "perimeter": perimeter.model_dump(),
            "data_meta": stats,
            "output_csv": str(DATA_AGENT_OUTPUT_CSV.relative_to(_PROJECT_ROOT)),
        }
        DATA_AGENT_OUTPUT_JSON.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
        df_raw.to_csv(DATA_AGENT_OUTPUT_CSV, index=False)
        print(f"  [save] {DATA_AGENT_OUTPUT_JSON.name} + {DATA_AGENT_OUTPUT_CSV.name}")

        logger.info(
            "DataAgent ✓ Completato — %d righe, %d rotte uniche",
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
        logger.error("DataAgent ✗ Errore: %s", e)
        return {
            **state,
            "df_raw"   : None,
            "df_allarmi": None,
            "df_viaggiatori": None,
            "data_meta": {"error": str(e)},
        }


# ══════════════════════════════════════════════════════════════════════════════
# TEST STANDALONE
# ══════════════════════════════════════════════════════════════════════════════

def _pick_value(df, col: str, label: str, cast=str, top: int = 30):
    """Mostra i valori unici della colonna e fa scegliere all'utente.

    L'utente puo' scrivere il numero (dalla lista) o digitare direttamente il valore.
    """
    if col not in df.columns:
        print(f"  ⚠ Colonna {col} non trovata, skip.")
        return None

    values = (
        df[col].dropna().value_counts().head(top).index.tolist()
    )
    if not values:
        print(f"  ⚠ Nessun valore disponibile per {col}.")
        return None

    print(f"\n  ── Valori disponibili per {label} (top {len(values)}) ──")
    for i, v in enumerate(values, 1):
        n = int((df[col] == v).sum())
        print(f"    {i:>3}. {v}  ({n} righe)")
    raw = input(f"  Scegli (numero o valore esatto): ").strip()
    if not raw:
        return None

    # Numero della lista?
    if raw.isdigit() and 1 <= int(raw) <= len(values):
        return cast(values[int(raw) - 1])

    # Valore digitato a mano
    try:
        return cast(raw)
    except Exception as e:
        print(f"  ⚠ valore non valido: {e}")
        return None


def _interactive_perimeter() -> dict:
    """CLI interattiva: mostra i filtri disponibili e i valori reali del dataset."""
    # Carica il dataset una sola volta per popolare i menu
    try:
        df_preview = pd.read_csv(PATHS["dataset_merged"])
    except Exception as e:
        print(f"  ⚠ Impossibile caricare dataset per anteprima: {e}")
        df_preview = None

    # (key, label, colonna_csv, cast)
    fields = [
        ("anno",               "Anno",                  "ANNO_PARTENZA",      int),
        ("aeroporto_partenza", "Aeroporto di partenza", "AREOPORTO_PARTENZA", str),
        ("aeroporto_arrivo",   "Aeroporto di arrivo",   "AREOPORTO_ARRIVO",   str),
        ("paese_partenza",     "Paese di partenza",     "PAESE_PART",         str),
        ("zona",               "Zona geografica",       "ZONA",               int),
    ]

    print("\n── Filtri disponibili (perimetro) ────────────────────")
    for i, (key, label, *_) in enumerate(fields, 1):
        print(f"  {i}. {key:20s} — {label}")
    print("──────────────────────────────────────────────────────")

    raw = input("Quali filtri vuoi applicare? (numeri o nomi separati da virgola, vuoto = nessuno): ").strip()
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
            print(f"  ⚠ ignorato: '{tok}' (non e' un numero ne' un nome valido)")

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
    print("  DataAgent — modalità interattiva")
    print("=" * 55)

    perimeter = _interactive_perimeter()
    print(f"\n  Perimetro selezionato: {perimeter or '(nessuno)'}")

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

    stato_finale = data_agent_node(stato_iniziale)

    print("── Risultato ─────────────────────────────────────────")
    if stato_finale["data_meta"].get("error"):
        print(f"  ERRORE: {stato_finale['data_meta']['error']}")
    else:
        meta = stato_finale["data_meta"]
        print(f"  Righe caricate:    {meta['n_righe']}")
        print(f"  Rotte uniche:      {meta['n_rotte_uniche']}")
        print(f"  Anni presenti:     {meta['anni_presenti']}")
        print(f"  Top 5 paesi:       {meta['paesi_partenza_top5']}")
        print(f"  Righe con allarmi: {meta['n_con_allarmi']}")
        print(f"  df_raw shape:      {stato_finale['df_raw'].shape}")
        print(f"  df_allarmi shape:  {stato_finale['df_allarmi'].shape}")
        print(f"  df_viag shape:     {stato_finale['df_viaggiatori'].shape}")
    print("=" * 55)
