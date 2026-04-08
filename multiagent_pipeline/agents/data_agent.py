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
Output (su AgentState): state["df_raw"], state["data_meta"]
"""

# ── Bootstrap per esecuzione diretta (python data_agent.py) ──────────────────
# Permette di lanciare il file sia come modulo (-m) sia come script (▶ VSCode).
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "multiagent_pipeline.agents"

import json
import pandas as pd
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from multiagent_pipeline.state import AgentState, Perimeter, PATHS

# Risolvi i path del dataset rispetto alla root del progetto, cosi' funziona
# da qualsiasi cwd (terminale, "Run File" di VSCode, debugger, ecc.)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS = {k: str(_PROJECT_ROOT / v) if not Path(v).is_absolute() else v
         for k, v in PATHS.items()}


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
        print(f"  [load_dataset] Caricato '{p.name}': {df.shape[0]} righe × {df.shape[1]} colonne")
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
        print(f"  [filter_by_perimeter] Filtri applicati: {label} → {len(df)} righe rimaste")
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

        print(f"  [get_dataset_stats] {stats['n_righe']} righe, "
              f"{stats['n_rotte_uniche']} rotte, "
              f"{stats['n_con_allarmi']} con allarmi")
        return json.dumps(stats)

    except Exception as e:
        return json.dumps({"error": f"Errore nel calcolo statistiche: {str(e)}"})


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE NODO LANGGRAPH
# ══════════════════════════════════════════════════════════════════════════════

def data_agent_node(state: AgentState) -> AgentState:
    """
    Nodo LangGraph per il DataAgent.

    Legge state["perimeter"], esegue i 3 tool in sequenza e scrive
    state["df_raw"] e state["data_meta"].

    In caso di errore non lancia eccezioni: popola data_meta["error"]
    e lascia df_raw = None, così il grafo può gestire il fallimento.
    """
    print("\n[DataAgent] ── Avvio ─────────────────────────────────────")

    try:
        # 1. Leggi e valida il perimetro
        raw_perimeter = state.get("perimeter", {})
        perimeter = Perimeter(**raw_perimeter)

        # 2. Carica il dataset
        data_json = load_dataset.invoke({"path": PATHS["dataset_merged"]})

        # 3. Applica i filtri
        data_json = filter_by_perimeter.invoke({
            "data_json"         : data_json,
            "anno"              : perimeter.anno,
            "aeroporto_arrivo"  : perimeter.aeroporto_arrivo,
            "aeroporto_partenza": perimeter.aeroporto_partenza,
            "paese_partenza"    : perimeter.paese_partenza,
            "zona"              : perimeter.zona,
        })

        # 4. Calcola statistiche
        stats_json = get_dataset_stats.invoke({"data_json": data_json})

        # 5. Controlla errori propagati
        stats = json.loads(stats_json)
        if "error" in stats:
            raise ValueError(stats["error"])

        # 6. Deserializza DataFrame
        df_raw = pd.DataFrame(json.loads(data_json))

        print(f"[DataAgent] ✓ Completato — "
              f"{stats['n_righe']} righe, "
              f"{stats['n_rotte_uniche']} rotte uniche")
        print("[DataAgent] ──────────────────────────────────────────────\n")

        return {
            **state,
            "df_raw"   : df_raw,
            "data_meta": stats,
        }

    except Exception as e:
        print(f"[DataAgent] ✗ Errore: {e}")
        print("[DataAgent] ──────────────────────────────────────────────\n")
        return {
            **state,
            "df_raw"   : None,
            "data_meta": {"error": str(e)},
        }


# ══════════════════════════════════════════════════════════════════════════════
# TEST STANDALONE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  TEST DataAgent — perimetro: anno=2024, Algeria")
    print("=" * 55)

    stato_iniziale: AgentState = {
        "perimeter"    : {"anno": 2024, "paese_partenza": "Algeria"},
        "df_raw"       : None,
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
    print("=" * 55)
