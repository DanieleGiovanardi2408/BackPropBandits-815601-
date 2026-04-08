"""Tool puri per il DataAgent.

Tutte funzioni pandas deterministiche, senza LLM né stato globale.
Testabili in isolazione con pytest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Carica il dataset merged prodotto dal preprocessing classico."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset non trovato: {path}")
    return pd.read_csv(path)


# Mapping: chiave del perimetro -> colonna reale nel CSV.
# Le chiavi DEVONO corrispondere ai field name di state.Perimeter (Pydantic),
# che e' il contratto del DataAgent.
_PERIMETER_COLS = {
    "anno":               "ANNO_PARTENZA",
    "mese":               "MESE_PARTENZA",      # extra (non in Perimeter, ma utile)
    "aeroporto_partenza": "AREOPORTO_PARTENZA",
    "aeroporto_arrivo":   "AREOPORTO_ARRIVO",
    "paese_partenza":     "PAESE_PART",
    "zona":               "ZONA",
}


def filter_by_perimeter(df: pd.DataFrame, perimeter: Optional[dict]) -> pd.DataFrame:
    """Filtra il DataFrame applicando i vincoli del perimetro.

    Chiavi non specificate -> nessun filtro su quella dimensione.
    Ritorna una copia (non modifica l'input).
    """
    if not perimeter:
        return df.copy()

    out = df
    for key, value in perimeter.items():
        if value is None:
            continue
        col = _PERIMETER_COLS.get(key)
        if col is None:
            raise KeyError(f"Chiave perimetro sconosciuta: {key}")
        if col not in out.columns:
            raise KeyError(f"Colonna {col} non presente nel dataset")
        out = out[out[col] == value]
    return out.copy()


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Statistiche descrittive del DataFrame filtrato (per il report e i log)."""
    stats = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
    }
    if "ANNO_PARTENZA" in df.columns and len(df):
        stats["anni"] = sorted(df["ANNO_PARTENZA"].dropna().unique().tolist())
    if "AREOPORTO_PARTENZA" in df.columns:
        stats["n_aeroporti_part"] = int(df["AREOPORTO_PARTENZA"].nunique())
    if "AREOPORTO_ARRIVO" in df.columns:
        stats["n_aeroporti_arr"] = int(df["AREOPORTO_ARRIVO"].nunique())
    if "PAESE_ARR" in df.columns:
        stats["n_paesi_arr"] = int(df["PAESE_ARR"].nunique())
    return stats
