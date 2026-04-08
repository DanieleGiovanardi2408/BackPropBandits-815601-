"""
preprocessing.py
────────────────
Pulizia e normalizzazione dei dataset ALLARMI e TIPOLOGIA_VIAGGIATORE.
Questo script è COMUNE a entrambe le pipeline (classica e multi-agent).

Input:
    data/raw/ALLARMI.csv
    data/raw/TIPOLOGIA_VIAGGIATORE.csv

Output:
    data/processed/allarmi_clean.csv
    data/processed/viaggiatori_clean.csv
    data/processed/dataset_merged.csv

Uso da terminale:
    python classical_pipeline/preprocessing.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ── Percorsi ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
RAW_DIR      = ROOT / "data" / "raw"
PROCESSED_DIR= ROOT / "data" / "processed"

ALLARMI_PATH     = RAW_DIR / "ALLARMI.csv"
VIAGGIATORI_PATH = RAW_DIR / "TIPOLOGIA_VIAGGIATORE.csv"

# ── Colonne duplicate da eliminare ────────────────────────────────────────────
# Sono copie di colonne già presenti con nome diverso
COLS_DROP_ALLARMI = [
    "Paese Partenza",      # duplicato di PAESE_PART
    "CODICE PAESE ARR",    # duplicato di CODICE_PAESE_ARR
    "3zona",               # duplicato di ZONA
    "paese%arr",           # duplicato di PAESE_ARR
    "tot voli",            # duplicato di TOT
]

COLS_DROP_VIAGGIATORI = [
    "Tipo Documento",      # duplicato di TIPO_DOCUMENTO
    "FASCIA ETA",          # duplicato di FASCIA_ETA
    "3nazionalita",        # duplicato di NAZIONALITA
    "compagnia%aerea",     # duplicato di COMPAGNIA_AEREA
    "num volo",            # duplicato di NUMERO_VOLO
]

# ── Mappature normalizzazione ─────────────────────────────────────────────────

# ANNO: tutte le varianti sporche → intero
ANNO_MAP = {
    "24":         2024,
    "anno 2024":  2024,
    "2024.":      2024,
    "2024":       2024,
    "2023":       2023,
}

# GENERE: tutte le varianti → M / F / ND
GENERE_MAP = {
    "M": "M", "m": "M", "Maschio": "M", "MALE": "M", "Male": "M", "1": "M",
    "F": "F", "f": "F", "Femmina": "F", "FEMALE": "F", "Female": "F", "2": "F",
    "N.D.": "ND", "n.d.": "ND", "ND": "ND", "//": "ND", "?": "ND",
    "-": "ND", " ": "ND", "UNKN": "ND", "X": "ND", "N/B": "ND", "unknown": "ND",
}

# TIPO_DOCUMENTO: valori spuri → ND
TIPO_DOC_VALID = {"Passaporto", "Carta d'identità", "Visto", "Permesso di soggiorno", "N.D."}
TIPO_DOC_INVALID = {"?", "unknown", "//", "ND", "n.d.", "-", " "}

# FASCIA_ETA: valori validi da tenere
FASCIA_ETA_VALID = {"0-17", "18-30", "31-45", "46-60", "61+", "N.D."}

# FLAG_TRANSITO: normalizza case
FLAG_TRANSITO_MAP = {
    "singola tratta": "Singola Tratta",
    "N/C":            "N.D.",
}

# ZONA: valori validi [1-9]
ZONE_VALIDE = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}


# ══════════════════════════════════════════════════════════════════════════════
#  FUNZIONI DI PULIZIA
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(path: Path) -> pd.DataFrame:
    """Carica un CSV con rilevamento automatico del separatore."""
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", dtype=str)
            if df.shape[1] > 1:
                print(f"  Caricato '{path.name}' con sep='{sep}' — {df.shape[0]} righe, {df.shape[1]} colonne")
                return df
        except Exception:
            continue
    raise ValueError(f"Impossibile caricare {path}")


def clean_anno(series: pd.Series) -> pd.Series:
    """Normalizza ANNO_PARTENZA → intero 2023/2024."""
    def _fix(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip().lower()
        return ANNO_MAP.get(val, np.nan)
    return series.map(_fix)


def clean_genere(series: pd.Series) -> pd.Series:
    """Normalizza GENERE → M / F / ND."""
    return series.map(lambda x: GENERE_MAP.get(str(x).strip(), "ND") if pd.notna(x) else "ND")


def clean_tipo_documento(series: pd.Series) -> pd.Series:
    """Normalizza TIPO_DOCUMENTO: valori non validi → N.D."""
    def _fix(val):
        if pd.isna(val):
            return "N.D."
        val = str(val).strip()
        if val in TIPO_DOC_INVALID or val not in TIPO_DOC_VALID:
            return "N.D."
        return val
    return series.map(_fix)


def clean_fascia_eta(series: pd.Series) -> pd.Series:
    """Normalizza FASCIA_ETA: valori fuori range → N.D."""
    def _fix(val):
        if pd.isna(val):
            return "N.D."
        val = str(val).strip()
        if val not in FASCIA_ETA_VALID:
            return "N.D."
        return val
    return series.map(_fix)


def clean_data_partenza(series: pd.Series) -> pd.Series:
    """
    Parsa DATA_PARTENZA in datetime uniforme.
    Gestisce formati misti come '2024-02-13 07:30:00' e 'FEB 13 2024'.
    """
    return pd.to_datetime(series, errors="coerce")


def align_temporal_columns(
    df: pd.DataFrame,
    *,
    has_day: bool = False,
    context_label: str = "dataset",
) -> pd.DataFrame:
    """
    Riallinea ANNO/MESE(/GIORNO)_PARTENZA a DATA_PARTENZA quando la data e' valida.
    DATA_PARTENZA viene trattata come fonte autorevole per evitare inconsistenze.
    """
    valid_date_mask = df["DATA_PARTENZA"].notna()
    if valid_date_mask.sum() == 0:
        print(f"  Nessuna DATA_PARTENZA valida da riallineare per {context_label}")
        return df

    mismatch_mask = valid_date_mask & (df["ANNO_PARTENZA"] != df["DATA_PARTENZA"].dt.year)
    corrected_years = int(mismatch_mask.sum())
    df.loc[valid_date_mask, "ANNO_PARTENZA"] = df.loc[valid_date_mask, "DATA_PARTENZA"].dt.year

    if "MESE_PARTENZA" in df.columns:
        df.loc[valid_date_mask, "MESE_PARTENZA"] = df.loc[valid_date_mask, "DATA_PARTENZA"].dt.month

    if has_day and "GIORNO_PARTENZA" in df.columns:
        df.loc[valid_date_mask, "GIORNO_PARTENZA"] = df.loc[valid_date_mask, "DATA_PARTENZA"].dt.day

    fields_label = "ANNO/MESE/GIORNO" if has_day else "ANNO/MESE"
    print(f"  Riallineati {fields_label}: {corrected_years} anno/i corretti in {context_label}")
    return df


def clean_flag_transito(series: pd.Series) -> pd.Series:
    """Normalizza FLAG_TRANSITO: strip + title case + mappa valori noti."""
    def _fix(val):
        if pd.isna(val):
            return "N.D."
        val = str(val).strip()
        return FLAG_TRANSITO_MAP.get(val, val)
    return series.map(_fix)


def clean_zona(series: pd.Series) -> pd.Series:
    """Rimuove valori di ZONA fuori dal range [1-9]."""
    def _fix(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        return val if val in ZONE_VALIDE else np.nan
    return series.map(_fix)


def clean_anno_allarmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    In ALLARMI il campo ANNO_PARTENZA può avere anche '24' e 'anno 2024'.
    Normalizza tutto a intero.
    """
    df["ANNO_PARTENZA"] = clean_anno(df["ANNO_PARTENZA"])
    return df


def add_derived_features_viaggiatori(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge colonne derivate fondamentali per la detection:
    - tasso_allarme    = ALLARMATI / ENTRATI
    - tasso_investigati = INVESTIGATI / ENTRATI
    """
    for col in ["ENTRATI", "ALLARMATI", "INVESTIGATI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Se ENTRATI e' 0, i tassi sono definiti 0.0; se ENTRATI e' mancante restano NaN.
    df["tasso_allarme"] = np.where(
        df["ENTRATI"].isna(),
        np.nan,
        np.where(df["ENTRATI"] > 0, df["ALLARMATI"] / df["ENTRATI"], 0.0),
    )
    df["tasso_investigati"] = np.where(
        df["ENTRATI"].isna(),
        np.nan,
        np.where(df["ENTRATI"] > 0, df["INVESTIGATI"] / df["ENTRATI"], 0.0),
    )
    return df


def clean_tot_allarmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza TOT come conteggio non negativo.
    Valori negativi vengono impostati a NaN per evitare distorsioni.
    """
    df["TOT"] = pd.to_numeric(df["TOT"], errors="coerce")
    n_negative = int((df["TOT"] < 0).sum())
    if n_negative > 0:
        df.loc[df["TOT"] < 0, "TOT"] = np.nan
    print(f"  TOT normalizzato: {n_negative} valori negativi impostati a NaN")
    return df


def enforce_count_constraints_viaggiatori(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applica vincoli di dominio sui conteggi:
      - ENTRATI >= 0
      - 0 <= INVESTIGATI <= ENTRATI
      - 0 <= ALLARMATI <= ENTRATI
    """
    for col in ["ENTRATI", "INVESTIGATI", "ALLARMATI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    neg_entrati = int((df["ENTRATI"] < 0).sum())
    neg_investigati = int((df["INVESTIGATI"] < 0).sum())
    neg_allarmati = int((df["ALLARMATI"] < 0).sum())

    if neg_entrati > 0:
        df.loc[df["ENTRATI"] < 0, "ENTRATI"] = np.nan
    if neg_investigati > 0:
        df.loc[df["INVESTIGATI"] < 0, "INVESTIGATI"] = np.nan
    if neg_allarmati > 0:
        df.loc[df["ALLARMATI"] < 0, "ALLARMATI"] = np.nan

    over_investigati = int(((df["INVESTIGATI"] > df["ENTRATI"]) & df["ENTRATI"].notna()).sum())
    over_allarmati = int(((df["ALLARMATI"] > df["ENTRATI"]) & df["ENTRATI"].notna()).sum())

    df.loc[
        (df["ENTRATI"].notna()) & (df["INVESTIGATI"] > df["ENTRATI"]),
        "INVESTIGATI",
    ] = df.loc[
        (df["ENTRATI"].notna()) & (df["INVESTIGATI"] > df["ENTRATI"]),
        "ENTRATI",
    ]
    df.loc[
        (df["ENTRATI"].notna()) & (df["ALLARMATI"] > df["ENTRATI"]),
        "ALLARMATI",
    ] = df.loc[
        (df["ENTRATI"].notna()) & (df["ALLARMATI"] > df["ENTRATI"]),
        "ENTRATI",
    ]

    print(
        "  Conteggi vincolati: "
        f"ENTRATI<0 -> NaN: {neg_entrati}, "
        f"INVESTIGATI<0 -> NaN: {neg_investigati}, "
        f"ALLARMATI<0 -> NaN: {neg_allarmati}, "
        f"INVESTIGATI>ENTRATI capped: {over_investigati}, "
        f"ALLARMATI>ENTRATI capped: {over_allarmati}"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE DI PULIZIA ALLARMI
# ══════════════════════════════════════════════════════════════════════════════

def clean_allarmi(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Pulizia ALLARMI ──────────────────────────────────────")

    # 1. Rimuovi colonne duplicate
    cols_to_drop = [c for c in COLS_DROP_ALLARMI if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  Rimosse {len(cols_to_drop)} colonne duplicate: {cols_to_drop}")

    # 2. Normalizza ANNO_PARTENZA
    before = df["ANNO_PARTENZA"].value_counts().to_dict()
    df = clean_anno_allarmi(df)
    after  = df["ANNO_PARTENZA"].value_counts(dropna=False).to_dict()
    print(f"  ANNO_PARTENZA: {before} → {after}")

    # 3. Normalizza DATA_PARTENZA
    n_before = df["DATA_PARTENZA"].notna().sum()
    df["DATA_PARTENZA"] = clean_data_partenza(df["DATA_PARTENZA"])
    n_after  = df["DATA_PARTENZA"].notna().sum()
    print(f"  DATA_PARTENZA parsed: {n_after}/{n_before} validi")

    # 4. Riallinea anno/mese con la data
    df = align_temporal_columns(df, has_day=False, context_label="ALLARMI")

    # 5. Normalizza ZONA
    n_invalid = (~df["ZONA"].isin(ZONE_VALIDE)).sum()
    df["ZONA"] = clean_zona(df["ZONA"])
    print(f"  ZONA: rimossi {n_invalid} valori fuori range")

    # 6. Normalizza TOT come conteggio non negativo
    df = clean_tot_allarmi(df)

    # 7. Strip su colonne stringa chiave
    for col in ["AREOPORTO_ARRIVO", "AREOPORTO_PARTENZA", "CODICE_PAESE_ARR",
                "CODICE_PAESE_PART", "PAESE_ARR", "PAESE_PART", "MOTIVO_ALLARME"]:
        if col in df.columns:
            df[col] = df[col].str.strip()

    # 8. Normalizza CODICE_PAESE_ARR: 'IT' → 'ITA' (inconsistenza trovata nell'EDA)
    df["CODICE_PAESE_ARR"] = df["CODICE_PAESE_ARR"].replace({"IT": "ITA"})
    df["CODICE_PAESE_PART"] = df["CODICE_PAESE_PART"].replace({"GB": "GBR", "TR": "TUR"})

    # 9. Estrai colonne temporali da DATA_PARTENZA (utili per feature engineering)
    df["ora_partenza"]      = df["DATA_PARTENZA"].dt.hour
    df["giorno_settimana"]  = df["DATA_PARTENZA"].dt.dayofweek   # 0=lunedì
    df["mese"]              = df["DATA_PARTENZA"].dt.month

    print(f"  Shape finale: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE DI PULIZIA VIAGGIATORI
# ══════════════════════════════════════════════════════════════════════════════

def clean_viaggiatori(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Pulizia TIPOLOGIA_VIAGGIATORE ────────────────────────")

    # 1. Rimuovi colonne duplicate
    cols_to_drop = [c for c in COLS_DROP_VIAGGIATORI if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  Rimosse {len(cols_to_drop)} colonne duplicate: {cols_to_drop}")

    # 2. Normalizza ANNO_PARTENZA
    df["ANNO_PARTENZA"] = clean_anno(df["ANNO_PARTENZA"])

    # 3. Normalizza DATA_PARTENZA
    df["DATA_PARTENZA"] = clean_data_partenza(df["DATA_PARTENZA"])
    print(f"  DATA_PARTENZA parsed: {df['DATA_PARTENZA'].notna().sum()}/{len(df)} validi")

    # 4. Riallinea anno/mese/giorno con la data
    df = align_temporal_columns(df, has_day=True, context_label="TIPOLOGIA_VIAGGIATORE")

    # 5. Normalizza GENERE
    before_vals = df["GENERE"].value_counts().nlargest(5).to_dict()
    df["GENERE"] = clean_genere(df["GENERE"])
    print(f"  GENERE normalizzato: {before_vals} → {df['GENERE'].value_counts().to_dict()}")

    # 6. Normalizza TIPO_DOCUMENTO
    df["TIPO_DOCUMENTO"] = clean_tipo_documento(df["TIPO_DOCUMENTO"])
    print(f"  TIPO_DOCUMENTO: {df['TIPO_DOCUMENTO'].value_counts().to_dict()}")

    # 7. Normalizza FASCIA_ETA
    n_invalid = (~df["FASCIA_ETA"].isin(FASCIA_ETA_VALID)).sum()
    df["FASCIA_ETA"] = clean_fascia_eta(df["FASCIA_ETA"])
    print(f"  FASCIA_ETA: corretti {n_invalid} valori non validi")

    # 8. Normalizza FLAG_TRANSITO
    df["FLAG_TRANSITO"] = clean_flag_transito(df["FLAG_TRANSITO"])

    # 9. Normalizza ZONA
    df["ZONA"] = clean_zona(df["ZONA"])

    # 10. Strip su colonne stringa chiave
    for col in ["AREOPORTO_ARRIVO", "AREOPORTO_PARTENZA", "NAZIONALITA",
                "CODICE_PAESE_ARR", "CODICE_PAESE_PART", "ESITO_CONTROLLO",
                "COMPAGNIA_AEREA", "NUMERO_VOLO"]:
        if col in df.columns:
            df[col] = df[col].str.strip() if df[col].dtype == object else df[col]

    # 11. Normalizza NAZIONALITA: valori spuri → ND
    valori_spuri_naz = {"ND", "-", "?", "n.d.", "//", "unknown"}
    df["NAZIONALITA"] = df["NAZIONALITA"].apply(
        lambda x: "ND" if pd.isna(x) or str(x).strip() in valori_spuri_naz else str(x).strip()
    )

    # 12. Applica vincoli di dominio sui conteggi
    df = enforce_count_constraints_viaggiatori(df)

    # 13. Aggiungi feature derivate
    df = add_derived_features_viaggiatori(df)

    # 14. Estrai colonne temporali
    df["ora_partenza"]     = df["DATA_PARTENZA"].dt.hour
    df["giorno_settimana"] = df["DATA_PARTENZA"].dt.dayofweek
    df["mese"]             = df["DATA_PARTENZA"].dt.month

    print(f"  Shape finale: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  JOIN TRA I DUE DATASET
# ══════════════════════════════════════════════════════════════════════════════

def merge_datasets(df_allarmi: pd.DataFrame, df_viaggiatori: pd.DataFrame) -> pd.DataFrame:
    """
    Join tra ALLARMI e TIPOLOGIA_VIAGGIATORE su:
    AREOPORTO_ARRIVO + AREOPORTO_PARTENZA + DATA_PARTENZA (solo data, no ora).

    Left join: mantiene tutti i record di ALLARMI, aggiunge info viaggiatori
    dove disponibile.
    """
    print("\n── Merge dataset ────────────────────────────────────────")

    # Crea chiave di join solo sulla data (senza orario) per aumentare i match
    df_allarmi["_data_key"]     = df_allarmi["DATA_PARTENZA"].dt.date
    df_viaggiatori["_data_key"] = df_viaggiatori["DATA_PARTENZA"].dt.date

    join_keys = ["AREOPORTO_ARRIVO", "AREOPORTO_PARTENZA", "_data_key"]

    # Aggrega VIAGGIATORI per volo (somma passeggeri, tasso medio)
    agg_viaggiatori = df_viaggiatori.groupby(join_keys).agg(
        tot_entrati        = ("ENTRATI",          "sum"),
        tot_allarmati      = ("ALLARMATI",         "sum"),
        tot_investigati    = ("INVESTIGATI",       "sum"),
        tasso_allarme_volo = ("tasso_allarme",     "mean"),
        tasso_inv_volo     = ("tasso_investigati", "mean"),
        n_nazionalita      = ("NAZIONALITA",       "nunique"),
        n_respinti         = ("ESITO_CONTROLLO",   lambda x: (x == "RESPINTO").sum()),
        n_fermati          = ("ESITO_CONTROLLO",   lambda x: (x == "FERMATO").sum()),
        n_segnalati        = ("ESITO_CONTROLLO",   lambda x: (x == "SEGNALATO").sum()),
    ).reset_index()

    merged = df_allarmi.merge(agg_viaggiatori, on=join_keys, how="left")

    # Rimuovi colonna temporanea di join da tutti i dataframe
    for _df in [merged, df_allarmi, df_viaggiatori]:
        if "_data_key" in _df.columns:
            _df.drop(columns=["_data_key"], inplace=True)

    print(f"  Righe ALLARMI:      {len(df_allarmi)}")
    print(f"  Righe VIAGGIATORI:  {len(df_viaggiatori)}")
    print(f"  Righe dopo merge:   {len(merged)}")
    print(f"  Match trovati:      {merged['tot_entrati'].notna().sum()}/{len(merged)}")

    return merged


# ══════════════════════════════════════════════════════════════════════════════
#  QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_quality_report(df: pd.DataFrame, name: str):
    print(f"\n{'='*55}")
    print(f"  Quality Report — {name}")
    print(f"{'='*55}")
    print(f"  Shape: {df.shape[0]} righe × {df.shape[1]} colonne")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) > 0:
        print("  Null values rimasti:")
        for col, n in nulls.items():
            pct = n / len(df) * 100
            print(f"    {col:<35} {n:>5} ({pct:.1f}%)")
    else:
        print("  Nessun null rilevante rimasto.")
    print(f"{'='*55}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_preprocessing() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Esegue l'intera pipeline di preprocessing.
    Ritorna (df_allarmi_clean, df_viaggiatori_clean, df_merged).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Caricamento dataset...")
    df_allarmi     = load_csv(ALLARMI_PATH)
    df_viaggiatori = load_csv(VIAGGIATORI_PATH)

    # Pulizia
    df_allarmi     = clean_allarmi(df_allarmi)
    df_viaggiatori = clean_viaggiatori(df_viaggiatori)

    # Merge
    df_merged = merge_datasets(df_allarmi, df_viaggiatori)

    # Quality report
    print_quality_report(df_allarmi,     "ALLARMI clean")
    print_quality_report(df_viaggiatori, "VIAGGIATORI clean")
    print_quality_report(df_merged,      "MERGED")

    # Salvataggio
    allarmi_out     = PROCESSED_DIR / "allarmi_clean.csv"
    viaggiatori_out = PROCESSED_DIR / "viaggiatori_clean.csv"
    merged_out      = PROCESSED_DIR / "dataset_merged.csv"

    df_allarmi.to_csv(allarmi_out,         index=False)
    df_viaggiatori.to_csv(viaggiatori_out, index=False)
    df_merged.to_csv(merged_out,           index=False)

    print(f"\nFile salvati in data/processed/:")
    print(f"  {allarmi_out.name}")
    print(f"  {viaggiatori_out.name}")
    print(f"  {merged_out.name}")

    return df_allarmi, df_viaggiatori, df_merged


if __name__ == "__main__":
    run_preprocessing()
