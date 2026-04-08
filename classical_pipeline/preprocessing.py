"""
preprocessing.py
────────────────
Pulizia e normalizzazione dei dataset ALLARMI e TIPOLOGIA_VIAGGIATORE.
Questo script è COMUNE a entrambe le pipeline (classica e multi-agent).

Versione unificata: integra lo script originale con le correzioni del
notebook 01_Data_Ingestion_And_Cleaning.ipynb del collega.

Miglioramenti rispetto alla versione precedente:
  - NULL_VALUES globale applicato a inizio pipeline
  - parse_date con mappatura mesi italiani e fallback multipli
  - italian_months per MESE_PARTENZA testuale
  - anno_corrections: '2023'→2024 (encoding error noto nel raw)
  - extract_number: gestisce '123 pax', '~45', '1,5', ecc.
  - iso2_to_iso3 completa (~65 paesi) su CODICE_PAESE + NAZIONALITA
  - .str.upper() sui codici aeroporto (fix bug merge)
  - normalize_gender: '1'/'2' → NaN (conservativo)
  - fascia_map: recupera 'minore'→'0-17', 'adulto'→'31-45', '101+'→'61+'
  - TOT: rimuove non-interi e placeholder >9999
  - OCCORRENZE: '???', 'N/C', 'ALLARMATI' → NaN
  - Drop colonne con >50% null
  - Overwrite ANNO/MESE da DATA_PARTENZA quando disponibile

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

import pandas as pd
import numpy as np
from pathlib import Path

# ── Percorsi ──────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

ALLARMI_PATH     = RAW_DIR / "ALLARMI.csv"
VIAGGIATORI_PATH = RAW_DIR / "TIPOLOGIA_VIAGGIATORE.csv"

# ── Colonne duplicate da eliminare ────────────────────────────────────────────
COLS_DROP_ALLARMI = [
    "Paese Partenza", "CODICE PAESE ARR", "3zona", "paese%arr", "tot voli",
]
COLS_DROP_VIAGGIATORI = [
    "Tipo Documento", "FASCIA ETA", "3nazionalita", "compagnia%aerea", "num volo",
]

# ── NULL: tutte le varianti trovate nel dataset raw ───────────────────────────
NULL_VALUES = [
    "N.D.", "n.d.", "ND", "N/D", "N/A", "n/a", "?", "??", "???",
    "//", "-", "null", "NULL", "unknown", "Unknown", "UNKN", "UNK",
    " ", "", "ZZ", "XX", "EU",
]

# ── Date: mesi italiani ───────────────────────────────────────────────────────
ITALIAN_MONTHS = {
    "GEN": 1, "FEB": 2,  "MAR": 3,  "APR": 4,
    "MAG": 5, "GIU": 6,  "LUG": 7,  "AGO": 8,
    "SET": 9, "OTT": 10, "NOV": 11, "DIC": 12,
}

ITALIAN_MONTHS_LONG = {
    "GEN": "Jan", "FEB": "Feb", "MAR": "Mar", "APR": "Apr",
    "MAG": "May", "GIU": "Jun", "LUG": "Jul", "AGO": "Aug",
    "SET": "Sep", "OTT": "Oct", "NOV": "Nov", "DIC": "Dec",
}

# ── Anno: correzioni encoding note ───────────────────────────────────────────
ANNO_CORRECTIONS = {
    "24": "2024", "anno 2024": "2024",
    "2023": "2024",   # encoding error noto nel raw (vedi EDA)
    "2024.": "2024", "2024": "2024",
}

# ── Paese: ISO2 → ISO3 completa ───────────────────────────────────────────────
ISO2_TO_ISO3 = {
    "IT": "ITA", "AL": "ALB", "TR": "TUR", "AE": "ARE", "GB": "GBR",
    "EG": "EGY", "DE": "DEU", "FR": "FRA", "ES": "ESP", "PT": "PRT",
    "NL": "NLD", "BE": "BEL", "CH": "CHE", "AT": "AUT", "GR": "GRC",
    "HR": "HRV", "RS": "SRB", "BG": "BGR", "RO": "ROU", "PL": "POL",
    "UA": "UKR", "RU": "RUS", "US": "USA", "CA": "CAN", "MX": "MEX",
    "BR": "BRA", "AR": "ARG", "CN": "CHN", "JP": "JPN", "KR": "KOR",
    "IN": "IND", "PK": "PAK", "BD": "BGD", "TH": "THA", "VN": "VNM",
    "ID": "IDN", "PH": "PHL", "MY": "MYS", "SG": "SGP", "AU": "AUS",
    "NZ": "NZL", "ZA": "ZAF", "NG": "NGA", "KE": "KEN", "ET": "ETH",
    "MA": "MAR", "TN": "TUN", "DZ": "DZA", "LY": "LBY", "SD": "SDN",
    "SA": "SAU", "QA": "QAT", "KW": "KWT", "IR": "IRN", "IQ": "IRQ",
    "SY": "SYR", "LB": "LBN", "JO": "JOR", "IL": "ISR", "AF": "AFG",
    "AO": "AGO", "AD": "AND", "MD": "MDA", "MK": "MKD", "XK": "RKS",
    "MV": "MDV", "AZ": "AZE", "GE": "GEO", "AM": "ARM", "BY": "BLR",
    "LT": "LTU", "LV": "LVA", "EE": "EST", "FI": "FIN", "SE": "SWE",
    "NO": "NOR", "DK": "DNK", "IS": "ISL", "IE": "IRL", "CZ": "CZE",
    "SK": "SVK", "HU": "HUN", "SI": "SVN", "LU": "LUX", "MT": "MLT",
    "CY": "CYP",
}

# ── Genere ────────────────────────────────────────────────────────────────────
FEMALE_VALS = {"f", "femmina", "female", "donna", "f."}
MALE_VALS   = {"m", "maschio", "male", "uomo", "m."}

# ── Fascia età ────────────────────────────────────────────────────────────────
FASCIA_ETA_VALID = {"0-17", "18-30", "31-45", "46-60", "61+", "N.D."}
FASCIA_MAP = {
    "minore": "0-17",
    "adulto": "31-45",
    "101+":   "61+",
    "-5":     np.nan,
    "N/C":    np.nan,
}

# ── Zona ─────────────────────────────────────────────────────────────────────
ZONE_VALIDE = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}

# ── OCCORRENZE non valide ─────────────────────────────────────────────────────
OCC_INVALID = {"???", "N/C", "ALLARMATI"}

# ── Soglia colonne sparse ─────────────────────────────────────────────────────
NULL_DROP_THRESHOLD = 0.50


# ══════════════════════════════════════════════════════════════════════════════
#  FUNZIONI HELPER
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(path: Path) -> pd.DataFrame:
    """Carica un CSV con rilevamento automatico del separatore."""
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", dtype=str)
            if df.shape[1] > 1:
                print(f"  Caricato '{path.name}' con sep='{sep}' "
                      f"— {df.shape[0]} righe, {df.shape[1]} colonne")
                return df
        except Exception:
            continue
    raise ValueError(f"Impossibile caricare {path}")


def parse_date(val) -> pd.Timestamp:
    """
    Parser robusto per DATA_PARTENZA.
    Gestisce mesi italiani ('FEB 13 2024'), formati misti e fallback.
    """
    if pd.isna(val):
        return pd.NaT
    val = str(val).strip()
    for ita, eng in ITALIAN_MONTHS_LONG.items():
        val = val.replace(ita, eng)
    for fmt in (
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
        "%Y/%m/%d", "%d.%m.%Y", "%d-%m-%y", "%b %d %Y",
    ):
        try:
            return pd.to_datetime(val, format=fmt)
        except Exception:
            pass
    try:
        return pd.to_datetime(val, dayfirst=True)
    except Exception:
        return pd.NaT


def extract_number(val) -> float:
    """
    Estrae il valore numerico da stringhe con rumore testuale.
    Gestisce: '123 pax', '~45', '1,5', '20 voli', ecc.
    """
    if pd.isna(val):
        return np.nan
    val = (str(val).strip()
           .replace(",", ".")
           .replace("~", "")
           .replace("pax", "")
           .replace("voli", "")
           .strip())
    try:
        return float(val)
    except Exception:
        return np.nan


def normalize_gender(val) -> str:
    """
    Normalizza GENERE → M / F / NaN.
    Conservativo: '1'/'2' → NaN (potrebbero essere rumore, non codifica).
    """
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    if v in FEMALE_VALS:
        return "F"
    if v in MALE_VALS:
        return "M"
    return np.nan


def apply_iso2_to_iso3(series: pd.Series) -> pd.Series:
    """Converte codici ISO2 in ISO3. Valori già ISO3 passano invariati."""
    return series.str.strip().str.upper().replace(ISO2_TO_ISO3)


def drop_sparse_columns(df: pd.DataFrame, threshold: float = NULL_DROP_THRESHOLD) -> pd.DataFrame:
    """Rimuove colonne con percentuale di null superiore alla soglia."""
    to_drop = [c for c in df.columns if df[c].isna().mean() > threshold]
    if to_drop:
        print(f"  Colonne eliminate (>{threshold*100:.0f}% null): {to_drop}")
    return df.drop(columns=to_drop)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE DI PULIZIA ALLARMI
# ══════════════════════════════════════════════════════════════════════════════

def clean_allarmi(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Pulizia ALLARMI ──────────────────────────────────────")

    # 1. Rimuovi colonne duplicate
    cols_to_drop = [c for c in COLS_DROP_ALLARMI if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  Rimosse {len(cols_to_drop)} colonne duplicate")

    # 2. Sostituisci tutti i valori NULL con NaN
    df = df.replace(NULL_VALUES, np.nan)

    # 3. ANNO_PARTENZA
    df["ANNO_PARTENZA"] = df["ANNO_PARTENZA"].replace(ANNO_CORRECTIONS)
    df["ANNO_PARTENZA"] = pd.to_numeric(df["ANNO_PARTENZA"], errors="coerce")

    # 4. MESE_PARTENZA: testo italiano → numerico
    df["MESE_PARTENZA"] = df["MESE_PARTENZA"].replace(ITALIAN_MONTHS)
    df["MESE_PARTENZA"] = pd.to_numeric(df["MESE_PARTENZA"], errors="coerce").astype("Int64")

    # 5. DATA_PARTENZA con parser robusto
    n_before = df["DATA_PARTENZA"].notna().sum()
    df["DATA_PARTENZA"] = df["DATA_PARTENZA"].apply(parse_date)
    n_after = df["DATA_PARTENZA"].notna().sum()
    print(f"  DATA_PARTENZA parsed: {n_after}/{n_before} validi")

    # 6. Overwrite ANNO e MESE da DATA_PARTENZA (più affidabile)
    mask = df["DATA_PARTENZA"].notna()
    df.loc[mask, "ANNO_PARTENZA"] = df.loc[mask, "DATA_PARTENZA"].dt.year
    df.loc[mask, "MESE_PARTENZA"] = df.loc[mask, "DATA_PARTENZA"].dt.month

    # 7. TOT: estrai numero, rimuovi negativi, non-interi e placeholder >9999
    df["TOT"] = df["TOT"].apply(extract_number)
    df.loc[df["TOT"] < 0, "TOT"] = np.nan
    df.loc[df["TOT"] != df["TOT"].round(), "TOT"] = np.nan
    df.loc[df["TOT"] > 9999, "TOT"] = np.nan

    # 8. ZONA
    df["ZONA"] = df["ZONA"].astype(str).str.strip()
    df.loc[~df["ZONA"].isin(ZONE_VALIDE), "ZONA"] = np.nan
    df["ZONA"] = pd.to_numeric(df["ZONA"], errors="coerce").astype("Int64")

    # 9. OCCORRENZE: valori non validi → NaN
    df["OCCORRENZE"] = df["OCCORRENZE"].replace({v: np.nan for v in OCC_INVALID})

    # 10. Codici aeroporto: strip + uppercase (fix bug merge)
    for col in ["AREOPORTO_ARRIVO", "AREOPORTO_PARTENZA"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()

    # 11. Codici paese: ISO2 → ISO3 completa
    for col in ["CODICE_PAESE_ARR", "CODICE_PAESE_PART"]:
        if col in df.columns:
            df[col] = apply_iso2_to_iso3(df[col])

    # 12. Strip su colonne stringa
    for col in ["PAESE_ARR", "PAESE_PART", "MOTIVO_ALLARME"]:
        if col in df.columns:
            df[col] = df[col].str.strip()

    # 13. Colonne temporali derivate da DATA_PARTENZA
    df["ora_partenza"]     = df["DATA_PARTENZA"].dt.hour
    df["giorno_settimana"] = df["DATA_PARTENZA"].dt.dayofweek
    df["mese"]             = df["DATA_PARTENZA"].dt.month

    # 14. Drop colonne sparse (>50% null)
    df = drop_sparse_columns(df)

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
    print(f"  Rimosse {len(cols_to_drop)} colonne duplicate")

    # 2. Sostituisci tutti i valori NULL con NaN
    df = df.replace(NULL_VALUES, np.nan)

    # 3. ANNO_PARTENZA
    df["ANNO_PARTENZA"] = df["ANNO_PARTENZA"].replace(ANNO_CORRECTIONS)
    df["ANNO_PARTENZA"] = pd.to_numeric(df["ANNO_PARTENZA"], errors="coerce")

    # 4. MESE_PARTENZA: testo italiano → numerico
    df["MESE_PARTENZA"] = df["MESE_PARTENZA"].replace(ITALIAN_MONTHS)
    df["MESE_PARTENZA"] = pd.to_numeric(df["MESE_PARTENZA"], errors="coerce").astype("Int64")

    # 5. DATA_PARTENZA con parser robusto
    n_before = df["DATA_PARTENZA"].notna().sum()
    df["DATA_PARTENZA"] = df["DATA_PARTENZA"].apply(parse_date)
    n_after = df["DATA_PARTENZA"].notna().sum()
    print(f"  DATA_PARTENZA parsed: {n_after}/{n_before} validi")

    # 6. Overwrite ANNO e MESE da DATA_PARTENZA
    mask = df["DATA_PARTENZA"].notna()
    df.loc[mask, "ANNO_PARTENZA"] = df.loc[mask, "DATA_PARTENZA"].dt.year
    df.loc[mask, "MESE_PARTENZA"] = df.loc[mask, "DATA_PARTENZA"].dt.month

    # 7. ENTRATI, INVESTIGATI, ALLARMATI: estrai numero, rimuovi negativi
    for col in ["ENTRATI", "INVESTIGATI", "ALLARMATI"]:
        if col in df.columns:
            df[col] = df[col].apply(extract_number)
            df.loc[df[col] < 0, col] = np.nan

    # 8. Vincoli di dominio: INVESTIGATI ≤ ENTRATI, ALLARMATI ≤ ENTRATI
    if all(c in df.columns for c in ["ENTRATI", "INVESTIGATI", "ALLARMATI"]):
        df.loc[df["INVESTIGATI"] > df["ENTRATI"], "INVESTIGATI"] = np.nan
        df.loc[df["ALLARMATI"]   > df["ENTRATI"], "ALLARMATI"]   = np.nan

    # 9. GENERE: conservativo, '1'/'2' → NaN
    before = df["GENERE"].value_counts().nlargest(3).to_dict()
    df["GENERE"] = df["GENERE"].apply(normalize_gender)
    print(f"  GENERE: {before} → {df['GENERE'].value_counts(dropna=False).to_dict()}")

    # 10. TIPO_DOCUMENTO
    tipo_doc_valid = {"Passaporto", "Carta d'identità", "Visto", "Permesso di soggiorno"}
    df["TIPO_DOCUMENTO"] = df["TIPO_DOCUMENTO"].where(
        df["TIPO_DOCUMENTO"].isin(tipo_doc_valid), other=np.nan
    )

    # 11. FASCIA_ETA: recupera label testuali, invalida il resto
    df["FASCIA_ETA"] = df["FASCIA_ETA"].replace(FASCIA_MAP)
    df["FASCIA_ETA"] = df["FASCIA_ETA"].where(
        df["FASCIA_ETA"].isin(FASCIA_ETA_VALID), other=np.nan
    )

    # 12. ZONA
    df["ZONA"] = df["ZONA"].astype(str).str.strip()
    df.loc[~df["ZONA"].isin(ZONE_VALIDE), "ZONA"] = np.nan
    df["ZONA"] = pd.to_numeric(df["ZONA"], errors="coerce").astype("Int64")

    # 13. Codici aeroporto: strip + uppercase
    for col in ["AREOPORTO_ARRIVO", "AREOPORTO_PARTENZA"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()

    # 14. Codici paese: ISO2 → ISO3
    for col in ["CODICE_PAESE_ARR", "CODICE_PAESE_PART"]:
        if col in df.columns:
            df[col] = apply_iso2_to_iso3(df[col])

    # 15. NAZIONALITA: ISO2 → ISO3, poi invalida tutto ciò che non è 3 lettere
    if "NAZIONALITA" in df.columns:
        df["NAZIONALITA"] = df["NAZIONALITA"].str.strip().str.upper().replace(ISO2_TO_ISO3)
        df.loc[df["NAZIONALITA"].str.len() != 3, "NAZIONALITA"] = np.nan

    # 16. Feature derivate
    for col in ["ENTRATI", "ALLARMATI", "INVESTIGATI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["tasso_allarme"]     = np.where(df["ENTRATI"] > 0, df["ALLARMATI"]    / df["ENTRATI"], 0.0)
    df["tasso_investigati"] = np.where(df["ENTRATI"] > 0, df["INVESTIGATI"]  / df["ENTRATI"], 0.0)

    # 17. Colonne temporali
    df["ora_partenza"]     = df["DATA_PARTENZA"].dt.hour
    df["giorno_settimana"] = df["DATA_PARTENZA"].dt.dayofweek
    df["mese"]             = df["DATA_PARTENZA"].dt.month

    # 18. Drop colonne sparse
    df = drop_sparse_columns(df)

    print(f"  Shape finale: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MERGE
# ══════════════════════════════════════════════════════════════════════════════

def merge_datasets(df_allarmi: pd.DataFrame, df_viaggiatori: pd.DataFrame) -> pd.DataFrame:
    """
    Join tra ALLARMI e TIPOLOGIA_VIAGGIATORE su:
    AREOPORTO_ARRIVO + AREOPORTO_PARTENZA + DATA_PARTENZA (solo data).
    Left join: mantiene tutti i record di ALLARMI.
    """
    print("\n── Merge dataset ────────────────────────────────────────")

    df_allarmi     = df_allarmi.copy()
    df_viaggiatori = df_viaggiatori.copy()

    df_allarmi["_data_key"]     = pd.to_datetime(df_allarmi["DATA_PARTENZA"]).dt.date
    df_viaggiatori["_data_key"] = pd.to_datetime(df_viaggiatori["DATA_PARTENZA"]).dt.date

    join_keys = ["AREOPORTO_ARRIVO", "AREOPORTO_PARTENZA", "_data_key"]

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

    for _df in [merged, df_allarmi, df_viaggiatori]:
        if "_data_key" in _df.columns:
            _df.drop(columns=["_data_key"], inplace=True)

    print(f"  Righe ALLARMI:     {len(df_allarmi)}")
    print(f"  Righe VIAGGIATORI: {len(df_viaggiatori)}")
    print(f"  Righe dopo merge:  {len(merged)}")
    print(f"  Match trovati:     {merged['tot_entrati'].notna().sum()}/{len(merged)}")

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

def run_preprocessing() -> tuple:
    """
    Esegue l'intera pipeline di preprocessing.
    Ritorna (df_allarmi_clean, df_viaggiatori_clean, df_merged).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Caricamento dataset...")
    df_allarmi     = load_csv(ALLARMI_PATH)
    df_viaggiatori = load_csv(VIAGGIATORI_PATH)

    df_allarmi     = clean_allarmi(df_allarmi)
    df_viaggiatori = clean_viaggiatori(df_viaggiatori)
    df_merged      = merge_datasets(df_allarmi, df_viaggiatori)

    print_quality_report(df_allarmi,     "ALLARMI clean")
    print_quality_report(df_viaggiatori, "VIAGGIATORI clean")
    print_quality_report(df_merged,      "MERGED")

    df_allarmi.to_csv(PROCESSED_DIR / "allarmi_clean.csv",         index=False)
    df_viaggiatori.to_csv(PROCESSED_DIR / "viaggiatori_clean.csv", index=False)
    df_merged.to_csv(PROCESSED_DIR / "dataset_merged.csv",         index=False)

    print(f"\nFile salvati in data/processed/")
    return df_allarmi, df_viaggiatori, df_merged


if __name__ == "__main__":
    run_preprocessing()
