"""
features.py
───────────
Classi di feature engineering per la pipeline multi-agent.

Ogni classe incapsula un gruppo logico di trasformazioni, specchiando
esattamente i notebook della pipeline classica (02_feature_engineering.ipynb).
Questo garantisce un confronto onesto: stessa logica, architettura diversa.

Classi:
    OccurrencePivot         — pivot del tipo di OCCORRENZA per rotta
    MotivoAllarmeFeatures   — percentuali per motivo allarme (INTERPOL, SDI…)
    AllarmiAggregator       — metadati + feature derivate da ALLARMI
    ViaggiatoriAggregator   — aggregazioni da TIPOLOGIA_VIAGGIATORE
    EsitiPivot              — pivot di ESITO_CONTROLLO + tassi di rischio
    FeatureBuilder          — orchestratore: combina tutte le classi sopra

Uso:
    builder = FeatureBuilder()
    df_features = builder.build(df_allarmi, df_viaggiatori)
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def safe_div(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Divisione vettorizzata sicura: restituisce 0.0 dove b == 0."""
    return np.where(b > 0, a / b, 0.0)


def safe_mode(x: pd.Series):
    """Moda di una serie; restituisce 'ND' se la serie è vuota."""
    m = x.mode()
    return m.iloc[0] if len(m) > 0 else "ND"


# ══════════════════════════════════════════════════════════════════════════════
# MAPPATURE (identiche al notebook classico)
# ══════════════════════════════════════════════════════════════════════════════

RENAME_OCC = {
    "Viaggiatori entrati nel Sistema"              : "vg_entrati_occ",
    "Viaggiatori con Allarmi"                      : "vg_con_allarmi",
    "Viaggiatori investigati"                      : "vg_investigati_occ",
    "Voli con Allarmi"                             : "voli_con_allarmi",
    "Voli disponibili in ingresso al Sistema"      : "voli_disponibili",
    "Voli investigati (SDI/NSIS - INTERPOL - TSC)" : "voli_investigati",
    "Voli solo visualizzati, ma NON investigati"   : "voli_non_investigati",
    "Allarmi generati"                             : "allarmi_generati",
    "Allarmi generati da SDI/NSIS"                 : "allarmi_sdi_occ",
    "Allarmi generati da INTERPOL"                 : "allarmi_interpol_occ",
    "Allarmi generati da BCS"                      : "allarmi_bcs_occ",
    "Allarmi Chiusi"                               : "allarmi_chiusi",
    "Allarmi Chiusi con Azione (CC.xx)"            : "allarmi_chiusi_azione",
    "Allarmi NON Chiusi"                           : "allarmi_non_chiusi",
    "Allarmi Rilevanti"                            : "allarmi_rilevanti",
    "Respinto/a"                                   : "vg_respinti_occ",
    "Errata segnalazione SDI"                      : "err_sdi",
    "Errata segnalazione NSIS"                     : "err_nsis",
    "Errata segnalazione BCS"                      : "err_bcs",
    "Nulla a procedere SDI"                        : "np_sdi",
    "Nulla a procedere NSIS"                       : "np_nsis",
    "Nulla a procedere INT"                        : "np_int",
    "Notifica Atti/Provv"                          : "notifica_atti",
    "Mancato aggiornamento SDI"                    : "mancato_agg_sdi",
    "Mancato aggiornamento Schengen NSIS"          : "mancato_agg_nsis",
    "Inammissibilita Schengen - Art.24"            : "inammissib_schengen",
    "ALLARMATI"                                    : "allarmati_occ",
    "Altro"                                        : "altro_occ",
    "N/C"                                          : "nc_occ",
    "???"                                          : "unknown_occ",
}

RENAME_ESITI = {
    "SEGNALATO" : "n_segnalati",
    "IN ATTESA"  : "n_in_attesa",
    "RESPINTO"   : "n_respinti",
    "FERMATO"    : "n_fermati",
    "OK"         : "n_ok",
}


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE 1 — OccurrencePivot
# ══════════════════════════════════════════════════════════════════════════════

class OccurrencePivot:
    """
    Crea un pivot del dataset ALLARMI: una colonna per ogni tipo di
    OCCORRENZA, con valore = somma di TOT per rotta.

    Input:  df_allarmi con colonne [ROTTA, OCCORRENZE, TOT]
    Output: DataFrame con ROTTA come indice e ~30 colonne numeriche
    """

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pivot = (
            df.pivot_table(
                index      = "ROTTA",
                columns    = "OCCORRENZE",
                values     = "TOT",
                aggfunc    = "sum",
                fill_value = 0,
            )
            .reset_index()
        )
        pivot.columns.name = None
        pivot = pivot.rename(columns=RENAME_OCC)

        # Clip negativi (np_sdi e voli_non_investigati hanno TOT < 0 nel raw)
        num_cols = pivot.select_dtypes(include="number").columns
        pivot[num_cols] = pivot[num_cols].clip(lower=0)

        return pivot


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE 2 — MotivoAllarmeFeatures
# ══════════════════════════════════════════════════════════════════════════════

class MotivoAllarmeFeatures:
    """
    Calcola le percentuali di allarme per motivo (INTERPOL, SDI, NSIS,
    TSC, Manuale) per ogni rotta.

    Input:  df_allarmi con colonne [ROTTA, MOTIVO_ALLARME]
    Output: DataFrame con ROTTA + colonne pct_interpol, pct_sdi, pct_nsis,
            pct_tsc, pct_manuale
    """

    MOTIVI = ["INTERPOL", "SDI", "NSIS", "TSC", "Manuale"]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        counts = (
            df.groupby("ROTTA")["MOTIVO_ALLARME"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        counts.columns.name = None

        totale = counts.drop(columns=["ROTTA"]).sum(axis=1)

        for motivo in self.MOTIVI:
            col_pct = f"pct_{motivo.lower()}"
            if motivo in counts.columns:
                counts[col_pct] = safe_div(counts[motivo], totale).clip(0, 1)
                counts = counts.drop(columns=[motivo])
            else:
                counts[col_pct] = 0.0

        # Tieni solo ROTTA e le colonne pct_
        pct_cols = [c for c in counts.columns if c.startswith("pct_") or c == "ROTTA"]
        return counts[pct_cols].fillna(0)


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE 3 — AllarmiAggregator
# ══════════════════════════════════════════════════════════════════════════════

class AllarmiAggregator:
    """
    Aggrega il dataset ALLARMI per rotta, combinando:
    - Metadati (ZONA, PAESE_PART, n_osservazioni)
    - OccurrencePivot
    - MotivoAllarmeFeatures
    - Feature derivate: tasso_chiusura, tasso_rilevanza, tot_allarmi_log, false_positive_rate

    Input:  df_allarmi pulito da preprocessing.py
    Output: DataFrame aggregato per ROTTA (una riga per rotta)
    """

    def __init__(self):
        self._occ_pivot    = OccurrencePivot()
        self._motivo_feat  = MotivoAllarmeFeatures()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Metadati base
        meta = df.groupby("ROTTA").agg(
            ZONA                   = ("ZONA",       "first"),
            PAESE_PART             = ("PAESE_PART",  "first"),
            n_osservazioni_allarmi = ("TOT",         "count"),
            tot_allarmi_sum        = ("TOT",         "sum"),
        ).reset_index()

        # Log-transform: distribuzione skewed (max ~103k)
        meta["tot_allarmi_log"] = np.log1p(meta["tot_allarmi_sum"])

        # 2. Pivot occorrenze + motivi
        occ   = self._occ_pivot.fit_transform(df)
        motiv = self._motivo_feat.fit_transform(df)

        # 3. Merge
        agg = meta.merge(occ,   on="ROTTA", how="left")
        agg = agg.merge(motiv,  on="ROTTA", how="left")

        # 4. Clip residui negativi dopo merge
        num_cols = agg.select_dtypes(include="number").columns
        agg[num_cols] = agg[num_cols].clip(lower=0)

        # 5. Feature derivate
        agg["tasso_chiusura"] = safe_div(
            agg["allarmi_chiusi"],
            agg["allarmi_chiusi"] + agg.get("allarmi_non_chiusi", pd.Series(0, index=agg.index))
        ).clip(0, 1)

        agg["tasso_rilevanza"] = safe_div(
            agg.get("allarmi_rilevanti", pd.Series(0, index=agg.index)),
            agg.get("voli_con_allarmi",  pd.Series(0, index=agg.index))
        ).clip(0, 1)

        # 6. false_positive_rate — integrata dal notebook del collega
        #    (np_sdi + np_nsis + np_int) / (allarmi_sdi + allarmi_interpol)
        numeratore_fp = (
            agg.get("np_sdi",               pd.Series(0, index=agg.index)) +
            agg.get("np_nsis",              pd.Series(0, index=agg.index)) +
            agg.get("np_int",               pd.Series(0, index=agg.index))
        )
        denominatore_fp = (
            agg.get("allarmi_sdi_occ",      pd.Series(0, index=agg.index)) +
            agg.get("allarmi_interpol_occ", pd.Series(0, index=agg.index))
        )
        agg["false_positive_rate"] = safe_div(numeratore_fp, denominatore_fp).clip(0, 1)

        # Fill NaN nelle pct_
        pct_cols = [c for c in agg.columns if c.startswith("pct_")]
        agg[pct_cols] = agg[pct_cols].fillna(0)

        return agg


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE 4 — EsitiPivot
# ══════════════════════════════════════════════════════════════════════════════

class EsitiPivot:
    """
    Crea un pivot di ESITO_CONTROLLO per rotta e calcola i tassi di rischio.

    Input:  df_viaggiatori con colonne [ROTTA, ESITO_CONTROLLO, ENTRATI]
    Output: DataFrame con ROTTA + n_segnalati, n_respinti, n_fermati,
            n_in_attesa, n_ok, tasso_respinti, tasso_fermati, score_rischio_esiti
    """

    def fit_transform(self, df: pd.DataFrame, n_osservazioni: pd.Series = None) -> pd.DataFrame:
        pivot = (
            df.pivot_table(
                index      = "ROTTA",
                columns    = "ESITO_CONTROLLO",
                values     = "ENTRATI",
                aggfunc    = "count",
                fill_value = 0,
            )
            .reset_index()
        )
        pivot.columns.name = None
        pivot = pivot.rename(columns=RENAME_ESITI)

        # Assicura colonne presenti anche se mancanti nel dataset
        for col in ["n_segnalati", "n_in_attesa", "n_respinti", "n_fermati", "n_ok"]:
            if col not in pivot.columns:
                pivot[col] = 0
            pivot[col] = pivot[col].fillna(0).astype(int)

        # Tassi (su n_osservazioni se fornito, altrimenti su totale esiti)
        totale = pivot[["n_segnalati", "n_in_attesa", "n_respinti", "n_fermati", "n_ok"]].sum(axis=1)
        denom  = n_osservazioni if n_osservazioni is not None else totale

        pivot["tasso_respinti"] = safe_div(pivot["n_respinti"], denom).clip(0, 1)
        pivot["tasso_fermati"]  = safe_div(pivot["n_fermati"],  denom).clip(0, 1)

        # Score rischio esiti: respinto 60% + fermato 40%
        pivot["score_rischio_esiti"] = (
            pivot["tasso_respinti"] * 0.6 +
            pivot["tasso_fermati"]  * 0.4
        ).clip(0, 1)

        return pivot


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE 5 — ViaggiatoriAggregator
# ══════════════════════════════════════════════════════════════════════════════

class ViaggiatoriAggregator:
    """
    Aggrega il dataset TIPOLOGIA_VIAGGIATORE per rotta, combinando:
    - Conteggi e tassi di allarme/investigazione
    - Profilo demografico predominante
    - EsitiPivot (con tassi di rischio)
    - alarm_per_invest: tot_allarmati/tot_investigati, capped al p99

    Input:  df_viaggiatori pulito da preprocessing.py
    Output: DataFrame aggregato per ROTTA (una riga per rotta)
    """

    def __init__(self):
        self._esiti_pivot = EsitiPivot()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Aggregazioni base
        agg = df.groupby("ROTTA").agg(
            tot_entrati             = ("ENTRATI",          "sum"),
            tot_allarmati           = ("ALLARMATI",         "sum"),
            tot_investigati         = ("INVESTIGATI",       "sum"),
            n_osservazioni_viag     = ("ENTRATI",           "count"),
            tasso_allarme_medio     = ("tasso_allarme",     "mean"),
            tasso_inv_medio         = ("tasso_investigati", "mean"),
            genere_predominante     = ("GENERE",            safe_mode),
            fascia_eta_predominante = ("FASCIA_ETA",        safe_mode),
            tipo_doc_prevalente     = ("TIPO_DOCUMENTO",    safe_mode),
            nazionalita_top         = ("NAZIONALITA",       safe_mode),
            compagnia_top           = ("COMPAGNIA_AEREA",   safe_mode),
        ).reset_index()

        # 2. Clip e cap
        agg["tot_entrati"]         = agg["tot_entrati"].clip(lower=0)
        agg["tot_allarmati"]       = agg["tot_allarmati"].clip(lower=0)
        agg["tot_investigati"]     = agg["tot_investigati"].clip(lower=0)
        agg["tasso_allarme_medio"] = agg["tasso_allarme_medio"].clip(0, 1)
        agg["tasso_inv_medio"]     = agg["tasso_inv_medio"].clip(0, 1)

        # alarm_per_invest — integrata dal notebook del collega
        #   tot_allarmati / tot_investigati, capped al p99 per outlier estremi
        agg["alarm_per_invest"] = safe_div(agg["tot_allarmati"], agg["tot_investigati"])
        cap_p99 = agg["alarm_per_invest"].quantile(0.99)
        agg["alarm_per_invest"] = agg["alarm_per_invest"].clip(upper=cap_p99)

        # 3. Merge con esiti pivot
        esiti = self._esiti_pivot.fit_transform(df, n_osservazioni=None)

        # Allinea n_osservazioni per i tassi
        esiti = esiti.merge(agg[["ROTTA", "n_osservazioni_viag"]], on="ROTTA", how="left")
        esiti["tasso_respinti"] = safe_div(esiti["n_respinti"], esiti["n_osservazioni_viag"]).clip(0, 1)
        esiti["tasso_fermati"]  = safe_div(esiti["n_fermati"],  esiti["n_osservazioni_viag"]).clip(0, 1)
        esiti["score_rischio_esiti"] = (
            esiti["tasso_respinti"] * 0.6 + esiti["tasso_fermati"] * 0.4
        ).clip(0, 1)
        esiti = esiti.drop(columns=["n_osservazioni_viag"])

        agg = agg.merge(esiti, on="ROTTA", how="left")

        return agg


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE 6 — FeatureBuilder  (orchestratore principale)
# ══════════════════════════════════════════════════════════════════════════════

class FeatureBuilder:
    """
    Orchestratore principale: combina AllarmiAggregator e ViaggiatoriAggregator,
    calcola lo score_composito finale e produce il DataFrame pronto per i modelli.

    Uso:
        builder = FeatureBuilder()
        df_features = builder.build(df_allarmi, df_viaggiatori)

    Output identico a features_classical.csv prodotto dal notebook classico.
    """

    # Pesi score_composito — identici al notebook 02
    W_ALLARMI_LOG   = 0.35
    W_RISCHIO_ESITI = 0.35
    W_INTERPOL      = 0.15
    W_RILEVANZA     = 0.15

    def __init__(self):
        self._allarmi_agg    = AllarmiAggregator()
        self._viaggiatori_agg = ViaggiatoriAggregator()

    def build(self, df_allarmi: pd.DataFrame, df_viaggiatori: pd.DataFrame) -> pd.DataFrame:
        """
        Esegue l'intero feature engineering.

        Args:
            df_allarmi:     output di clean_allarmi() da preprocessing.py
            df_viaggiatori: output di clean_viaggiatori() da preprocessing.py

        Returns:
            DataFrame con una riga per ROTTA e tutte le feature numeriche
            + score_composito
        """
        # 1. Aggiungi colonna ROTTA a entrambi i dataset
        df_allarmi     = self._add_rotta(df_allarmi)
        df_viaggiatori = self._add_rotta(df_viaggiatori)

        # 2. Aggrega separatamente
        agg_a = self._allarmi_agg.fit_transform(df_allarmi)
        agg_v = self._viaggiatori_agg.fit_transform(df_viaggiatori)

        # 3. Outer join: mantiene rotte presenti in almeno uno dei due dataset
        features = agg_a.merge(agg_v, on="ROTTA", how="outer")

        # 4. Fix PAESE_PART e ZONA per rotte solo in VIAGGIATORI
        features = self._fix_paese_zona(features, df_viaggiatori)

        # 5. Fillna: numerici → 0, categorici → "ND"
        features = self._fillna(features)

        # 6. Score composito
        features = self._add_score_composito(features)

        return features

    # ── metodi privati ─────────────────────────────────────────────────────────

    @staticmethod
    def _add_rotta(df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge la colonna ROTTA = AREOPORTO_PARTENZA-AREOPORTO_ARRIVO."""
        if "ROTTA" not in df.columns:
            df = df.copy()
            df["ROTTA"] = (
                df["AREOPORTO_PARTENZA"].str.upper() + "-" +
                df["AREOPORTO_ARRIVO"].str.upper()
            )
        return df

    @staticmethod
    def _fix_paese_zona(features: pd.DataFrame, df_viaggiatori: pd.DataFrame) -> pd.DataFrame:
        """
        Le rotte presenti solo in VIAGGIATORI hanno PAESE_PART e ZONA a NaN.
        Le recupera dal dataset viaggiatori come fallback.
        """
        fallback = (
            df_viaggiatori.groupby("ROTTA")
            .agg(
                PAESE_PART_viag = ("PAESE_PART", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "ND"),
                ZONA_viag       = ("ZONA",       lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "ND"),
            )
            .reset_index()
        )
        features = features.merge(fallback, on="ROTTA", how="left")

        features["PAESE_PART"] = (
            features["PAESE_PART"]
            .replace(["ND", "//", ""], None)
            .fillna(features["PAESE_PART_viag"])
            .fillna("ND")
        )
        features["ZONA"] = (
            features["ZONA"]
            .replace(["ND", "//", ""], None)
            .fillna(features["ZONA_viag"])
            .fillna("ND")
        )
        return features.drop(columns=["PAESE_PART_viag", "ZONA_viag"])

    @staticmethod
    def _fillna(features: pd.DataFrame) -> pd.DataFrame:
        num_cols = features.select_dtypes(include="number").columns
        features[num_cols] = features[num_cols].fillna(0)

        cat_cols = features.select_dtypes(include="object").columns.drop("ROTTA", errors="ignore")
        for col in cat_cols:
            features[col] = features[col].fillna("ND")

        return features

    def _add_score_composito(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Score composito [0,1] — identico al notebook classico:
            35% tot_allarmi_log normalizzato
            35% score_rischio_esiti
            15% pct_interpol
            15% tasso_rilevanza
        """
        log_max = features["tot_allarmi_log"].max()

        features["score_composito"] = (
            (features["tot_allarmi_log"] / max(log_max, 1)) * self.W_ALLARMI_LOG +
            features["score_rischio_esiti"]                 * self.W_RISCHIO_ESITI +
            features["pct_interpol"]                        * self.W_INTERPOL +
            features["tasso_rilevanza"]                     * self.W_RILEVANZA
        ).clip(0, 1)

        return features

    def quality_report(self, features: pd.DataFrame) -> dict:
        """Restituisce un dizionario con le statistiche di qualità del DataFrame."""
        num = features.select_dtypes(include="number")
        return {
            "n_rotte"      : len(features),
            "n_features"   : len(num.columns),
            "null_totali"  : int(features.isnull().sum().sum()),
            "negativi"     : {c: int((features[c] < 0).sum()) for c in num.columns if (features[c] < 0).any()},
            "score_composito_stats": {
                "mean" : round(float(features["score_composito"].mean()), 4),
                "max"  : round(float(features["score_composito"].max()),  4),
                "min"  : round(float(features["score_composito"].min()),  4),
            },
            "copertura": {
                "entrambi_dataset" : int(((features["n_osservazioni_allarmi"] > 0) & (features["n_osservazioni_viag"] > 0)).sum()),
                "solo_allarmi"     : int(((features["n_osservazioni_allarmi"] > 0) & (features["n_osservazioni_viag"] == 0)).sum()),
                "solo_viaggiatori" : int(((features["n_osservazioni_allarmi"] == 0) & (features["n_osservazioni_viag"] > 0)).sum()),
            }
        }
